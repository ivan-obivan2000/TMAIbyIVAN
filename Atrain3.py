import glob
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from torch.utils.data import Dataset, DataLoader

# ==========================
# КОНФИГУРАЦИЯ
# ==========================
LOG_DIR = "tmi_logs3"
K = 30
BATCH = 1024
EPOCHS = 80
LR = 1e-3
MODEL_NAME = "policy_trajectory_v4.pt"

# Сколько лучших заездов брать
N_BEST_EPISODES = 7

# Lookahead: на сколько кадров вперед смотреть?
# 10 кадров * 20мс = 0.2 сек. Это сглаживает управление.
LOOKAHEAD_STEPS = 10

# Глобальные переменные
GHOST_DATA = None     # [x, y, z, vx, vy, vz]
GHOST_TREE = None     # KDTree

# ==========================
# 1. ЗАГРУЗКА И СОЗДАНИЕ МУЛЬТИ-ПРИЗРАКА
# ==========================
def load_episodes_and_build_ghost():
    global GHOST_DATA, GHOST_TREE
    
    files = sorted(glob.glob(f"{LOG_DIR}/*.jsonl"))
    print(f"Поиск файлов в {LOG_DIR}...")
    
    raw_episodes = []
    
    # --- Чтение ---
    for fp in files:
        current_ep = []
        last_ep_id = -1
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        ep_id = rec.get("episode_id", -1) 
                        if ep_id != last_ep_id:
                            if len(current_ep) > K: raw_episodes.append(current_ep)
                            current_ep = []
                            last_ep_id = ep_id
                        current_ep.append(rec)
                    except: pass
                if len(current_ep) > K: raw_episodes.append(current_ep)
        except: pass

    # --- Фильтрация ---
    finished_eps = []
    for ep in raw_episodes:
        if any([r.get("finished", 0) for r in ep[-20:]]):
            duration = ep[-1]["time_ms"] - ep[0]["time_ms"]
            finished_eps.append({"data": ep, "time": duration})

    if not finished_eps:
        print("!!! НЕТ ФИНИШЕЙ. Обучение невозможно.")
        return [], []

    # Сортировка: от быстрых к медленным
    finished_eps.sort(key=lambda x: x["time"])
    
    # Берем ТОП-N
    top_n_count = min(len(finished_eps), N_BEST_EPISODES)
    best_runs = finished_eps[:top_n_count]
    
    print(f"Формирование траектории из Топ-{top_n_count} заездов.")
    print(f"Лучшее время: {best_runs[0]['time']/1000:.2f} сек")
    print(f"Худшее из топа: {best_runs[-1]['time']/1000:.2f} сек")

    # --- СОЗДАЕМ 6D ОБЛАКО ТОЧЕК ---
    ghost_points = []
    for run in best_runs:
        for r in run["data"]:
            p = r.get("position", [0,0,0])
            v = r.get("velocity", [0,0,0])
            ghost_points.append([p[0], p[1], p[2], v[0], v[1], v[2]])
            
    GHOST_DATA = np.array(ghost_points, dtype=np.float32)
    print(f"Строим KDTree на {len(GHOST_DATA)} точках (6D)...")
    GHOST_TREE = KDTree(GHOST_DATA)

    # --- ВЕСА ---
    final_episodes = [item["data"] for item in best_runs]
    
    # Если заездов > 1, делаем градиент весов от 2.0 до 1.0
    if len(final_episodes) > 1:
        weights = np.linspace(2.0, 1.0, len(final_episodes))
    else:
        weights = np.array([1.0])
    
    return final_episodes, weights

# ==========================
# ВЕКТОРИЗАЦИЯ (6D + LOOKAHEAD)
# ==========================
def obs_vec(r):
    p = np.array(r.get("position", [0,0,0]), dtype=np.float32)
    v = np.array(r.get("velocity", [0,0,0]), dtype=np.float32)
    
    query_point = np.concatenate([p, v])
    dist_6d, idx = GHOST_TREE.query(query_point)
    
    # === LOOKAHEAD ===
    # Смещаем индекс вперед, чтобы смотреть в будущее
    # Ограничиваем, чтобы не выйти за пределы массива
    target_idx = min(idx + LOOKAHEAD_STEPS, len(GHOST_DATA) - 1)
    
    target_all = GHOST_DATA[target_idx]
    target_p = target_all[:3]
    target_v = target_all[3:]
    
    delta_p = target_p - p
    delta_v = target_v - v 

    # Остальные данные
    car = r.get("car") or {}
    eng = r.get("engine") or {}
    dyna = r.get("dyna") or {}
    quat = dyna.get("quat") or [0, 0, 0, 1] 
    ang_vel = dyna.get("angular_speed") or [0, 0, 0]
    loc_v = car.get("current_local_speed") or [0, 0, 0]

    feat = [
        # Ошибки (относительно точки впереди!)
        delta_p[0], delta_p[1], delta_p[2],
        delta_v[0], delta_v[1], delta_v[2],
        
        # Абсолютная физика
        v[0], v[1], v[2],
        r.get("speed_norm", 0) / 100.0,
        
        quat[0], quat[1], quat[2], quat[3],
        ang_vel[0], ang_vel[1], ang_vel[2],

        loc_v[0], loc_v[1], loc_v[2],
        car.get("turning_rate", 0.0),
        float(car.get("is_sliding", False)),
        eng.get("actual_rpm", 0.0) / 10000.0,
        float(eng.get("gear", 1)) / 5.0,
    ]
    return np.array(feat, dtype=np.float32)

def action_vec(r):
    inputs = r.get("inputs", {})
    steer = 0.0
    if inputs.get("left"): steer = -1.0
    elif inputs.get("right"): steer = 1.0
    throttle = 1.0 if inputs.get("accelerate") else 0.0
    brake = 1.0 if inputs.get("brake") else 0.0
    return np.array([steer, throttle, brake], dtype=np.float32)

# ==========================
# DATASET
# ==========================
class TMWeightedDataset(Dataset):
    def __init__(self, episodes, weights, K):
        self.X = []
        self.Y = []
        self.W = [] 

        for ep, weight in zip(episodes, weights):
            if not ep: continue
            
            processed_obs = []
            processed_act = []
            
            for r in ep:
                processed_obs.append(obs_vec(r))
                processed_act.append(action_vec(r))
            
            for t in range(K, len(ep)):
                x = np.concatenate(processed_obs[t-K:t], axis=0)
                y = processed_act[t]
                
                self.X.append(x)
                self.Y.append(y)
                self.W.append(weight)

        self.X = np.stack(self.X)
        self.Y = np.stack(self.Y)
        self.W = np.array(self.W, dtype=np.float32)

        print("Вычисление нормализации...")
        self.mean = self.X.mean(axis=0, keepdims=True)
        self.std = self.X.std(axis=0, keepdims=True) + 1e-6
        self.X = (self.X - self.mean) / self.std
        
        print(f"Dataset ready: {len(self.X)} samples.")

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i]), torch.tensor(self.W[i])

# ==========================
# МОДЕЛЬ
# ==========================
class PolicyNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head_steer = nn.Linear(256, 1)
        self.head_tb = nn.Linear(256, 2)

    def forward(self, x):
        h = self.net(x)
        steer = torch.tanh(self.head_steer(h))
        tb = torch.sigmoid(self.head_tb(h))
        return steer, tb

# ==========================
# TRAIN
# ==========================
def train():
    episodes, weights = load_episodes_and_build_ghost()
    if not episodes: return

    print("Сохранение данных мульти-призрака...")
    np.savez("ghost_data.npz", 
             pos=GHOST_DATA[:, :3], 
             vel=GHOST_DATA[:, 3:]
    )

    ds = TMWeightedDataset(episodes, weights, K=K)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = PolicyNet(ds.X.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    milestone = int(EPOCHS * 0.7)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[milestone], gamma=0.1)
    mse_none = nn.MSELoss(reduction='none')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for x, y, w in dl:
            x, y, w = x.to(device), y.to(device), w.to(device)
            w = w.view(-1, 1)
            
            opt.zero_grad()
            steer_pred, tb_pred = model(x)
            
            # Weighted Loss
            loss_s = (mse_none(steer_pred, y[:, 0:1]) * w).mean()
            loss_t = (mse_none(tb_pred[:, 0:1], y[:, 1:2]) * w).mean()
            loss_b = (mse_none(tb_pred[:, 1:2], y[:, 2:3]) * w).mean()
            
            loss = 2.0 * loss_s + 1.0 * loss_t + 2.0 * loss_b
            
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        scheduler.step()
        print(f"Ep {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dl):.5f}")

    model.to("cpu")
    torch.save({
        "model": model.state_dict(),
        "mean": ds.mean,
        "std": ds.std,
        "K": K,
        "ghost_file": "ghost_data.npz",
        "lookahead": LOOKAHEAD_STEPS # Сохраняем, чтобы агент знал
    }, MODEL_NAME)
    print("Готово.")

if __name__ == "__main__":
    train()