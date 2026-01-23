import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================
# КОНФИГУРАЦИЯ
# ==========================
LOG_DIR = "tmi_logs3"
K = 10                # Длина истории (контекст)
BATCH = 1024
EPOCHS = 25
LR = 1e-3
MODEL_NAME = "policy_bc_v4.pt"

# Параметры взвешивания (Learning from Best)
WEIGHT_BEST = 5.0     # Усиливаем вес лучших заездов
WEIGHT_WORST = 1.0

# ==========================
# ЗАГРУЗКА И ФИЛЬТРАЦИЯ
# ==========================
def load_and_process_episodes():
    files = sorted(glob.glob(f"{LOG_DIR}/*.jsonl"))
    print(f"1. Поиск файлов в {LOG_DIR}...")
    
    raw_episodes = []
    
    for fp in files:
        current_ep = []
        last_ep_id = -1
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        # --- ИЗМЕНЕНИЕ 1: Правильное имя ключа ---
                        ep_id = rec.get("episode_id", -1) 
                        
                        if ep_id != last_ep_id:
                            # Сохраняем предыдущий, если он длинный
                            if len(current_ep) > K: raw_episodes.append(current_ep)
                            current_ep = []
                            last_ep_id = ep_id
                        
                        current_ep.append(rec)
                    except: pass
                # Добавляем хвост файла
                if len(current_ep) > K: raw_episodes.append(current_ep)
        except Exception as e:
            print(f"Ошибка чтения {fp}: {e}")

    print(f"   Загружено {len(raw_episodes)} сырых эпизодов.")
    if not raw_episodes: return [], []

    # --- ЭТАП 2: ФИЛЬТРАЦИЯ ---
    finished_eps = []
    
    for ep in raw_episodes:
        # Проверяем финиш в конце
        is_finished = any([r.get("finished", 0) for r in ep[-20:]])
        
        if is_finished:
            # Считаем время: конец - начало
            t_start = ep[0]["time_ms"]
            t_end = ep[-1]["time_ms"]
            duration = t_end - t_start
            
            finished_eps.append({
                "data": ep,
                "time": duration
            })

    if not finished_eps:
        print("!!! ВНИМАНИЕ: Нет успешных финишей. Обучаемся на всем подряд (вес 1.0).")
        return raw_episodes, [1.0] * len(raw_episodes)

    # Сортировка: Самое маленькое время (лучшее) -> в начало
    finished_eps.sort(key=lambda x: x["time"])
    
    best_time = finished_eps[0]["time"]
    worst_time = finished_eps[-1]["time"]
    
    print(f"2. Статистика ({len(finished_eps)} заездов):")
    print(f"   Best : {best_time/1000:.2f} сек")
    print(f"   Worst: {worst_time/1000:.2f} сек")

    final_episodes = []
    final_weights = []

    for item in finished_eps:
        t = item["time"]
        
        if worst_time == best_time:
            norm_t = 0.0
        else:
            norm_t = (t - best_time) / (worst_time - best_time)
        
        # Гауссово распределение весов
        # norm_t = 0 (best) -> exp(0)=1 -> max weight
        gauss_val = np.exp(-4.0 * (norm_t ** 2))
        weight = WEIGHT_WORST + (WEIGHT_BEST - WEIGHT_WORST) * gauss_val
        
        final_episodes.append(item["data"])
        final_weights.append(weight)

    return final_episodes, final_weights

# ==========================
# ВЕКТОРИЗАЦИЯ (ИСПРАВЛЕННАЯ)
# ==========================
def obs_vec(r):
    # Данные из JSON (с учетом структуры tminterface)
    p = r.get("position", [0,0,0])
    v = r.get("velocity", [0,0,0])
    # r["position_rel"] мы создаем сами в Dataset
    p_rel = r.get("position_rel", [0,0,0]) 
    
    # Безопасное получение вложенных словарей
    car = r.get("car") or {}
    eng = r.get("engine") or {}
    sync = r.get("sync") or {}
    dyna = r.get("dyna") or {}
    
    # --- ИЗМЕНЕНИЕ 2: Берем Quaternions и Angular Speed вместо YPR ---
    # Если dyna нет, ставим дефолты
    quat = dyna.get("quat") or [0, 0, 0, 1] 
    ang_vel = dyna.get("angular_speed") or [0, 0, 0]
    
    # Локальная скорость машины (важнее глобальной velocity для заносов)
    loc_v = car.get("current_local_speed") or [0, 0, 0]

    feat = [
        # 1. Относительная позиция (куда ехать от старта)
        p_rel[0] / 100.0, 
        p_rel[1] / 100.0, 
        p_rel[2] / 100.0,
        
        # 2. Физика тела (Global)
        v[0] / 10.0, v[1] / 10.0, v[2] / 10.0,
        r.get("speed_norm", 0) / 100.0, # Нормализуем (~0-10)
        
        # 3. Ориентация (Quaternions - 4 числа)
        quat[0], quat[1], quat[2], quat[3],
        
        # 4. Вращение (Angular Velocity - важно для контроля заноса!)
        ang_vel[0], ang_vel[1], ang_vel[2],

        # 5. Состояние машины
        loc_v[0]/10.0, loc_v[1]/10.0, loc_v[2]/10.0,
        car.get("turning_rate", 0.0),
        float(car.get("is_sliding", False)),
        
        # 6. Двигатель
        eng.get("actual_rpm", 0.0) / 10000.0,
        float(eng.get("gear", 1)) / 5.0, # Примерно 1..5
    ]
    return np.array(feat, dtype=np.float32)

def action_vec(r):
    inputs = r.get("inputs", {})
    
    # --- ИЗМЕНЕНИЕ 3: Работаем только с кнопками (клавиатура) ---
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
            
            # Запоминаем старт для вычисления relative position
            start_pos = ep[0].get("position", [0,0,0])
            
            processed_obs = []
            processed_act = []
            
            for r in ep:
                # Вычисляем position_rel и кладем внутрь словаря, чтобы obs_vec его увидел
                curr_p = r.get("position", [0,0,0])
                r["position_rel"] = [
                    curr_p[0] - start_pos[0],
                    curr_p[1] - start_pos[1],
                    curr_p[2] - start_pos[2]
                ]
                
                processed_obs.append(obs_vec(r))
                processed_act.append(action_vec(r))
            
            # Формируем окна [T-K : T]
            for t in range(K, len(ep)):
                x = np.concatenate(processed_obs[t-K:t], axis=0) # Shape: [K * Features]
                y = processed_act[t]
                
                self.X.append(x)
                self.Y.append(y)
                self.W.append(weight)

        self.X = np.stack(self.X)
        self.Y = np.stack(self.Y)
        self.W = np.array(self.W, dtype=np.float32)

        # Нормализация входов (Mean/Std)
        print("Вычисление нормализации...")
        self.mean = self.X.mean(axis=0, keepdims=True)
        self.std = self.X.std(axis=0, keepdims=True) + 1e-6
        self.X = (self.X - self.mean) / self.std
        
        print(f"Dataset ready: {len(self.X)} samples. Input dim: {self.X.shape[1]}")

    def __len__(self): return len(self.X)
    
    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]), 
            torch.from_numpy(self.Y[i]),
            torch.tensor(self.W[i])
        )

# ==========================
# МОДЕЛЬ
# ==========================
class PolicyNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # Расширим сеть, так как входов стало больше
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512), # LayerNorm помогает при разнородных данных
            nn.ReLU(),
            nn.Dropout(0.05),
            
            nn.Linear(512, 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head_steer = nn.Linear(256, 1)
        self.head_tb = nn.Linear(256, 2)

    def forward(self, x):
        h = self.net(x)
        steer = torch.tanh(self.head_steer(h))    # -1 .. 1
        tb = torch.sigmoid(self.head_tb(h))       # 0 .. 1
        return steer, tb

# ==========================
# TRAIN
# ==========================
def train():
    episodes, weights = load_and_process_episodes()
    if not episodes:
        print("Нет данных для обучения.")
        return

    ds = TMWeightedDataset(episodes, weights, K=K)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = PolicyNet(ds.X.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    

    milestone = int(EPOCHS * 0.85)
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
            
            steer_gt = y[:, 0:1]
            throttle_gt = y[:, 1:2]
            brake_gt = y[:, 2:3]
            
            loss_s = (mse_none(steer_pred, steer_gt) * w).mean()
            loss_t = (mse_none(tb_pred[:, 0:1], throttle_gt) * w).mean()
            loss_b = (mse_none(tb_pred[:, 1:2], brake_gt) * w).mean()
            
            loss = 2.0 * loss_s + 1.0 * loss_t + 2.0 * loss_b
            
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        # --- ИЗМЕНЕНИЕ: Шаг шедулера ---
        scheduler.step()
        
        # Получаем текущий LR для вывода
        current_lr = scheduler.get_last_lr()[0]
        print(f"Ep {epoch+1}/{EPOCHS} | LR: {current_lr:.1e} | Loss: {total_loss/len(dl):.5f}")

    model.to("cpu")
    torch.save({
        "model": model.state_dict(),
        "mean": ds.mean,
        "std": ds.std,
        "K": K,
    }, MODEL_NAME)
    
    print(f"Модель сохранена в {MODEL_NAME}")
if __name__ == "__main__":
    train()