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
K = 20
BATCH = 1024
EPOCHS = 100
LR = 1e-3
MODEL_NAME = "policy_bc3.pt"

# Параметры взвешивания
WEIGHT_BEST = 3  # Вес самого быстрого заезда
WEIGHT_WORST = 1  # Вес самого медленного (но успешного) заезда

# ==========================
# ЗАГРУЗКА И ФИЛЬТРАЦИЯ
# ==========================
def load_and_process_episodes():
    files = sorted(glob.glob(f"{LOG_DIR}/*.jsonl"))
    print(f"1. Поиск файлов в {LOG_DIR}...")
    
    raw_episodes = []
    # Читаем все файлы и разбиваем на эпизоды
    for fp in files:
        current_ep = []
        last_ep_id = -1
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        ep_id = rec.get("ep", 0)
                        
                        if ep_id != last_ep_id:
                            if len(current_ep) > K: raw_episodes.append(current_ep)
                            current_ep = []
                            last_ep_id = ep_id
                        
                        current_ep.append(rec)
                    except: pass
                if len(current_ep) > K: raw_episodes.append(current_ep)
        except Exception as e:
            print(f"Ошибка чтения {fp}: {e}")

    print(f"   Загружено {len(raw_episodes)} сырых эпизодов.")
    if not raw_episodes: return [], []

    # --- ЭТАП 2: ФИЛЬТРАЦИЯ И ВЗВЕШИВАНИЕ ---
    finished_eps = []
    
    for ep in raw_episodes:
        # Проверяем флаг finished в последних 10 кадрах
        # (вдруг он мигнул, берем max)
        is_finished = any([r.get("finished", 0) for r in ep[-10:]])
        
        if is_finished:
            # Время заезда - это время последнего кадра
            duration = ep[-1]["time_ms"] - ep[0]["time_ms"]
            finished_eps.append({
                "data": ep,
                "time": duration
            })

    if not finished_eps:
        print("!!! ВНИМАНИЕ: Нет ни одного успешного финиша.")
        print("!!! Обучение будет идти на ВСЕХ данных с весом 1.0.")
        return raw_episodes, [1.0] * len(raw_episodes)

    # Сортируем по времени (от быстрого к медленному)
    finished_eps.sort(key=lambda x: x["time"])
    
    best_time = finished_eps[0]["time"]
    worst_time = finished_eps[-1]["time"]
    
    print(f"2. Статистика успешных заездов ({len(finished_eps)} шт):")
    print(f"   Best : {best_time/1000:.2f} сек")
    print(f"   Worst: {worst_time/1000:.2f} сек")

    # Рассчитываем веса по Гауссу
    # Формула: w = 1 + 9 * exp(-k * norm_time^2)
    final_episodes = []
    final_weights = []

    for item in finished_eps:
        t = item["time"]
        
        # Нормализация времени от 0 (лучшее) до 1 (худшее)
        if worst_time == best_time:
            norm_t = 0.0
        else:
            norm_t = (t - best_time) / (worst_time - best_time)
        
        # Гауссово затухание (sigma ~ 0.5)
        # При norm_t=0 (best) -> exp(0) = 1 -> вес 10
        # При norm_t=1 (worst) -> exp(-4) ~ 0 -> вес 1
        gauss_val = np.exp(-4.0 * (norm_t ** 2))
        
        # Масштабируем в диапазон [1.0 ... 10.0]
        weight = WEIGHT_WORST + (WEIGHT_BEST - WEIGHT_WORST) * gauss_val
        
        final_episodes.append(item["data"])
        final_weights.append(weight)

    print(f"   Веса распределены: Лучший={final_weights[0]:.2f} ... Худший={final_weights[-1]:.2f}")
    
    return final_episodes, final_weights

# ==========================
# ФУНКЦИИ ВЕКТОРИЗАЦИИ
# ==========================
def obs_vec(r):
    # Достаем данные
    p = r["position"]          # [x, y, z]
    v = r["velocity"]

    
    car = r.get("car", {})
    eng = r.get("engine", {})
    sync = r.get("sync", {})

    # Время гонки (важно для таймингов!)
    # В recorder оно пишется как 'race_time' (в мс). Переводим в секунды/100
    race_t = r.get("race_time", 0) / 10000.0 

    feat = [
        # === ГЕОГРАФИЯ (ГДЕ Я?) ===
        # Делим на 100, чтобы координаты 500 превратились в 5.0
        # Это спасет нейросеть от "взрыва" весов
        p[0] / 100.0, 
        p[1] / 10.0,   # Высота обычно меньше, делим на 100
        p[2] / 100.0,
        
        race_t,         # Прогресс по времени

        # === ФИЗИКА (КАК Я ДВИГАЮСЬ?) ===
        v[0], v[1], v[2],
        r.get("speed_norm", 0),


        # === МАШИНА (ЧТО СО МНОЙ?) ===
        *(car.get("current_local_speed") or [0, 0, 0]),
        car.get("turning_rate", 0.0),
        float(car.get("is_sliding", 0.0)),
        
        eng.get("actual_rpm", 0.0) / 10000.0,
        eng.get("gear", 0.0),
        
        sync.get("speed_forward", 0.0),
        sync.get("speed_sideward", 0.0),
    ]
    return np.array(feat, dtype=np.float32)

def action_vec(r):
    # Обработка клавиатуры (КАК В АГЕНТЕ)
    inputs = r.get("inputs", {})
    steer = inputs.get("steer", 0) / 65536.0
    
    # Если аналог 0, проверяем кнопки
    if steer == 0.0:
        if inputs.get("left"): steer = -1.0
        elif inputs.get("right"): steer = 1.0
            
    # Газ: максимум из аналога и кнопки
    gas_btn = 1.0 if inputs.get("accelerate") else 0.0
    gas_analog = inputs.get("gas", 0) / 65536.0
    throttle = max(gas_btn, gas_analog)
    
    brake = 1.0 if inputs.get("brake") else 0.0
    
    return np.array([steer, throttle, brake], dtype=np.float32)

# ==========================
# DATASET С ВЕСАМИ
# ==========================
class TMWeightedDataset(Dataset):
    def __init__(self, episodes, weights, K):
        self.X = []
        self.Y = []
        self.W = [] # Вектор весов

        for ep, weight in zip(episodes, weights):
            # Старт для относительных координат
            start_p = ep[0]["position"]
            
            # Предварительная обработка эпизода
            processed_obs = []
            processed_act = []
            
            for r in ep:
                # Добавляем rel_pos, если его нет
                p = r["position"]
                r["position_rel"] = [p[0]-start_p[0], p[1]-start_p[1], p[2]-start_p[2]]
                
                processed_obs.append(obs_vec(r))
                processed_act.append(action_vec(r))
            
            # Создаем окна
            for t in range(K, len(ep)):
                # Вход: история K кадров
                x = np.concatenate(processed_obs[t-K:t], axis=0)
                y = processed_act[t]
                
                self.X.append(x)
                self.Y.append(y)
                self.W.append(weight) # Присваиваем вес всему окну

        self.X = np.stack(self.X)
        self.Y = np.stack(self.Y)
        self.W = np.array(self.W, dtype=np.float32)

        # Нормализация
        print("Нормализация данных...")
        self.mean = self.X.mean(axis=0, keepdims=True)
        self.std = self.X.std(axis=0, keepdims=True) + 1e-6
        self.X = (self.X - self.mean) / self.std
        
        print(f"Датасет готов: {len(self.X)} сэмплов.")

    def __len__(self): return len(self.X)
    
    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]), 
            torch.from_numpy(self.Y[i]),
            torch.tensor(self.W[i]) # Возвращаем еще и вес
        )

# ==========================
# МОДЕЛЬ (АРХИТЕКТУРА 512)
# ==========================
class PolicyNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256), # Выход 256
            nn.ReLU(),
        )
        # Входы голов = 256
        self.head_steer = nn.Linear(256, 1)
        self.head_tb = nn.Linear(256, 2)

    def forward(self, x):
        h = self.net(x)
        steer = torch.tanh(self.head_steer(h))
        tb = torch.sigmoid(self.head_tb(h))
        return steer, tb

# ==========================
# TRAIN LOOP С ВЕСАМИ
# ==========================
def train():
    episodes, weights = load_and_process_episodes()
    if not episodes: return

    ds = TMWeightedDataset(episodes, weights, K=K)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = PolicyNet(ds.X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Функции потерь без усреднения (reduction='none'), чтобы умножить на веса
    mse_none = nn.MSELoss(reduction='none')
    bce_none = nn.BCELoss(reduction='none') 

    print("Начинаю обучение с весами...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for x, y, w in dl:
            # w имеет размер [BATCH], нужно [BATCH, 1] для умножения
            w = w.view(-1, 1) 
            
            opt.zero_grad()
            steer_pred, tb_pred = model(x)
            
            # Ground Truth
            steer_gt = y[:, 0:1]
            throttle_gt = y[:, 1:2]
            brake_gt = y[:, 2:3]
            
            # --- ВЗВЕШЕННЫЙ LOSS ---
            
            # 1. Steer (MSE)
            loss_s = mse_none(steer_pred, steer_gt) 
            loss_s = (loss_s * w).mean() # Умножаем на вес и усредняем
            
            # 2. Throttle (MSE, так как регрессия 0..1)
            loss_t = mse_none(tb_pred[:, 0:1], throttle_gt)
            loss_t = (loss_t * w).mean()
            
            # 3. Brake (MSE)
            loss_b = mse_none(tb_pred[:, 1:2], brake_gt)
            loss_b = (loss_b * w).mean()
            
            # Сумма
            loss = 2 * loss_s + 1 * loss_t + 1.5 * loss_b
            
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {total_loss/len(dl):.5f}")

    torch.save({
        "model": model.state_dict(),
        "mean": ds.mean,
        "std": ds.std,
        "K": K,
    }, MODEL_NAME)
    
    print(f"Готово! Модель сохранена в {MODEL_NAME}")

if __name__ == "__main__":
    train()