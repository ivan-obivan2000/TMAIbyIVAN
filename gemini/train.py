import glob, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

LOG_DIR = "tmi_logs_gemini"
K = 10          # История (100мс)
BATCH = 256
EPOCHS = 50     # Поставим побольше
LR = 0.0005     # Чуть помедленнее обучение для точности

def load_episodes():
    """Читает все файлы и разбивает на эпизоды по ключу 'ep'"""
    files = sorted(glob.glob(f"{LOG_DIR}/*.jsonl"))
    all_episodes = []
    
    print(f"Найдено файлов: {len(files)}")
    
    for fp in files:
        current_ep_data = []
        last_ep_id = -1
        
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try: 
                    rec = json.loads(line)
                    ep_id = rec.get("ep", 0)
                    
                    # Если ID эпизода сменился, сохраняем предыдущий и начинаем новый
                    if ep_id != last_ep_id:
                        if len(current_ep_data) > K + 10: # Фильтр совсем коротких
                            all_episodes.append(current_ep_data)
                        current_ep_data = []
                        last_ep_id = ep_id
                    
                    current_ep_data.append(rec)
                except: pass
            
            # Сохраняем последний хвост
            if len(current_ep_data) > K + 10:
                all_episodes.append(current_ep_data)

    print(f"Всего валидных заездов (эпизодов): {len(all_episodes)}")
    return all_episodes

# Вектор наблюдения (ВХОД)
def obs_vec(r):
    feat = [
        r["vel"][0], r["vel"][1], r["vel"][2], # Глобальная скорость
        r["spd"],                              # Общая скорость
        r["ypr"][0], r["ypr"][1], r["ypr"][2], # Углы
        r["loc_v"][0], r["loc_v"][2],          # Локальная скорость (бок, вперед)
        r["ang_v"][1],                         # Вращение (Yaw rate)
        float(r["slide"]),                     # Дрифт
        r["rpm"] / 10000.0,                    # Обороты
        r["gear"]                              # Передача
    ]
    return np.array(feat, dtype=np.float32)

# Вектор действий (ВЫХОД)
def action_vec(r):
    steer = r["in_steer"] / 65536.0 # Нормализуем -1..1
    gas = 1.0 if r["in_gas"] > 0 else 0.0
    brake = float(r["in_brake"])
    return np.array([steer, gas, brake], dtype=np.float32)

class TMDataset(Dataset):
    def __init__(self, episodes):
        self.X, self.Y = [], []
        
        for ep in episodes:
            # Превращаем сырые данные в вектора
            obs = [obs_vec(r) for r in ep]
            act = [action_vec(r) for r in ep]
            
            # Создаем "окна" истории длиной K
            for t in range(K, len(ep)):
                # Вход: склейка K прошлых кадров
                x = np.concatenate(obs[t-K:t], axis=0)
                self.X.append(x)
                self.Y.append(act[t])
        
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        
        # Нормализация данных (критически важно!)
        self.mean = self.X.mean(axis=0, keepdims=True)
        self.std = self.X.std(axis=0, keepdims=True) + 1e-6
        self.X = (self.X - self.mean) / self.std
        
        print(f"Датасет готов: {len(self.X)} примеров.")

    def __len__(self): return len(self.X)
    def __getitem__(self, i): 
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

# Нейросеть
class PolicyNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        # Голова руля (число -1..1)
        self.steer_head = nn.Linear(256, 1)
        # Голова педалей (2 числа: вероятность газа и тормоза)
        self.pedals_head = nn.Linear(256, 2)

    def forward(self, x):
        h = self.net(x)
        steer = torch.tanh(self.steer_head(h))
        pedals = self.pedals_head(h) # Logits
        return steer, pedals

def train():
    eps = load_episodes()
    if not eps: return
    
    ds = TMDataset(eps)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    
    model = PolicyNet(ds.X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_pedals = nn.BCEWithLogitsLoss()

    print("Начинаю обучение...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in dl:
            opt.zero_grad()
            pred_steer, pred_pedals = model(x)
            
            # Loss руля (MSE)
            l_steer = ((pred_steer - y[:, 0:1]) ** 2).mean()
            # Loss педалей (BCE)
            l_pedals = loss_pedals(pred_pedals, y[:, 1:3])
            
            loss = l_steer + l_pedals
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
        print(f"Эпоха {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dl):.5f}")

    # Сохраняем модель и параметры нормализации
    torch.save({
        "model": model.state_dict(),
        "mean": ds.mean,
        "std": ds.std,
        "K": K
    }, "model_final.pt")
    print("Готово! Модель сохранена в model_final.pt")

if __name__ == "__main__":
    train()