import sys
import numpy as np
import torch
import torch.nn as nn
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

# ==========================================
# 1. АРХИТЕКТУРА МОДЕЛИ (Копия из trainer.py)
# ==========================================
# ==========================================
# 1. АРХИТЕКТУРА МОДЕЛИ (ИСПРАВЛЕНА)
# ==========================================
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
        self.steer_head = nn.Linear(256, 1)
        
        # ВАЖНО: Имя должно совпадать с тренером (pedals_head)
        self.pedals_head = nn.Linear(256, 2) 

    def forward(self, x):
        h = self.net(x)
        steer = torch.tanh(self.steer_head(h))
        pedals = self.pedals_head(h) # Тут тоже используем pedals_head
        return steer, pedals

# Функция подготовки вектора (Должна быть 1-в-1 как при обучении)
def obs_vec(r):
    feat = [
        r["vel"][0], r["vel"][1], r["vel"][2], 
        r["spd"],                              
        r["ypr"][0], r["ypr"][1], r["ypr"][2], 
        r["loc_v"][0], r["loc_v"][2],          
        r["ang_v"][1],                         
        float(r["slide"]),                     
        r["rpm"] / 10000.0,                    
        r["gear"]                              
    ]
    return np.array(feat, dtype=np.float32)

# ==========================================
# 2. АГЕНТ (С исправлениями)
# ==========================================
class AI(Client):
    def __init__(self, model_path):
        super().__init__()
        print(f"Загрузка модели из: {model_path}...")
        try:
            dat = torch.load(model_path, map_location="cpu") # map_location="cpu" важно если нет GPU
        except FileNotFoundError:
            print(f"ОШИБКА: Файл {model_path} не найден! Проверь имя файла.")
            sys.exit(1)

        self.K = dat["K"]
        self.mean = dat["mean"]
        self.std = dat["std"]
        
        # Инициализация сети
        input_dim = self.mean.shape[1]
        self.model = PolicyNet(input_dim)
        self.model.load_state_dict(dat["model"])
        self.model.eval()
        
        self.hist = [] 
        self.last_print = 0

    def on_registered(self, iface):
        print(f"[OK] Агент готов. История K={self.K}")
        print("Жми газ, робот!")

    def on_run_step(self, iface, t):
        # 1. Сбор данных из игры
        try: 
            s = iface.get_simulation_state()
            scene = getattr(s, "scene_mobil", None)
            dyna = getattr(s, "dyna", None)
            curr_dyna = getattr(dyna, "current_state", None) if dyna else None
            engine = getattr(scene, "engine", None) if scene else None
        except: 
            return

        # Если время отрицательное (отсчет), очищаем историю и ничего не делаем
        if t < 0:
            self.hist = []
            return

        # 2. Формируем словарь данных (как в recorder)
        # Важно использовать безопасное получение данных, чтобы не упасть
        vel = [float(x) for x in getattr(s, "velocity", [0,0,0])]
        spd = float(np.linalg.norm(vel)) * 3.6 # км/ч для удобства отображения
        
        rec = {
            "vel": vel,
            "spd": float(np.linalg.norm(vel)), # тут м/с для нейронки
            "ypr": [float(x) for x in getattr(s, "yaw_pitch_roll", [0,0,0])],
            "loc_v": [float(x) for x in getattr(scene, "current_local_speed", [0,0,0])] if scene else [0,0,0],
            "ang_v": [float(x) for x in getattr(curr_dyna, "angular_speed", [0,0,0])] if curr_dyna else [0,0,0],
            "slide": 1 if getattr(scene, "is_sliding", False) else 0,
            "rpm": float(getattr(engine, "actual_rpm", 0)) if engine else 0,
            "gear": float(getattr(engine, "gear", 0)) if engine else 0
        }

        # 3. Обновляем буфер истории
        self.hist.append(obs_vec(rec))
        if len(self.hist) > self.K: 
            self.hist.pop(0)
        
        # 4. Если буфер заполнился — предсказываем
        if len(self.hist) == self.K:
            # Нормализация входных данных
            x = np.concatenate(self.hist).reshape(1, -1)
            x = (x - self.mean) / self.std
            
            with torch.no_grad():
                steer_out, gb_out = self.model(torch.from_numpy(x.astype(np.float32)))
            
            # --- ИНТЕРПРЕТАЦИЯ ---
            raw_steer = float(steer_out)
            probs = torch.sigmoid(gb_out).numpy()[0]
            prob_gas, prob_brake = probs[0], probs[1]

            # --- ЛОГИКА УПРАВЛЕНИЯ ---
            
            # РУЛЬ: Перевод из float (-1..1) в int (-65536..65536)
            # Добавим "мертвую зону", чтобы не вилял на прямой
            steer_cmd = 0
            if abs(raw_steer) > 0.1: 
                steer_cmd = int(raw_steer * 65536)

            # ГАЗ И ТОРМОЗ
            gas_cmd = 65536 if prob_gas > 0.5 else 0
            brake_cmd = True if prob_brake > 0.5 else False

            # === ЛОНЧ-КОНТРОЛЬ (Start Assist) ===
            # Если скорость меньше 5 км/ч и нет явной команды тормозить - ЖМИ ГАЗ!
            # Это лечит проблему "стояния на старте"
            if spd < 5.0 and not brake_cmd:
                gas_cmd = 65536

            # Отправка в игру
            # sim_clear_buffer=False позволяет накладывать команды поверх существующих (безопаснее)
            iface.set_input_state(sim_clear_buffer=False, steer=steer_cmd, gas=gas_cmd, brake=brake_cmd)

            # 5. Отладка в консоль (раз в 200мс)
            if t - self.last_print > 200:
                self.last_print = t
                # Визуализация руля
                bar = [" "] * 10
                idx = int((raw_steer + 1) / 2 * 9)
                idx = max(0, min(9, idx))
                bar[idx] = "|"
                bar_str = "".join(bar)
                
                print(f"\r[{bar_str}] Steer:{raw_steer:.2f} Gas:{prob_gas:.2f} Brake:{prob_brake:.2f} Spd:{int(spd)}", end="")

def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    # Убедись, что имя файла совпадает с тем, что создал trainer.py!
    model_file = "model_final.pt" 
    
    run_client(AI(model_file), server_name)

if __name__ == "__main__":
    main()