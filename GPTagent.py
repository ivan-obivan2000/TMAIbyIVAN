import sys
import numpy as np
import torch
import torch.nn as nn

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

MODEL ="policy_bc3.pt"
# ==========================
# КОНФИГУРАЦИЯ
# ==========================
CONTROL_EVERY_MS = 20
PRINT_EVERY_MS = 200        # Чаще обновляем консоль
STEER_SMOOTH_ALPHA = 0.1    # Чуть резче руль (было 0.15)
MAX_STEER = 65536

# ==========================
# МОДЕЛЬ (Должна совпадать с Train)
# ==========================
class PolicyNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            # Слой 1
            nn.Linear(in_dim, 512),  # <--- Проверь это число
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Слой 2
            nn.Linear(512, 512),     # <--- И это
            nn.ReLU(),
            
            # Слой 3
            nn.Linear(512, 256),     # <--- И это
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
# ПОДГОТОВКА ДАННЫХ
# ==========================
def safe_get(obj, name, default=None):
    try: return getattr(obj, name)
    except: return default

def f3(v):
    try: return [float(v[0]), float(v[1]), float(v[2])]
    except: return [0.0, 0.0, 0.0]

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
        # Делим на 1000, чтобы координаты 5000 превратились в 5.0
        # Это спасет нейросеть от "взрыва" весов
        p[0] / 1000.0, 
        p[1] / 100.0,   # Высота обычно меньше, делим на 100
        p[2] / 1000.0,
        
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

# ==========================
# АГЕНТ
# ==========================
class BCPlayer(Client):
    def __init__(self, model_path=MODEL):
        super().__init__()

        # Загрузка модели (CPU safe)
        print(f"Загрузка модели: {model_path} ...")
        ckpt = torch.load(model_path, map_location="cpu")

        self.K = int(ckpt["K"])
        self.mean = ckpt["mean"].astype(np.float32)
        self.std = ckpt["std"].astype(np.float32)

        self.model = PolicyNet(self.mean.shape[1])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # === ВКЛЮЧАЕМ ПО УМОЛЧАНИЮ ===
        self.enabled = True 
        
        self.hist = []
        self.steer_smoothed = 0.0
        self.start_pos_abs = None # Для расчета относительной позиции

        self.prev_t = None
        self.prev_race_time = None
        self.last_control_time = -10**9
        self.last_print_time = -10**9

    def on_registered(self, iface: TMInterface):
        print(f"[OK] Подключено к {iface.server_name}")
        print(f"[OK] Бот АКТИВЕН. История K={self.K}")
        iface.register_custom_command("bc")

    def on_custom_command(self, iface, time_from, time_to, command, args):
        if command == "bc":
            if args and args[0].lower() in ("off", "0", "false"):
                self.enabled = False
                iface.log("BC = OFF", "warning")
            else:
                self.enabled = True
                self.hist.clear()
                self.start_pos_abs = None
                iface.log("BC = ON", "success")

    def on_run_step(self, iface: TMInterface, t: int):
        if t < 0: return

        try: s = iface.get_simulation_state()
        except: return

        race_time = int(safe_get(s, "race_time", 0))

        # === ДЕТЕКТОР РЕСТАРТА ===
        # Если время откатилось назад - сбрасываем историю и начальную точку
        if (self.prev_t is not None and t < self.prev_t) or \
           (self.prev_race_time is not None and race_time < self.prev_race_time):
            self.hist.clear()
            self.steer_smoothed = 0.0
            self.start_pos_abs = None # Сброс стартовой позиции!
            self.last_control_time = -10**9
        
        self.prev_t = t
        self.prev_race_time = race_time

        # Лимит частоты управления
        if t - self.last_control_time < CONTROL_EVERY_MS:
            return
        self.last_control_time = t

        # === СБОР ДАННЫХ ===
        pos_abs = f3(safe_get(s, "position"))
        race_time_val = int(safe_get(s, "race_time", 0))
        
        # Если это первый кадр заезда, запоминаем где старт
        if self.start_pos_abs is None:
            self.start_pos_abs = pos_abs

        # Относительная позиция (как далеко мы уехали от старта)
        pos_rel = [
            pos_abs[0] - self.start_pos_abs[0],
            pos_abs[1] - self.start_pos_abs[1],
            pos_abs[2] - self.start_pos_abs[2],
        ]

        vel = f3(safe_get(s, "velocity"))
   
        
        # Данные для obs_vec
        rec = {
            "position": pos_abs,
            "position_rel": pos_rel,
            "velocity": vel,
            "race_time": race_time_val,
            "speed_norm": float(np.linalg.norm(vel)),
            
            "car": {}, "engine": {}, "sync": {}
        }

        scene = safe_get(s, "scene_mobil")
        if scene:
            rec["car"] = {
                "current_local_speed": f3(safe_get(scene, "current_local_speed")),
                "turning_rate": float(safe_get(scene, "turning_rate", 0.0)),
                "is_sliding": bool(safe_get(scene, "is_sliding", False)),
            }
            engine = safe_get(scene, "engine")
            if engine:
                rec["engine"] = {
                    "actual_rpm": float(safe_get(engine, "actual_rpm", 0.0)),
                    "gear": float(safe_get(engine, "gear", 0.0)),
                }
            sync = safe_get(scene, "sync_vehicle_state")
            if sync:
                rec["sync"] = {
                    "speed_forward": float(safe_get(sync, "speed_forward", 0.0)),
                    "speed_sideward": float(safe_get(sync, "speed_sideward", 0.0)),
                }

        # === ИНФЕРЕНС ===
        self.hist.append(obs_vec(rec))
        if len(self.hist) > self.K:
            self.hist.pop(0)

        # Ждем накопления истории
        if len(self.hist) < self.K or not self.enabled:
            return

        x = np.concatenate(self.hist).reshape(1, -1)
        x = (x - self.mean) / self.std
        xt = torch.from_numpy(x).float()

        with torch.no_grad():
            steer_pred, tb = self.model(xt)

        steer = float(steer_pred.item())
        throttle_p = float(tb[0, 0])
        brake_p = float(tb[0, 1])

        # === ЛОГИКА УПРАВЛЕНИЯ ===
        
        # Сглаживание руля
        self.steer_smoothed = (1 - STEER_SMOOTH_ALPHA) * self.steer_smoothed + STEER_SMOOTH_ALPHA * steer
        steer_int = int(np.clip(self.steer_smoothed, -1.0, 1.0) * MAX_STEER)

        left = steer_int < -2000
        right = steer_int > 2000

        # Газ и Тормоз по порогу
        accel_on = throttle_p > 0.4
        brake_on = brake_p > 0.9

        # !!! LAUNCH CONTROL (ПОМОЩЬ НА СТАРТЕ) !!!
        # Если скорость < 10 км/ч (примерно 2.7 м/с) и не тормозим — ЖМЕМ ГАЗ
        speed_kmh = rec["speed_norm"] * 3.6
        if speed_kmh < 10.0 and not brake_on:
            accel_on = True

        # Отправка в игру
        iface.set_input_state(
            sim_clear_buffer=False,
            accelerate=accel_on,
            brake=brake_on,
            left=left,
            right=right,
            steer=steer_int,
            gas=65536 if accel_on else 0,
        )

        # === ВИЗУАЛИЗАЦИЯ ===
        if t - self.last_print_time >= PRINT_EVERY_MS:
            self.last_print_time = t
            # Рисуем бар руля
            bar = [" "] * 10
            idx = int((steer + 1) / 2 * 9)
            idx = max(0, min(9, idx))
            bar[idx] = "|"
            bar_str = "".join(bar)
            
            # Помечаем газ звездочкой, если нажат
            gas_str = "GAS!" if accel_on else "...."
            
            print(f"\r[{bar_str}] Str:{steer:.2f} {gas_str} (p={throttle_p:.2f}) Spd:{int(speed_kmh)}", end="")

def main():
    server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    model_path = MODEL
    run_client(BCPlayer(model_path), server_name)

if __name__ == "__main__":
    main()