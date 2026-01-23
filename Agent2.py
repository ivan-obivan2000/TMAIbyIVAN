import sys
import numpy as np
import torch
import torch.nn as nn

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

# Имя модели должно совпадать с тем, что сохранил train.py
MODEL_PATH = "policy_bc_v4.pt"

# ==========================
# КОНФИГУРАЦИЯ
# ==========================
CONTROL_EVERY_MS = 20
PRINT_EVERY_MS = 100       # Частота обновления в консоли
STEER_SMOOTH_ALPHA = 0.6   # 0.1 - плавно, 0.5 - резко, 1.0 - без сглаживания
MAX_STEER = 65536

# ==========================
# МОДЕЛЬ (Копия из train.py)
# ==========================
class PolicyNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # Архитектура должна 1-в-1 совпадать с train.py
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),       # <--- Важно! В обучении мы добавили это
            nn.ReLU(),
            nn.Dropout(0.05),        # <--- Было 0.05
            
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
# ПОДГОТОВКА ДАННЫХ
# ==========================
def safe_get(obj, name, default=None):
    try: return getattr(obj, name)
    except: return default

def f3(v):
    try: return [float(v[0]), float(v[1]), float(v[2])]
    except: return [0.0, 0.0, 0.0]

def f4(v):
    try: return [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
    except: return [0.0, 0.0, 0.0, 1.0]

# Эта функция должна быть КОПИЕЙ логики из train.py
def obs_vec(r):
    # Достаем данные из подготовленного словаря rec
    p_rel = r["position_rel"]      # [x, y, z] - уже посчитано относительно старта
    v = r["velocity"]
    
    car = r.get("car", {})
    eng = r.get("engine", {})
    dyna = r.get("dyna", {})

    # Значения по умолчанию для физики
    quat = dyna.get("quat") or [0, 0, 0, 1] 
    ang_vel = dyna.get("angular_speed") or [0, 0, 0]
    loc_v = car.get("current_local_speed") or [0, 0, 0]

    feat = [
        # 1. Относительная позиция (делим на 100, как в train)
        p_rel[0] / 100.0, 
        p_rel[1] / 100.0, 
        p_rel[2] / 100.0,
        
        # 2. Физика тела (Global)
        v[0] / 10.0, v[1] / 10.0, v[2] / 10.0,
        r.get("speed_norm", 0) / 100.0,
        
        # 3. Ориентация (Quaternions - 4 числа) - ВМЕСТО YPR
        quat[0], quat[1], quat[2], quat[3],
        
        # 4. Вращение (Angular Velocity)
        ang_vel[0], ang_vel[1], ang_vel[2],

        # 5. Состояние машины
        loc_v[0]/10.0, loc_v[1]/10.0, loc_v[2]/10.0,
        car.get("turning_rate", 0.0),
        float(car.get("is_sliding", False)),
        
        # 6. Двигатель
        eng.get("actual_rpm", 0.0) / 10000.0,
        float(eng.get("gear", 1)) / 5.0,
    ]
    return np.array(feat, dtype=np.float32)

# ==========================
# АГЕНТ
# ==========================
class BCPlayer(Client):
    def __init__(self, model_path):
        super().__init__()

        print(f"Загрузка модели: {model_path} ...")
        # map_location="cpu" позволяет запускать на ноуте без CUDA
        ckpt = torch.load(model_path, map_location="cpu")

        self.K = int(ckpt["K"])
        self.mean = ckpt["mean"].astype(np.float32)
        self.std = ckpt["std"].astype(np.float32)

        # Инициализируем сеть правильным размером входа
        in_dim = self.mean.shape[1]
        print(f"Входной вектор: {in_dim} признаков, История K={self.K}")
        
        self.model = PolicyNet(in_dim)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.enabled = True 
        self.hist = []
        self.steer_smoothed = 0.0
        
        self.start_pos_abs = None 
        self.prev_t = None
        self.prev_race_time = None
        
        self.last_control_time = -10**9
        self.last_print_time = -10**9

    def on_registered(self, iface: TMInterface):
        print(f"[OK] Подключено к {iface.server_name}")
        iface.register_custom_command("bc")

    def on_custom_command(self, iface, time_from, time_to, command, args):
        if command == "bc":
            if args and args[0].lower() in ("off", "0", "false"):
                self.enabled = False
                iface.log("AI OFF", "warning")
            else:
                self.enabled = True
                # Сброс при включении
                self.hist.clear()
                self.start_pos_abs = None
                iface.log("AI ON", "success")

    def on_run_step(self, iface: TMInterface, t: int):
        if t < 0: return

        try: 
            s = iface.get_simulation_state()
            # Нужно для извлечения физики
            dyna = safe_get(s, "dyna")
            cur_dyna = safe_get(dyna, "current_state") if dyna else None
            scene = safe_get(s, "scene_mobil")
        except: 
            return

        race_time = int(safe_get(s, "race_time", 0))

        # === ДЕТЕКТОР РЕСТАРТА ===
        # Если время откатилось назад - сбрасываем всё
        if (self.prev_t is not None and t < self.prev_t) or \
           (self.prev_race_time is not None and race_time < self.prev_race_time):
            self.hist.clear()
            self.steer_smoothed = 0.0
            self.start_pos_abs = None
            self.last_control_time = -10**9
        
        self.prev_t = t
        self.prev_race_time = race_time

        if t - self.last_control_time < CONTROL_EVERY_MS:
            return
        self.last_control_time = t

        # === СБОР ДАННЫХ (Формируем словарь как в JSON) ===
        pos_abs = f3(safe_get(s, "position"))
        
        # Запоминаем точку старта
        if self.start_pos_abs is None:
            self.start_pos_abs = pos_abs

        # Относительная позиция
        pos_rel = [
            pos_abs[0] - self.start_pos_abs[0],
            pos_abs[1] - self.start_pos_abs[1],
            pos_abs[2] - self.start_pos_abs[2],
        ]

        vel = f3(safe_get(s, "velocity"))
        
        # Собираем словарь 'rec', чтобы передать в obs_vec
        rec = {
            "position": pos_abs,
            "position_rel": pos_rel, # Важно!
            "velocity": vel,
            "speed_norm": float(np.linalg.norm(vel)),
            "car": {}, "engine": {}, "dyna": {}
        }

        # Заполняем car / engine
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
                    "gear": int(safe_get(engine, "gear", 0)),
                }

        # Заполняем dyna (кватернионы и угловая скорость)
        if cur_dyna:
            rec["dyna"] = {
                "quat": f4(safe_get(cur_dyna, "quat")),
                "angular_speed": f3(safe_get(cur_dyna, "angular_speed")),
            }

        # === ИНФЕРЕНС ===
        # Получаем вектор признаков
        vec = obs_vec(rec)
        self.hist.append(vec)
        
        if len(self.hist) > self.K:
            self.hist.pop(0)

        # Ждем накопления истории
        if len(self.hist) < self.K or not self.enabled:
            return

        # [1, K * Features]
        x = np.concatenate(self.hist).reshape(1, -1)
        # Нормализация (Mean/Std из файла)
        x = (x - self.mean) / self.std
        xt = torch.from_numpy(x).float()

        with torch.no_grad():
            steer_pred, tb = self.model(xt)

        steer_raw = float(steer_pred.item())
        throttle_p = float(tb[0, 0])
        brake_p = float(tb[0, 1])

        # === ЛОГИКА УПРАВЛЕНИЯ ===
        
        # Сглаживание руля
        self.steer_smoothed = (1 - STEER_SMOOTH_ALPHA) * self.steer_smoothed + STEER_SMOOTH_ALPHA * steer_raw
        steer_int = int(np.clip(self.steer_smoothed, -1.0, 1.0) * MAX_STEER)

        left = steer_int < -1000
        right = steer_int > 1000

        # Газ и Тормоз (пороги)
        accel_on = throttle_p > 0.95
        brake_on = brake_p > 0.85  # Чуть повысил порог тормоза, чтобы не дергался

        # Launch Control (чтобы не стоять на месте)
        speed_kmh = rec["speed_norm"] * 3.6
        if speed_kmh < 5.0 and not brake_on:
            accel_on = True

        # Отправка
        iface.set_input_state(
            sim_clear_buffer=False,
            accelerate=accel_on,
            brake=brake_on,
            left=left,
            right=right,
            steer=steer_int,
            gas=65536 if accel_on else 0,
        )

        # === КОНСОЛЬ ===
        if t - self.last_print_time >= PRINT_EVERY_MS:
            self.last_print_time = t
            
            # Визуал руля
            s_vis = int((steer_raw + 1) / 2 * 10)
            s_vis = max(0, min(10, s_vis))
            bar = ["."] * 11
            bar[s_vis] = "O"
            bar_str = "".join(bar)
            
            state_str = "GAS" if accel_on else "..."
            if brake_on: state_str = "BRK"

            print(f"\r[{bar_str}] {state_str} | T:{throttle_p:.2f} B:{brake_p:.2f} | Kmh:{int(speed_kmh)}", end="")

def main():
    server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    run_client(BCPlayer(MODEL_PATH), server_name)

if __name__ == "__main__":
    main()