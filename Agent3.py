import sys
import os
import numpy as np
import torch
import torch.nn as nn

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

# ==========================
# НАСТРОЙКИ
# ==========================
MODEL_PATH = "policy_trajectory_v4.pt" # Убедись, что имя совпадает с train.py
GHOST_FILE = "ghost_data.npz"          # Файл с мульти-призраком

CONTROL_EVERY_MS = 20
PRINT_EVERY_MS = 100
STEER_SMOOTH_ALPHA = 0.1
MAX_STEER = 65536

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
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
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

# ==========================
# АГЕНТ
# ==========================
class GhostFollowingPlayer(Client):
    def __init__(self, model_path, ghost_path):
        super().__init__()
        
        # 1. Загрузка Модели и Параметров
        print(f"Загрузка модели: {model_path} ...")
        if not os.path.exists(model_path):
            print("ОШИБКА: Файл модели не найден.")
            sys.exit(1)

        ckpt = torch.load(model_path, map_location="cpu")

        self.K = int(ckpt["K"])
        self.mean = ckpt["mean"].astype(np.float32)
        self.std = ckpt["std"].astype(np.float32)
        
        # Загружаем Lookahead, если он есть, иначе 0
        self.lookahead = int(ckpt.get("lookahead", 0))
        print(f"Параметры: K={self.K}, Lookahead={self.lookahead}")

        in_dim = self.mean.shape[1]
        self.model = PolicyNet(in_dim)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # 2. Загрузка Призрака
        if not os.path.exists(ghost_path):
            print(f"ОШИБКА: Не найден файл {ghost_path}! Запустите train.py.")
            sys.exit(1)
            
        print(f"Загрузка траектории: {ghost_path} ...")
        g_data = np.load(ghost_path)
        self.ghost_pos = g_data['pos'].astype(np.float32)
        self.ghost_vel = g_data['vel'].astype(np.float32)
        
        # Подготовка 6D массива для поиска (Pos + Vel)
        # Это нужно, чтобы отличать одни и те же координаты с разной скоростью
        self.ghost_6d = np.hstack([self.ghost_pos, self.ghost_vel])
        
        print(f"Призрак готов: {len(self.ghost_pos)} точек.")

        # Состояние
        self.enabled = True
        self.hist = []
        self.steer_smoothed = 0.0
        
        self.prev_t = None
        self.prev_race_time = None
        self.last_control_time = -10**9
        self.last_print_time = -10**9

    def on_registered(self, iface: TMInterface):
        print(f"[OK] Агент готов. Lookahead активен.")
        iface.register_custom_command("bc")

    def on_custom_command(self, iface, time_from, time_to, command, args):
        if command == "bc":
            if args and args[0].lower() in ("off", "0", "false"):
                self.enabled = False
                iface.log("AI OFF", "warning")
            else:
                self.enabled = True
                self.hist.clear()
                iface.log("AI ON", "success")

    # --- ПОСТРОЕНИЕ ВЕКТОРА (6D + LOOKAHEAD) ---
    def make_obs_vec(self, r):
        # Текущее состояние
        p = np.array(r["position"], dtype=np.float32)
        v = np.array(r["velocity"], dtype=np.float32)

        # 1. Формируем 6D запрос [Pos, Vel]
        query = np.concatenate([p, v])

        # 2. Ищем ближайшего соседа по 6 измерениям
        # (Используем broadcasting numpy - это быстро для <100k точек)
        # dists^2 = (x-gx)^2 + ... + (vz-gvz)^2
        diff = self.ghost_6d - query
        dists_sq = np.sum(diff**2, axis=1)
        idx = np.argmin(dists_sq)
        
        # Дистанция до ближайшей точки (для логов)
        dist_scalar = np.sqrt(dists_sq[idx])

        # 3. ПРИМЕНЯЕМ LOOKAHEAD
        # Берем индекс на N шагов вперед
        target_idx = min(idx + self.lookahead, len(self.ghost_pos) - 1)

        target_p = self.ghost_pos[target_idx]
        target_v = self.ghost_vel[target_idx]

        # 4. Вектора ошибок (относительно будущей точки!)
        delta_p = target_p - p
        delta_v = target_v - v

        # Остальные данные
        car = r.get("car", {})
        eng = r.get("engine", {})
        dyna = r.get("dyna", {})
        
        quat = dyna.get("quat") or [0, 0, 0, 1] 
        ang_vel = dyna.get("angular_speed") or [0, 0, 0]
        loc_v = car.get("current_local_speed") or [0, 0, 0]

        # 5. Собираем вектор
        feat = [
            # Ошибка позиции
            delta_p[0], delta_p[1], delta_p[2],
            # Ошибка скорости
            delta_v[0], delta_v[1], delta_v[2],

            # Абсолютная физика
            v[0], v[1], v[2],
            r.get("speed_norm", 0) / 100.0,

            # Ориентация
            quat[0], quat[1], quat[2], quat[3],
            ang_vel[0], ang_vel[1], ang_vel[2],

            # Состояние машины
            loc_v[0], loc_v[1], loc_v[2],
            car.get("turning_rate", 0.0),
            float(car.get("is_sliding", False)),
            eng.get("actual_rpm", 0.0) / 10000.0,
            float(eng.get("gear", 1)) / 5.0,
        ]
        return np.array(feat, dtype=np.float32), dist_scalar

    def on_run_step(self, iface: TMInterface, t: int):
        if t < 0: return

        try: 
            s = iface.get_simulation_state()
            dyna = safe_get(s, "dyna")
            cur_dyna = safe_get(dyna, "current_state") if dyna else None
            scene = safe_get(s, "scene_mobil")
        except: return

        race_time = int(safe_get(s, "race_time", 0))

        # Рестарт детекшн
        if (self.prev_t is not None and t < self.prev_t) or \
           (self.prev_race_time is not None and race_time < self.prev_race_time):
            self.hist.clear()
            self.steer_smoothed = 0.0
            self.last_control_time = -10**9
        
        self.prev_t = t
        self.prev_race_time = race_time

        if t - self.last_control_time < CONTROL_EVERY_MS:
            return
        self.last_control_time = t

        # Сбор данных
        rec = {
            "position": f3(safe_get(s, "position")),
            "velocity": f3(safe_get(s, "velocity")),
            "speed_norm": float(np.linalg.norm(f3(safe_get(s, "velocity")))),
            "car": {}, "engine": {}, "dyna": {}
        }
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
        if cur_dyna:
            rec["dyna"] = {
                "quat": f4(safe_get(cur_dyna, "quat")),
                "angular_speed": f3(safe_get(cur_dyna, "angular_speed")),
            }

        # === ИНФЕРЕНС ===
        vec, current_dist_error = self.make_obs_vec(rec)
        self.hist.append(vec)
        
        if len(self.hist) > self.K:
            self.hist.pop(0)

        if len(self.hist) < self.K or not self.enabled:
            return

        x = np.concatenate(self.hist).reshape(1, -1)
        x = (x - self.mean) / self.std
        xt = torch.from_numpy(x).float()

        with torch.no_grad():
            steer_pred, tb = self.model(xt)

        steer_raw = float(steer_pred.item())
        throttle_p = float(tb[0, 0])
        brake_p = float(tb[0, 1])

        # === УПРАВЛЕНИЕ ===
        self.steer_smoothed = (1 - STEER_SMOOTH_ALPHA) * self.steer_smoothed + STEER_SMOOTH_ALPHA * steer_raw
        steer_int = int(np.clip(self.steer_smoothed, -1.0, 1.0) * MAX_STEER)

        left = steer_int < -2000
        right = steer_int > 2000
        
        # Пороги
        accel_on = throttle_p > 0.5
        brake_on = brake_p > 0.8

        # Launch Control
        speed_kmh = rec["speed_norm"] * 3.6
        if speed_kmh < 5.0 and not brake_on:
            accel_on = True

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
            
            s_vis = int((steer_raw + 1) / 2 * 10)
            s_vis = max(0, min(10, s_vis))
            bar = ["."] * 11
            bar[s_vis] = "O"
            bar_str = "".join(bar)
            
            print(f"\r[{bar_str}] T:{throttle_p:.2f} B:{brake_p:.2f} | Err: {current_dist_error:.2f}m", end="")

def main():
    server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    run_client(GhostFollowingPlayer(MODEL_PATH, GHOST_FILE), server_name)

if __name__ == "__main__":
    main()