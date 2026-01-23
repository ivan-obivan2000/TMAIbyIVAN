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
MODEL_PATH = "model_track_3.pt" # Убедись, что имя совпадает с train.py
GHOST_FILE = "ghost_data_3.npz"          # Файл с мульти-призраком

CONTROL_EVERY_MS = 20
PRINT_EVERY_MS = 100
STEER_SMOOTH_ALPHA = 0.15
MAX_STEER = 65536

# Принудительный газ на старте (мс)
FORCE_GAS_TIME_MS = 1500 

# ==========================
# МОДЕЛЬ (Копия из train.py)
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
        
        # Читаем конфиг Lookahead
        self.lookahead_cfg = ckpt.get("lookahead_cfg", {"min": 4, "max": 12, "factor": 0.1})
        print(f"Конфиг: K={self.K}, Lookahead={self.lookahead_cfg}")

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

    # --- ИСПРАВЛЕННАЯ ФУНКЦИЯ make_obs_vec (6D + Dynamic Lookahead) ---
    def make_obs_vec(self, r):
        # Текущее состояние
        p = np.array(r["position"], dtype=np.float32)
        v = np.array(r["velocity"], dtype=np.float32)

        # 1. Формируем 6D запрос [Pos, Vel]
        query = np.concatenate([p, v])

        # 2. Ищем ближайшего соседа по 6 измерениям (numpy)
        diff = self.ghost_6d - query
        dists_sq = np.sum(diff**2, axis=1)
        idx = np.argmin(dists_sq)
        
        # Дистанция для логов
        dist_scalar = np.sqrt(dists_sq[idx])

        # 3. DYNAMIC LOOKAHEAD
        speed = np.linalg.norm(v)
        
        # Считаем шаги: min + speed * factor
        dyn_steps = int(self.lookahead_cfg["min"] + speed * self.lookahead_cfg["factor"])
        # Ограничиваем максимумом
        dyn_steps = min(dyn_steps, self.lookahead_cfg["max"])
        
        # Защита от конца массива
        remaining_steps = len(self.ghost_pos) - 1 - idx
        actual_steps = min(dyn_steps, remaining_steps)

        # Индекс цели
        target_idx = idx + actual_steps

        target_p = self.ghost_pos[target_idx]
        target_v = self.ghost_vel[target_idx]

        # 4. Вектора ошибок (относительно будущей точки!)
        delta_p = target_p - p
        delta_v = target_v - v

        # 5. Heading Error (Куда смотрит нос?)
        v_norm = v / (speed + 1e-6)
        tv_norm = target_v / (np.linalg.norm(target_v) + 1e-6)
        heading_err = np.dot(v_norm, tv_norm)

        # Остальные данные
        car = r.get("car", {})
        eng = r.get("engine", {})
        dyna = r.get("dyna", {})
        
        quat = dyna.get("quat") or [0, 0, 0, 1] 
        ang_vel = dyna.get("angular_speed") or [0, 0, 0]
        loc_v = car.get("current_local_speed") or [0, 0, 0]

        # 6. Собираем вектор (ТОЧНО КАК В TRAIN.PY)
        feat = [
            delta_p[0], delta_p[1], delta_p[2],
            delta_v[0], delta_v[1], delta_v[2],

            v[0], v[1], v[2],
            r.get("speed_norm", 0) / 100.0,
            
            heading_err, # <--- NEW FIELD

            quat[0], quat[1], quat[2], quat[3],
            ang_vel[0], ang_vel[1], ang_vel[2],

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

        # === HARD FORCE GAS BEFORE MODEL / HIST ===
        if race_time < FORCE_GAS_TIME_MS:
            iface.set_input_state(
                sim_clear_buffer=False,
                accelerate=True,
                brake=False,
                left=False,
                right=False,
                steer=0,
                gas=65536,
            )
            return


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
        
        # 1. Принудительный старт
        # Если время гонки < 2000 мс (2 секунды), жмем газ в пол и не рулим (или рулим как сеть говорит)
        if race_time < FORCE_GAS_TIME_MS:
            accel_on = True
            brake_on = False
            # Можно блокировать руль на 0, если нужно, или оставить как есть:
            # steer_raw = 0.0 
        else:
            # Обычная логика
            accel_on = throttle_p > 0.5
            brake_on = brake_p > 0.5
            # Launch Control (для застреваний)
            speed_kmh = rec["speed_norm"] * 3.6
            if speed_kmh < 5.0 and not brake_on:
                accel_on = True

        self.steer_smoothed = (1 - STEER_SMOOTH_ALPHA) * self.steer_smoothed + STEER_SMOOTH_ALPHA * steer_raw
        steer_int = int(np.clip(self.steer_smoothed, -1.0, 1.0) * MAX_STEER)

        left = steer_int < -2000
        right = steer_int > 2000

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
            
            forced_str = " (FORCED)" if race_time < FORCE_GAS_TIME_MS else ""
            print(f"\r[{bar_str}] T:{throttle_p:.2f} B:{brake_p:.2f} | Err: {current_dist_error:.2f}m{forced_str}", end="")

def main():
    server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    run_client(GhostFollowingPlayer(MODEL_PATH, GHOST_FILE), server_name)

if __name__ == "__main__":
    main()