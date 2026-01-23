import os
import sys
import json
import threading
import queue
from pathlib import Path
from datetime import datetime


import numpy as np

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client


# ==========================
# Настройки
# ==========================
LOG_DIR = "tmi_logs01"
os.makedirs(LOG_DIR, exist_ok=True)
QUEUE_MAX = 100000

DEFAULT_LOGMODE = "full"      # min / mid / full
DEFAULT_LOG_EVERY_MS = 20    # 10=каждый тик
PRINT_EVERY_MS = 1000

ROTATE_ON_TIME_RESET = True  # новый файл при рестарте/сбросе времени


# ==========================
# helpers
# ==========================
def vec_norm(v):
    try:
        return float(np.linalg.norm(np.array(v, dtype=np.float32)))
    except Exception:
        return None


def f3(v):
    try:
        return [float(v[0]), float(v[1]), float(v[2])]
    except Exception:
        return None


def safe_get(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


# ==========================
# writer thread (с ротацией файлов)
# ==========================
class WriterThread(threading.Thread):
    def __init__(self, q: queue.Queue, first_path: Path):
        super().__init__(daemon=True)
        self.q = q
        self.path = first_path
        self._stop = threading.Event()
        self._file = None

    def stop(self):
        self._stop.set()

    def _open(self, path: Path):
        if self._file:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
        self.path = path
        self._file = open(self.path, "w", encoding="utf-8")

    def run(self):
        self._open(self.path)

        while not self._stop.is_set():
            try:
                rec = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            # Спец-команда: ротация файла
            if isinstance(rec, dict) and rec.get("_cmd") == "rotate":
                new_path = Path(rec["path"])
                self._open(new_path)
                continue

            # обычная запись
            try:
                self._file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                # если вдруг запись упала — не валим поток, просто пропускаем
                pass


# ==========================
# client
# ==========================
class MidLoggerEpisodes(Client):
    def __init__(self):
        super().__init__()

        self.autopilot = False
        self.logmode = DEFAULT_LOGMODE
        self.log_every_ms = DEFAULT_LOG_EVERY_MS

        self.q = queue.Queue(maxsize=QUEUE_MAX)
        self.writer = None

        self.session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_id = 0

        self.last_log_time = -10**9
        self.last_print_time = -10**9

        self.prev_t = None
        self.prev_race_time = None

        self.current_log_path = None

    # ---------- файлы/эпизоды ----------
    def _make_log_path(self):
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        return Path(LOG_DIR) / f"tmi_{self.logmode}_{self.session_ts}_ep{self.episode_id:03d}.jsonl"

    def _rotate_episode(self, iface: TMInterface, reason: str):
        self.episode_id += 1
        self.last_log_time = -10**9
        self.prev_t = None
        self.prev_race_time = None

        self.current_log_path = self._make_log_path()
        try:
            self.q.put_nowait({"_cmd": "rotate", "path": str(self.current_log_path)})
        except queue.Full:
            # если очередь переполнена — тогда просто не ротируем прямо сейчас
            pass

        iface.log(f"[logger] NEW EPISODE #{self.episode_id} ({reason})", "success")
        iface.log(f"[logger] file: {self.current_log_path}", "log")

    # ---------- TMInterface hooks ----------
    def on_registered(self, iface: TMInterface) -> None:
        print(f"[OK] Registered to {iface.server_name}")

        # стартуем эпизод 0
        self.episode_id = 0
        self.current_log_path = self._make_log_path()
        print(f"[LOG] {self.current_log_path}")

        iface.register_custom_command("autopilot")
        iface.register_custom_command("logmode")
        iface.register_custom_command("lograte")
        iface.register_custom_command("loginfo")
        iface.register_custom_command("newrun")  # ручной старт нового файла

        iface.log("Commands:", "success")
        iface.log("  autopilot on/off", "log")
        iface.log("  logmode min/mid/full", "log")
        iface.log("  lograte <ms>   (lograte 20)", "log")
        iface.log("  loginfo", "log")
        iface.log("  newrun ", "log")

        self.writer = WriterThread(self.q, self.current_log_path)
        self.writer.start()

    def on_shutdown(self, iface: TMInterface):
        self._stop_writer()

    def on_deregistered(self, iface: TMInterface):
        self._stop_writer()

    def _stop_writer(self):
        if self.writer:
            self.writer.stop()
            self.writer = None
            print("[LOG] writer stopped")

    def on_custom_command(self, iface, time_from: int, time_to: int, command: str, args: list):
        cmd = command.lower().strip()

        if cmd == "autopilot":
            if not args:
                iface.log("Usage: autopilot on/off", "warning")
                return
            v = args[0].strip().lower()
            if v in ("on", "1", "true", "yes"):
                self.autopilot = True
                iface.log("Autopilot = ON (gas full, steer=0)", "success")
            elif v in ("off", "0", "false", "no"):
                self.autopilot = False
                iface.log("Autopilot = OFF (you control)", "success")
            else:
                iface.log("Usage: autopilot on/off", "warning")
            return

        if cmd == "logmode":
            if not args:
                iface.log("Usage: logmode min/mid/full", "warning")
                return
            v = args[0].strip().lower()
            if v not in ("min", "mid", "full"):
                iface.log("Usage: logmode min/mid/full", "warning")
                return
            self.logmode = v
            iface.log(f"logmode = {self.logmode}", "success")
            return

        if cmd == "lograte":
            if not args:
                iface.log("Usage: lograte <ms>", "warning")
                return
            try:
                ms = int(args[0])
                ms = max(10, min(ms, 200))
                self.log_every_ms = ms
                iface.log(f"lograte = {self.log_every_ms} ms", "success")
            except ValueError:
                iface.log("lograte expects integer ms (например 20)", "error")
            return

        if cmd == "loginfo":
            iface.log(f"logmode={self.logmode}, lograte={self.log_every_ms}ms, autopilot={self.autopilot}", "log")
            iface.log(f"episode={self.episode_id}", "log")
            iface.log(f"logfile={self.current_log_path}", "log")
            iface.log(f"queue_size={self.q.qsize()}/{QUEUE_MAX}", "log")
            return

        if cmd == "newrun":
            self._rotate_episode(iface, "manual newrun")
            return

    def on_run_step(self, iface: TMInterface, t: int):
        if t < 0:
            return

        # автопилот (опционально)
        if self.autopilot:
            iface.set_input_state(
                sim_clear_buffer=False,
                accelerate=True,
                brake=False,
                left=False,
                right=False,
                steer=0,
                gas=65536,
            )

        # Снять state (иногда может не дать player info — но базу обычно даёт)
        try:
            s = iface.get_simulation_state()
            p_info = getattr(s, "player_info", None)
        except Exception:
            return
        

        race_time = int(safe_get(s, "race_time", 0))

        is_finished = False
        if p_info:
            # race_finished - это флаг в памяти игры
            is_finished = getattr(p_info, "race_finished", False)

        # ✅ Детект “новый заезд”: время откатилось назад
        # Это главный фикс “залипания”
        if self.prev_t is not None and t < self.prev_t:
            self.last_log_time = -10**9  # чтобы сразу начать писать снова
            if ROTATE_ON_TIME_RESET:
                self._rotate_episode(iface, "time reset (t decreased)")
        if self.prev_race_time is not None and race_time < self.prev_race_time:
            self.last_log_time = -10**9
            if ROTATE_ON_TIME_RESET:
                self._rotate_episode(iface, "time reset (race_time decreased)")

        self.prev_t = t
        self.prev_race_time = race_time

        # частота логирования
        if t - self.last_log_time < self.log_every_ms:
            return
        self.last_log_time = t

        # базовое
        pos = safe_get(s, "position")
        vel = safe_get(s, "velocity")
        ypr = safe_get(s, "yaw_pitch_roll")

        rec = {
            "episode_id": int(self.episode_id),
            "time_ms": int(t),
            "finished": int(is_finished),
            # "race_time": race_time,
            # "display_speed": int(safe_get(s, "display_speed", 0)),
            

            "position": f3(pos),
            "velocity": f3(vel),
            "speed_norm": vec_norm(vel),
            # "yaw_pitch_roll": f3(ypr),

            "inputs": {
                "accelerate": bool(safe_get(s, "input_accelerate", False)),
                "brake": bool(safe_get(s, "input_brake", False)),
                "left": bool(safe_get(s, "input_left", False)),
                "right": bool(safe_get(s, "input_right", False)),
                # "steer": int(safe_get(s, "input_steer", 0)),
                # "gas": int(safe_get(s, "input_gas", 0)),
            },

            "autopilot": bool(self.autopilot),
            "logmode": self.logmode,
        }

        # MID / FULL
        if self.logmode in ("mid", "full"):
            scene = safe_get(s, "scene_mobil")
            if scene is not None:
                engine = safe_get(scene, "engine")
                sync_state = safe_get(scene, "sync_vehicle_state")

                rec["car"] = {
                    "turning_rate": float(safe_get(scene, "turning_rate", 0.0)),
                    "is_sliding": bool(safe_get(scene, "is_sliding", False)),
                    "turbo_type": int(safe_get(scene, "turbo_type", 0)),
                    "turbo_boost_factor": float(safe_get(scene, "turbo_boost_factor", 0.0)),
                    "current_local_speed": f3(safe_get(scene, "current_local_speed")),
                }

                if engine is not None:
                    rec["engine"] = {
                        "actual_rpm": float(safe_get(engine, "actual_rpm", 0.0)),
                        "clamped_rpm": float(safe_get(engine, "clamped_rpm", 0.0)),
                        # "max_rpm": float(safe_get(engine, "max_rpm", 0.0)),
                        "gear": int(safe_get(engine, "gear", 0)),
                        # "rear_gear": int(safe_get(engine, "rear_gear", 0)),
                        # "slide_factor": float(safe_get(engine, "slide_factor", 0.0)),
                        # "braking_factor": float(safe_get(engine, "braking_factor", 0.0)),
                    }

                if sync_state is not None:
                    rec["sync"] = {
                        "speed_forward": float(safe_get(sync_state, "speed_forward", 0.0)),
                        "speed_sideward": float(safe_get(sync_state, "speed_sideward", 0.0)),
                        "rpm": float(safe_get(sync_state, "rpm", 0.0)),
                        "is_turbo": bool(safe_get(sync_state, "is_turbo", False)),
                        "gearbox_state": int(safe_get(sync_state, "gearbox_state", 0)),
                    }

        # FULL: dyna (без матриц)
        if self.logmode == "full":
            dyna = safe_get(s, "dyna")
            cur = safe_get(dyna, "current_state") if dyna is not None else None
            if cur is not None:
                qv = safe_get(cur, "quat", [0, 0, 0, 1])
                rec["dyna"] = {
                    "quat": [float(qv[0]), float(qv[1]), float(qv[2]), float(qv[3])] if qv else None,
                    # "linear_speed": f3(safe_get(cur, "linear_speed")),
                    "angular_speed": f3(safe_get(cur, "angular_speed")),
                    "force": f3(safe_get(cur, "force")),
                    "torque": f3(safe_get(cur, "torque")),
                }

        # очередь записи
        try:
            self.q.put_nowait(rec)
        except queue.Full:
            pass

        # печать редко
        if t - self.last_print_time >= PRINT_EVERY_MS:
            self.last_print_time = t
            sp = rec.get("speed_norm") or 0.0
            print(f"[ep={self.episode_id} t={t}] mode={self.logmode} rate={self.log_every_ms}ms speed={sp:.2f} pos={rec['position']}")


def main():
    server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    print(f"Connecting to {server_name}...")
    run_client(MidLoggerEpisodes(), server_name)


if __name__ == "__main__":
    main()
