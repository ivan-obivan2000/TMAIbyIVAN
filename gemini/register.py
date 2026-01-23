import sys
import json
import threading
import queue
import time
import os
from pathlib import Path
from datetime import datetime
import numpy as np

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

# === НАСТРОЙКИ ===
LOG_DIR = "tmi_logs_gemini"
QUEUE_MAX = 100000
LOG_EVERY_MS = 10  # Пишем каждые 10мс (100 Гц)

# Создаем папку
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Вспомогательные функции
def vec_norm(v):
    try: return float(np.linalg.norm(np.array(v, dtype=np.float32)))
    except: return 0.0

def f3(v):
    try: return [float(v[0]), float(v[1]), float(v[2])]
    except: return [0.0, 0.0, 0.0]

# Поток записи в файл (чтобы не тормозить игру)
class WriterThread(threading.Thread):
    def __init__(self, q, path):
        super().__init__(daemon=True)
        self.q = q
        self.path = path
        self._file = open(self.path, "w", encoding="utf-8")

    def run(self):
        while True:
            rec = self.q.get()
            if rec is None: break # Команда на выход
            try:
                self._file.write(json.dumps(rec) + "\n")
                self._file.flush() # Принудительно сохраняем на диск
            except: pass
        self._file.close()

class DataRecorder(Client):
    def __init__(self):
        super().__init__()
        self.q = queue.Queue(maxsize=QUEUE_MAX)
        
        # Генерируем имя файла
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = Path(LOG_DIR) / f"session_{ts}.jsonl"
        
        self.writer = WriterThread(self.q, self.path)
        self.writer.start()
        
        self.episode_id = 0
        self.last_log_time = -10000
        self.prev_game_time = 0
        
        print(f"[REC] Запись идет в файл: {self.path}")
        print("[REC] Едь трассу. Жми рестарт когда нужно. Скрипт сам поймет.")

    def on_registered(self, iface):
        print(f"[CONN] Подключено к {iface.server_name}")

    def on_run_step(self, iface, t):
        # 1. Детектор рестарта
        # Если текущее время меньше предыдущего (нажал рестарт)
        # ИЛИ если время ушло в минус (отсчет)
        if t < self.prev_game_time and t < 1000: 
            self.episode_id += 1
            print(f"\n[INFO] Рестарт обнаружен! Новый эпизод: {self.episode_id}")
            self.last_log_time = -10000 # Сброс таймера лога
        
        self.prev_game_time = t

        # Не пишем данные во время отсчета (3..2..1)
        if t < 0: return

        # Ограничиваем частоту записи (каждые 10мс)
        if t - self.last_log_time < LOG_EVERY_MS:
            return
        self.last_log_time = t

        try: s = iface.get_simulation_state()
        except: return

        # === ЛОГИКА КЛАВИАТУРЫ И ГЕЙМПАДА ===
        raw_steer = getattr(s, "input_steer", 0)
        is_left = getattr(s, "input_left", False)
        is_right = getattr(s, "input_right", False)
        
        final_steer = raw_steer
        # Если аналог молчит, проверяем кнопки
        if raw_steer == 0:
            if is_left: final_steer = -65536
            elif is_right: final_steer = 65536

        raw_gas = getattr(s, "input_gas", 0)
        is_accel = getattr(s, "input_accelerate", False)
        final_gas = raw_gas
        if raw_gas == 0 and is_accel:
            final_gas = 65536

        # Достаем физику
        scene = getattr(s, "scene_mobil", None)
        dyna = getattr(s, "dyna", None)
        curr_dyna = getattr(dyna, "current_state", None) if dyna else None
        engine = getattr(scene, "engine", None) if scene else None

        # Формируем запись
        rec = {
            "ep": self.episode_id,  # ВАЖНО: Номер попытки
            "t": t,
            
            # Входные данные (что видит бот)
            "vel": f3(getattr(s, "velocity", [0,0,0])),
            "spd": vec_norm(getattr(s, "velocity", [0,0,0])), # Абсолютная скорость
            "ypr": f3(getattr(s, "yaw_pitch_roll", [0,0,0])),
            "loc_v": f3(getattr(scene, "current_local_speed", [0,0,0])) if scene else [0,0,0], # Важно для дрифта
            "ang_v": f3(getattr(curr_dyna, "angular_speed", [0,0,0])) if curr_dyna else [0,0,0], # Важно для стабилизации
            "slide": 1 if getattr(scene, "is_sliding", False) else 0,
            "rpm": float(getattr(engine, "actual_rpm", 0.0)) if engine else 0.0,
            "gear": float(getattr(engine, "gear", 0.0)) if engine else 0.0,

            # Выходные данные (что делает бот)
            "in_steer": int(final_steer),
            "in_gas": int(final_gas),
            "in_brake": 1 if getattr(s, "input_brake", False) else 0
        }
        
        self.q.put(rec)
        
        # Вывод в консоль
        sys.stdout.write(f"\rEp: {self.episode_id} | Time: {t/1000:.1f}s | Speed: {int(rec['spd']*3.6)} km/h | Input: {final_steer}   ")
        sys.stdout.flush()

    def on_shutdown(self, iface):
        self.q.put(None) # Сигнал стоп
        self.writer.join()
        print(f"\n[STOP] Файл сохранен: {self.path}")

def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    run_client(DataRecorder(), server_name)

if __name__ == '__main__':
    main()