import os
import sys
import json
import time
import random
import threading
import queue
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from pathlib import Path
from datetime import datetime

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

# ==========================
# КОНФИГУРАЦИЯ
# ==========================
MODEL_PATH = "self_learned_model_v3.pt"
LOG_DIR = "self_learning_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Обучение
BATCH_SIZE = 256
LR = 1e-4
TRAIN_EPOCHS_PER_GEN = 4
TRAIN_EVERY_EPISODES = 5
MEMORY_SIZE = 100000

# Награды
REWARD_SPEED = 3.0
REWARD_DISTANCE = 0.2
REWARD_CRASH = -100.0
REWARD_FINISH = 2000.0
REWARD_BEST_TIME = 3000.0

# Exploration
EXPLORATION_NOISE = 0.15

# Forced Start (сколько мс жать газ на старте без руления)
FORCE_GAS_MS = 1000 

# Скорость игры
GAME_SPEED = 1

# Логирование
QUEUE_MAX = 100000

# ==========================
# WRITER THREAD
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
            except Exception: pass
        self.path = path
        self._file = open(self.path, "w", encoding="utf-8")

    def run(self):
        self._open(self.path)
        while not self._stop.is_set():
            try:
                rec = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            
            if isinstance(rec, dict) and rec.get("_cmd") == "rotate":
                new_path = Path(rec["path"])
                self._open(new_path)
                continue
            
            try:
                self._file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception: pass

# ==========================
# МОДЕЛЬ
# ==========================
class PolicyNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.head_steer = nn.Linear(128, 1)
        self.head_tb = nn.Linear(128, 2) 

    def forward(self, x):
        h = self.net(x)
        return torch.tanh(self.head_steer(h)), torch.sigmoid(self.head_tb(h))

# ==========================
# ПАМЯТЬ
# ==========================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, act, reward):
        self.buffer.append((obs, act, reward))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        obs, act, rew = zip(*batch)
        return np.array(obs), np.array(act), np.array(rew)

    def save_episode(self, episode_data, extra_reward=0.0):
        gamma = 0.99
        running_add = 0
        if extra_reward != 0:
            episode_data[-1]['reward'] += extra_reward

        for step in reversed(episode_data):
            running_add = running_add * gamma + step['reward']
            self.push(step['obs'], step['act'], running_add)

# ==========================
# АГЕНТ
# ==========================
class SelfLearningAgent(Client):
    def __init__(self):
        super().__init__()
        self.model = PolicyNet(19)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        
        if os.path.exists(MODEL_PATH):
            print(f"Loading model: {MODEL_PATH}")
            try:
                self.model.load_state_dict(torch.load(MODEL_PATH))
                self.model.eval()
            except: pass
        
        self.current_episode = []
        self.last_pos = None
        self.total_reward = 0.0
        self.episode_count = 0
        
        # Флаг состояния
        self.is_waiting_for_restart = False 
        self.stuck_counter = 0 # Счетчик попыток рестарта

        self.best_time = 999999999
        
        self.q = queue.Queue(maxsize=QUEUE_MAX)
        self.writer = None
        self.session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_path = self._make_log_path()

    def _make_log_path(self):
        return Path(LOG_DIR) / f"self_{self.session_ts}.jsonl"

    def on_registered(self, iface: TMInterface):
        print(f"[OK] Ready. Speed: {GAME_SPEED}")
        iface.set_speed(GAME_SPEED)
        self.writer = WriterThread(self.q, self.current_log_path)
        self.writer.start()

    def on_shutdown(self, iface):
        if self.writer: self.writer.stop()

    def on_run_step(self, iface: TMInterface, t: int):
        # ---------------------------------------------------------
        # 1. АКТИВНЫЙ ЦИКЛ РЕСТАРТА (THE FIX)
        # ---------------------------------------------------------
        if self.is_waiting_for_restart:
            # Если время маленькое (0-1000мс), значит рестарт прошел успешно
            if t >= 0 and t < 1500:
                self.is_waiting_for_restart = False
                self.stuck_counter = 0
                self.current_episode = []
                self.total_reward = 0.0
                self.last_pos = None
                print(f"--- Start Ep {self.episode_count + 1} ---")
                
                # Обучение пока таймер тикает 3-2-1
                if self.episode_count > 0 and self.episode_count % TRAIN_EVERY_EPISODES == 0:
                    self.train_network()
            else:
                # МЫ ЗАСТРЯЛИ (время всё еще старое или мы в меню)
                self.stuck_counter += 1
                
                # Каждые 10 тиков спамим команду
                if self.stuck_counter % 10 == 0:
                    iface.give_up() # Обычный рестарт
                
                # Если долго не помогает (например, вылетели в меню медалей)
                # Жмем Respawn (это кнопка Enter/Improve в меню)
                if self.stuck_counter > 50 and self.stuck_counter % 20 == 0:
                     iface.respawn()
                
                return # Выходим, не управляем машиной

        # ---------------------------------------------------------
        # 2. ПРОВЕРКА НА КОНЕЦ ЗАЕЗДА
        # ---------------------------------------------------------
        try:
            s = iface.get_simulation_state()
            scene = getattr(s, "scene_mobil", None)
            if not scene: return
        except: return

        is_finished = False
        p_info = getattr(s, "player_info", None)
        if p_info and getattr(p_info, "race_finished", False):
            is_finished = True
        
        should_end = is_finished or (t < 100 and len(self.current_episode) > 20)

        if should_end:
            self.finish_episode(iface, is_finished, s.race_time)
            return 

        # ---------------------------------------------------------
        # 3. УПРАВЛЕНИЕ
        # ---------------------------------------------------------
        if t < 0:
            iface.set_input_state(accelerate=True, gas=65536)
            return

        obs = self.make_obs(s)
        
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            steer, tb = self.model(obs_t)
        
        noise = np.random.normal(0, EXPLORATION_NOISE)
        steer_val = np.clip(float(steer.item()) + noise, -1.0, 1.0)
        throttle_val = 1.0 if float(tb[0,0]) > 0.3 else 0.0
        brake_val = 0.0 

        if t < FORCE_GAS_MS:
            steer_val = 0.0
            throttle_val = 1.0

        reward = self.calculate_reward(s, steer_val)
        self.total_reward += reward

        self.current_episode.append({
            'obs': obs,
            'act': np.array([steer_val, throttle_val, brake_val]),
            'reward': reward
        })
        self.last_pos = np.array(self.f3(s.position))

        iface.set_input_state(
            accelerate=throttle_val > 0.5,
            brake=brake_val > 0.5,
            left=steer_val < -0.1,
            right=steer_val > 0.1,
            steer=int(steer_val * 65536),
            gas=65536 if throttle_val > 0.5 else 0
        )

        if t % 100 == 0:
            rec = {
                "time": int(t),
                "reward": float(reward),
                "speed": float(np.linalg.norm(self.f3(s.velocity)) * 3.6),
                "act": [float(steer_val), float(throttle_val)]
            }
            try: self.q.put_nowait(rec)
            except: pass

    def finish_episode(self, iface, finished, race_time):
        self.episode_count += 1
        extra_bonus = 0.0
        status = "RESTART"

        if finished:
            status = "FINISHED"
            extra_bonus += REWARD_FINISH
            if race_time < self.best_time:
                self.best_time = race_time
                extra_bonus += REWARD_BEST_TIME
                print(f"!!! NEW RECORD: {race_time} ms !!!")

        if finished or self.total_reward > 50.0:
            self.memory.save_episode(self.current_episode, extra_bonus)

        print(f"Ep {self.episode_count} [{status}]: Reward {self.total_reward+extra_bonus:.1f}")

        # === FIX ===
        self.is_waiting_for_restart = True
        self.stuck_counter = 0 # Сброс счетчика
        
        iface.set_input_state(accelerate=False, brake=False, steer=0, gas=0)
        
        # Первая попытка
        iface.give_up()

    def train_network(self):
        if len(self.memory.buffer) < BATCH_SIZE: return
        
        print(" [Training...] ", end="")
        self.model.train()
        try:
            for _ in range(TRAIN_EPOCHS_PER_GEN):
                obs, act, returns = self.memory.sample(BATCH_SIZE)
                obs_t = torch.FloatTensor(obs)
                act_t = torch.FloatTensor(act)
                ret_t = torch.FloatTensor(returns).unsqueeze(1)
                
                p_steer, p_tb = self.model(obs_t)
                
                weights = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-5)
                weights = torch.clamp(weights, 0.0, 5.0)
                
                loss_s = (nn.MSELoss(reduction='none')(p_steer, act_t[:, 0:1]) * weights).mean()
                loss_t = (nn.MSELoss(reduction='none')(p_tb, act_t[:, 1:3]) * weights).mean()
                
                loss = loss_s + loss_t
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()
            torch.save(self.model.state_dict(), MODEL_PATH)
            print("Done.")
        except Exception as e:
            print(f"Error: {e}")

    def make_obs(self, s):
        vel = np.array(self.f3(s.velocity))
        speed = np.linalg.norm(vel)
        ypr = self.f3(s.yaw_pitch_roll)
        feat = [vel[0]/100, vel[1]/100, vel[2]/100, speed/100, ypr[0], ypr[1], ypr[2]] + [0.0]*12 
        return np.array(feat[:19], dtype=np.float32)

    def calculate_reward(self, s, steer):
        vel = np.array(self.f3(s.velocity))
        speed_kmh = np.linalg.norm(vel) * 3.6
        r = (speed_kmh / 100.0) * REWARD_SPEED
        if self.last_pos is not None:
            r += np.linalg.norm(np.array(self.f3(s.position)) - self.last_pos) * REWARD_DISTANCE
        return r

    def f3(self, v):
        try: return [v[0], v[1], v[2]]
        except: return [0,0,0]

def main():
    server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    run_client(SelfLearningAgent(), server_name)

if __name__ == "__main__":
    main()