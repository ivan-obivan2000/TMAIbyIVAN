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
MODEL_PATH = "self_learned_model.pt"
LOG_DIR = "self_learning_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Обучение
BATCH_SIZE = 128
LR = 1e-4
TRAIN_EPOCHS_PER_GEN = 3
MEMORY_SIZE = 100000

# Награды (Rewards)
REWARD_SPEED = 2.0        
REWARD_DISTANCE = 0.1     
REWARD_CRASH = -400.0      
REWARD_SMOOTH = -0.5     
REWARD_FINISH = 1000.0    

# Exploration
EXPLORATION_NOISE = 0.2   

# Forced Start
FORCE_GAS_MS = 2000       

# Логирование
QUEUE_MAX = 100000

# ==========================
# WRITER THREAD (Для логов)
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
            
            # Спец-команда: ротация файла
            if isinstance(rec, dict) and rec.get("_cmd") == "rotate":
                new_path = Path(rec["path"])
                self._open(new_path)
                continue
            
            # Обычная запись
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

    def save_episode(self, episode_data):
        gamma = 0.99
        running_add = 0
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
            except:
                print("Error loading model, starting fresh.")
        
        self.current_episode = []
        self.last_pos = None
        self.last_steer = 0.0
        self.total_reward = 0.0
        self.episode_count = 0
        self.is_training = False
        self.best_reward = -10000.0

        # Логгер
        self.q = queue.Queue(maxsize=QUEUE_MAX)
        self.writer = None
        self.session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_path = None

    def _make_log_path(self):
        return Path(LOG_DIR) / f"self_{self.session_ts}_ep{self.episode_count:03d}.jsonl"

    def on_registered(self, iface: TMInterface):
        print("[OK] Self-Learner Ready.")
        iface.register_custom_command("save")
        
        # Запускаем логгер
        self.current_log_path = self._make_log_path()
        self.writer = WriterThread(self.q, self.current_log_path)
        self.writer.start()
        print(f"[LOG] Writing to {self.current_log_path}")

    def on_shutdown(self, iface):
        if self.writer: self.writer.stop()

    def on_run_step(self, iface: TMInterface, t: int):
        if t < 0 or self.is_training: return

        try:
            s = iface.get_simulation_state()
            scene = getattr(s, "scene_mobil", None)
            if not scene: return
        except: return

        # 1. Observation
        obs = self.make_obs(s)
        
        # 2. Action Selection
        if t < FORCE_GAS_MS:
            # === FORCED START ===
            steer_val = 0.0 
            throttle_val = 1.0 
            brake_val = 0.0
        else:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                steer, tb = self.model(obs_t)
            
            noise = np.random.normal(0, EXPLORATION_NOISE)
            steer_val = np.clip(float(steer.item()) + noise, -1.0, 1.0)
            
            # --- ХАК: Всегда газ, если сеть еще тупая ---
            raw_gas = float(tb[0,0])
            throttle_val = 1.0 if raw_gas > 0.3 else raw_gas
            brake_val = 0.0 # Отключаем тормоз пока учимся ехать вперед

        # 3. Reward Calculation
        reward = self.calculate_reward(s, steer_val)
        self.total_reward += reward

        action_vec = np.array([steer_val, throttle_val, brake_val])
        
        # 4. Store Step (Memory)
        self.current_episode.append({
            'obs': obs,
            'act': action_vec,
            'reward': reward
        })

        # 5. LOGGING (Запись в файл .jsonl)
        # Пишем только каждый 5-й кадр (100мс), чтобы не забивать диск
        if t % 100 == 0:
            rec = {
                "time": int(t),
                "reward": float(reward),
                "speed": float(np.linalg.norm(self.f3(s.velocity)) * 3.6),
                "pos": self.f3(s.position),
                "act": [float(steer_val), float(throttle_val), float(brake_val)]
            }
            try: self.q.put_nowait(rec)
            except: pass

        # 6. Apply Control
        iface.set_input_state(
            sim_clear_buffer=False,
            accelerate=throttle_val > 0.5,
            brake=brake_val > 0.5,
            left=steer_val < -0.1,
            right=steer_val > 0.1,
            steer=int(steer_val * 65536),
            gas=65536 if throttle_val > 0.5 else 0
        )

        # 7. Check End
        # Если рестарт (время сбросилось) или финиш
        if s.race_time < 100 and len(self.current_episode) > 20:
            self.end_episode(iface)

        self.last_pos = np.array(self.f3(s.position))
        self.last_steer = steer_val

    def make_obs(self, s):
        vel = np.array(self.f3(s.velocity))
        speed = np.linalg.norm(vel)
        ypr = self.f3(s.yaw_pitch_roll)
        
        feat = [
            vel[0]/100, vel[1]/100, vel[2]/100,
            speed/100,
            ypr[0], ypr[1], ypr[2],
        ] + [0.0] * 12 
        return np.array(feat[:19], dtype=np.float32)

    def calculate_reward(self, s, steer):
        vel = np.array(self.f3(s.velocity))
        speed_kmh = np.linalg.norm(vel) * 3.6
        
        r = 0.0
        # + Скорость (основной драйвер)
        r += (speed_kmh / 100.0) * REWARD_SPEED
        
        # + Дистанция
        if self.last_pos is not None:
            pos = np.array(self.f3(s.position))
            dist = np.linalg.norm(pos - self.last_pos)
            r += dist * REWARD_DISTANCE

        # - Плавность
        r += -abs(steer - self.last_steer) * 0.1
        return r

    def end_episode(self, iface):
        self.episode_count += 1
        print(f"Ep {self.episode_count}: Reward {self.total_reward:.1f}")

        # Сохранение в память для обучения
        if self.total_reward > 50.0: 
            self.memory.save_episode(self.current_episode)
            if self.total_reward > self.best_reward:
                self.best_reward = self.total_reward
                print(f"!!! NEW BEST REWARD: {self.best_reward:.1f} !!!")

        # Ротация лога (создаем новый файл)
        new_path = self._make_log_path()
        try: self.q.put_nowait({"_cmd": "rotate", "path": str(new_path)})
        except: pass

        # Обучение каждые 5 эпизодов
        if self.episode_count % 5 == 0 and len(self.memory.buffer) > BATCH_SIZE:
            # Чтобы не блокировать игру надолго, можно вынести train в отдельный поток,
            # но пока оставим синхронно
            self.train_step()
            
            # === AUTO-RESTART ===
            # Если мы только что обучились, полезно начать заезд заново
            print("Restarting track...")
            iface.give_up() # Это нажимает "Give Up" (Restart)

        self.current_episode = []
        self.total_reward = 0.0
        self.last_pos = None

    def train_step(self):
        print("Training...", end="")
        self.is_training = True
        self.model.train()
        
        total_loss = 0
        steps = 0
        
        for _ in range(TRAIN_EPOCHS_PER_GEN):
            obs, act, returns = self.memory.sample(BATCH_SIZE)
            
            obs_t = torch.FloatTensor(obs)
            act_t = torch.FloatTensor(act)
            returns_t = torch.FloatTensor(returns).unsqueeze(1)
            
            pred_steer, pred_tb = self.model(obs_t)
            
            # Loss взвешенный на награду
            weights = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-5)
            weights = torch.clamp(weights, 0.0, 5.0)
            
            loss_steer = (nn.MSELoss(reduction='none')(pred_steer, act_t[:, 0:1]) * weights).mean()
            loss_tb = (nn.MSELoss(reduction='none')(pred_tb, act_t[:, 1:3]) * weights).mean()
            
            loss = loss_steer + loss_tb
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1

        self.model.eval()
        torch.save(self.model.state_dict(), MODEL_PATH)
        print(f" Done. Loss: {total_loss/steps:.4f}")
        self.is_training = False

    def f3(self, v):
        try: return [v[0], v[1], v[2]]
        except: return [0,0,0]

def main():
    server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    run_client(SelfLearningAgent(), server_name)

if __name__ == "__main__":
    main()