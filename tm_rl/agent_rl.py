import sys
import numpy as np
import queue

from tminterface.client import Client, run_client
from tminterface.interface import TMInterface

from agents.agent_core import TrackmaniaAgent, AgentConfig
from models.policy import ActorCritic
from train_rl import ppo_update
from utils.buffer import RolloutBuffer
from utils.ghost_runtime import load_ghost_npz
from utils.log_writer import WriterThread
from utils.state import extract_state
import torch
import os
from pathlib import Path
from datetime import datetime
# ==========================
# CONFIG
# ==========================
LOG_DIR = "tr_tl_logs"
MODEL_PATH = "models/rl_policy.pt"
LOG_EVERY_STEPS = 200
UPDATE_EVERY_STEPS = 1024
LR = 3e-4

# ==========================
# LOADERS
# ==========================
def load_model(
    obs_dim: int,
    path: str = MODEL_PATH,
    device: str = "cpu",
):
    model = ActorCritic(obs_dim=obs_dim).to(device)

    if os.path.exists(path):
        print(f"[MODEL] loading weights from {path}")
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        print("[MODEL] no checkpoint found → starting from scratch")

    model.eval()

  
    return model



# ==========================
# RL CLIENT
# ==========================
class RLClient(Client):
    def __init__(self):
        super().__init__()
        
        # Вынесите импорты наверх файла или убедитесь, что структура папок верна.
        # Предполагаем, что obs_vec и LookaheadConfig доступны.
        from env.obs import obs_vec
        from utils.ghost import LookaheadConfig

        # Проверка наличия файла призрака
        ghost_path = "ghost_data01.npz"
        if not os.path.exists(ghost_path):
            print(f"[ERROR] Ghost file {ghost_path} not found!")
            sys.exit(1)

        self.ghost = load_ghost_npz(ghost_path)
        self.lookahead = LookaheadConfig()

        
        # Конфигурация истории
        K = self.lookahead.history  # Лучше брать из конфига, если там есть, или 10
        if not hasattr(self.lookahead, 'history'):
             K = 10 # Fallback если в конфиге нет поля

        # ---- dummy state для расчета размерности ----
        dummy_state = {
            "position": np.zeros(3, dtype=np.float32),
            "velocity": np.zeros(3, dtype=np.float32),
            "speed_norm": 0.0,
            "car": {}, "engine": {}, "dyna": {},
            "inputs": {} # obs_vec может обращаться к inputs
        }

        # Расчет размерности
        single_obs, _ = obs_vec(dummy_state, self.ghost, self.lookahead)
        BASE_OBS_DIM = len(single_obs)
        OBS_DIM = BASE_OBS_DIM * K

        print(f"[OBS] single={BASE_OBS_DIM}, K={K}, total={OBS_DIM}")

        # Инициализация агента
        self.agent = TrackmaniaAgent(
            model=load_model(obs_dim=OBS_DIM),
            ghost=self.ghost,
            # Важно передать конфиг с правильным lookahead
            cfg=AgentConfig(lookahead=self.lookahead) 
        )

        self.prev_t = None
        self.prev_race_time = None
        self.buffer = RolloutBuffer()
        self.optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=LR)
        self.global_steps = 0
        self.episode_reward = 0.0
        self.episode_steps = 0

        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(Path(MODEL_PATH).parent, exist_ok=True)
        log_name = f"rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.log_path = Path(LOG_DIR) / log_name
        self.log_queue = None
        self.log_writer = None

    # ---------- TM hooks ----------
    def on_registered(self, iface: TMInterface):
        print("[OK] RL agent connected")
        self.log_queue = queue.Queue(maxsize=10000)
        self.log_writer = WriterThread(self.log_queue, self.log_path)
        self.log_writer.start()

    def on_shutdown(self, iface: TMInterface):
        if self.log_writer:
            self.log_writer.stop()

    def on_run_step(self, iface: TMInterface, t: int):
        if t < 0:
            return

        try:
            s = iface.get_simulation_state()
        except Exception:
            return

        race_time = int(getattr(s, "race_time", 0))

        # ===== restart detection =====
        if (
            (self.prev_t is not None and t < self.prev_t)
            or (self.prev_race_time is not None and race_time < self.prev_race_time)
        ):
            self.agent.reset()

        self.prev_t = t
        self.prev_race_time = race_time

        # ===== build state =====
        state = extract_state(s)

        # ===== agent step =====
        controls, action, transition = self.agent.act(
            state=state,
            race_time_ms=int(s.race_time),
        )

        if transition:
            obs_t = torch.tensor(transition["obs"], dtype=torch.float32)
            act_t = torch.tensor(action, dtype=torch.float32)
            logprob_t = torch.tensor(transition["logprob"], dtype=torch.float32)
            value_t = torch.tensor(transition["value"], dtype=torch.float32)
            reward = float(transition["reward"])
            done = bool(transition["done"])

            self.buffer.add(
                obs=obs_t,
                action=act_t,
                logprob=logprob_t,
                reward=reward,
                value=value_t,
                done=done,
            )

            self.global_steps += 1
            self.episode_reward += reward
            self.episode_steps += 1

            if self.global_steps % LOG_EVERY_STEPS == 0:
                rec = {
                    "step": self.global_steps,
                    "reward": reward,
                    "episode_reward": self.episode_reward,
                    "episode_steps": self.episode_steps,
                    "speed": float(state.get("speed_norm", 0.0)),
                }
                if self.log_queue:
                    try:
                        self.log_queue.put_nowait(rec)
                    except Exception:
                        pass

            if self.global_steps % UPDATE_EVERY_STEPS == 0:
                self.agent.model.train()
                metrics = ppo_update(
                    model=self.agent.model,
                    buffer=self.buffer,
                    optimizer=self.optimizer,
                )
                self.agent.model.eval()
                torch.save(self.agent.model.state_dict(), MODEL_PATH)
                self.buffer.clear()
                if self.log_queue:
                    try:
                        self.log_queue.put_nowait(
                            {
                                "step": self.global_steps,
                                "update": True,
                                "loss": metrics["loss"],
                                "actor_loss": metrics["actor_loss"],
                                "critic_loss": metrics["critic_loss"],
                            }
                        )
                    except Exception:
                        pass

        # если агент ещё накапливает историю — пусть газует прямо сейчас
        if not controls:
            iface.set_input_state(
                accelerate=True,
                brake=False,
                left=False,
                right=False,
                steer=0,
            )
            return

        # ===== apply controls =====
        iface.set_input_state(
            accelerate=controls["accelerate"],
            brake=controls["brake"],
            left=controls["left"],
            right=controls["right"],
            steer=controls["steer"],   # <-- важно
        )

        # ===== episode end =====
        if transition and transition["done"]:
            self.agent.reset()
            if self.log_queue:
                try:
                    self.log_queue.put_nowait(
                        {
                            "episode": True,
                            "step": self.global_steps,
                            "episode_reward": self.episode_reward,
                            "episode_steps": self.episode_steps,
                        }
                    )
                except Exception:
                    pass
            self.episode_reward = 0.0
            self.episode_steps = 0


# ==========================
# RUN
# ==========================
def main():
    server_name = (
        f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    )
    print(f"Connecting to {server_name}...")
    run_client(RLClient(), server_name)


if __name__ == "__main__":
    main()
