import sys
import numpy as np

from tminterface.client import Client, run_client
from tminterface.interface import TMInterface

from agents.agent_core import TrackmaniaAgent, AgentConfig
from models.policy import ActorCritic
from utils.ghost_runtime import load_ghost_npz
from utils.state import extract_state
import torch
import os
# ==========================
# LOADERS
# ==========================
def load_model(
    obs_dim: int,
    path: str = "models/rl_policy.pt",
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

    # ---------- TM hooks ----------
    def on_registered(self, iface: TMInterface):
        print("[OK] RL agent connected")

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

        print(
            f"ACT raw: steer={action[0]:.3f}, "
            f"thr={action[1]:.3f}, "
            f"br={action[2]:.3f}"
        )

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
