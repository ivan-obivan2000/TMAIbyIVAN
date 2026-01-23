import sys
import csv
import os
import numpy as np
from datetime import datetime

from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

# Создаем папку
if not os.path.exists("data_logs"):
    os.makedirs("data_logs")

class DataRecorder(Client):
    def __init__(self) -> None:
        super(DataRecorder, self).__init__()
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f"data_logs/train_data_{timestamp}.csv"
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        self.headers = [
            'time',
            # --- ВЫХОДНЫЕ ДАННЫЕ (ЧТО ПРЕДСКАЗЫВАЕМ) ---
            'input_steer',
            'input_gas',
            'input_brake',
            # --- ВХОДНЫЕ ДАННЫЕ (ЧТО ВИДИТ МОДЕЛЬ) ---
            'speed_kmh',
            'speed_forward',      # Скорость по направлению движения
            'speed_sideward',     # Скорость бокового скольжения (drift speed)
            'rpm',
            'gear',
            'turbo_factor',       # Насколько сильно работает турбо
            'is_sliding',         # Булево: скользим или нет
            'wheels_contact',     # Сколько колес на земле
            'surface_id',         # Тип поверхности (асфальт/трава/грязь)
            # Координаты и ориентация (нужны для обучения на одной трассе)
            'pos_x', 'pos_y', 'pos_z',
            'vel_x', 'vel_y', 'vel_z',
            'yaw', 'pitch', 'roll',
        ]
        self.writer.writerow(self.headers)
        print(f"Запись v3 (Расширенная) началась! Файл: {self.filename}")
        print("Проедь трассу 10 раз. Если врезался — жми рестарт, плохие попытки отфильтруем потом.")

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Подключено к {iface.server_name}')

    def on_run_step(self, iface: TMInterface, _time: int):
        # Пропускаем время до старта (отсчет 3..2..1), чтобы не засорять данные
        if _time < 0:
            return

        try:
            state = iface.get_simulation_state()
        except Exception:
            return

        # --- 1. ОБРАБОТКА ВВОДА (Клавиатура + Геймпад) ---
        steer = state.input_steer
        if steer == 0:
            if state.input_left: steer = -65536
            elif state.input_right: steer = 65536

        gas = state.input_gas
        if gas == 0 and state.input_accelerate:
            gas = 65536

        brake = 1 if state.input_brake else 0

        # --- 2. СБОР ФИЗИКИ ---
        # Глобальная скорость
        vel_global = np.array(state.velocity)
        speed_kmh = np.linalg.norm(vel_global) * 3.6
        
        # Достаем данные машины
        car = state.scene_mobil
        engine = car.engine
        
        # ЛОКАЛЬНАЯ СКОРОСТЬ (Самое важное для ML!)
        # car.current_local_speed возвращает [Right, Up, Forward]
        local_speed = np.array(car.current_local_speed)
        speed_sideward = local_speed[0] # Скорость боком
        speed_forward = local_speed[2]  # Скорость вперед
        
        # Турбо
        turbo = car.turbo_boost_factor
        
        # Контакт колес и материал
        wheels_contact = 0
        surface_id = 0 # 0 = default/asphalt
        for w in state.simulation_wheels:
            rt = w.real_time_state
            if rt.has_ground_contact:
                wheels_contact += 1
                # Берем материал первого колеса, которое коснулось земли
                if surface_id == 0:
                    surface_id = rt.contact_material_id

        y, p, r = state.yaw_pitch_roll

        # --- 3. ЗАПИСЬ ---
        row = [
            _time,
            steer, gas, brake,
            round(speed_kmh, 2),
            round(speed_forward, 3),
            round(speed_sideward, 3),
            round(engine.actual_rpm, 1),
            engine.gear,
            round(turbo, 3),
            1 if car.is_sliding else 0,
            wheels_contact,
            surface_id,
            f"{state.position[0]:.2f}", f"{state.position[1]:.2f}", f"{state.position[2]:.2f}",
            f"{vel_global[0]:.3f}", f"{vel_global[1]:.3f}", f"{vel_global[2]:.3f}",
            f"{y:.3f}", f"{p:.3f}", f"{r:.3f}",
        ]

        self.writer.writerow(row)
        
        # Инфо в консоль
        sys.stdout.write(f"\rTime: {_time/1000:.1f}s | Spd: {int(speed_kmh)} | Slide: {speed_sideward:.1f} | Surface: {surface_id}")
        sys.stdout.flush()

    def on_shutdown(self, iface: TMInterface):
        self.file.close()
        print(f"\nЗапись завершена.")

def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    run_client(DataRecorder(), server_name)

if __name__ == '__main__':
    main()