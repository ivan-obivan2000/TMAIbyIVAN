import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def get_latest_file(folder='data_logs'):
    """Находит самый свежий CSV файл в папке"""
    list_of_files = glob.glob(f'{folder}/*.csv')
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def visualize_log():
    # 1. Загрузка данных
    filename = get_latest_file()
    if not filename:
        print("Нет файлов в папке data_logs!")
        return

    print(f"Анализирую файл: {filename}")
    df = pd.read_csv(filename)

    # 2. Обработка рестартов
    # Если время следующего кадра меньше предыдущего, значит был рестарт
    # Создадим колонку 'attempt_id' (номер попытки)
    df['attempt_id'] = (df['time'] < df['time'].shift(1)).cumsum()

    print(f"Найдено попыток (заездов): {df['attempt_id'].max() + 1}")

    # 3. Построение графиков
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # График 1: Скорость
    # Рисуем каждую попытку отдельным цветом
    for attempt in df['attempt_id'].unique():
        subset = df[df['attempt_id'] == attempt]
        # Используем индекс (номер строки) как ось X, чтобы заезды шли друг за другом
        axes[0].plot(subset.index, subset['speed_kmh'], label=f'Попытка {attempt+1}')
    
    axes[0].set_ylabel('Скорость (км/ч)')
    axes[0].set_title('Профиль скорости')
    axes[0].grid(True)
    axes[0].legend()

    # График 2: Руление (Steering)
    axes[1].plot(df.index, df['input_steer'], color='orange', alpha=0.7)
    axes[1].set_ylabel('Руль (-6..6)')
    axes[1].set_title('Действия рулем')
    axes[1].grid(True)
    # Добавим линии пределов руля
    axes[1].axhline(y=65536, color='r', linestyle='--', alpha=0.3)
    axes[1].axhline(y=-65536, color='r', linestyle='--', alpha=0.3)

    # График 3: Газ и Тормоз
    axes[2].plot(df.index, df['input_gas'] * 100, label='Газ (0-100%)', color='green', alpha=0.6)
    axes[2].plot(df.index, df['input_brake'] * 50, label='Тормоз (On/Off)', color='red', alpha=0.6)
    axes[2].set_ylabel('Педали')
    axes[2].set_title('Газ и Тормоз')
    axes[2].legend()
    axes[2].grid(True)

    plt.xlabel('Фреймы (время записи)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_log()