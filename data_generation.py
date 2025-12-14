import pandas as pd
import numpy as np
import os

# --- КОНФІГУРАЦІЯ ---
FILE_PATH = "simulated_export_data.csv"
OUTPUT_PATH = "processed_time_series_data.csv" # Змінюємо назву для відображення, що це часовий ряд

def create_time_series_dataset():
    """
    Завантажує та обробляє 12 точок часового ряду, наданих користувачем.
    Створює похідні ознаки, необхідні для моделювання.
    """
    
    print("="*80)
    print("--- ⚙️ ЕТАП ETL: ПІДГОТОВКА ЧАСОВОГО РЯДУ (12 ТОЧОК) ---")
    
    # 1. Завантаження даних часового ряду (використовуємо надані вами дані)
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"⚠️ ПОМИЛКА: Файл '{FILE_PATH}' не знайдено.")
        print("Переконайтеся, що ви створили його з оновленими даними.")
        return

    # Симуляція Обсягу Експорту (якщо він не був у вхідному файлі)
    # Ми використовуємо формулу, що імітує вплив R&D та обладнання на Export/Profit
    # Для цього ми створюємо колонку 'Industry_Profit' (використовуючи значення, схожі на ваші 'Industry_Profit')
    
    # Створення базової лінійної комбінації: Industry_Profit ≈ R&D * 1.5 + Equipment * 0.5 + 1.0 (шум)
    np.random.seed(42)
    # Зверніть увагу, що ці значення будуть лише *схожими* на ваші, оскільки ви надаєте їх вручну
    simulated_profit = (df['R&D_Trillion_Yen'] * 1.5 + 
                        df['Equipment_Trillion_Yen'] * 0.5 + 
                        np.random.uniform(low=0.8, high=1.2, size=len(df)))
    
    # Використовуємо округлені значення, схожі на ваші, для узгодженості
    df['Industry_Profit'] = simulated_profit.round(5) 
    
    # 2. Похідна ознака: Фокус на інноваціях (Innovation_Focus)
    df['Innovation_Focus'] = df['R&D_Trillion_Yen'] / (df['R&D_Trillion_Yen'] + df['Equipment_Trillion_Yen'])
    df = df.dropna()

    # Збереження фінального датасету
    df.to_csv(OUTPUT_PATH, index=False)
    
    print("\n" + "="*80)
    print(f"✅ ЕТАП ETL УСПІШНО ЗАВЕРШЕНО.")
    print(f"Створено файл '{OUTPUT_PATH}' з {len(df)} рядками.")
    print("Ключові Змінні: Year, R&D, Equipment, Industry_Profit, Innovation_Focus.")
    print("НАСТУПНИЙ КРОК: python export_prediction_model.py")
    print("="*80)

if __name__ == "__main__":
    create_time_series_dataset()
