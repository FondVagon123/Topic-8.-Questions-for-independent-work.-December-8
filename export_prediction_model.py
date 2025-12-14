import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# --- КОНФІГУРАЦІЯ ---
FILE_PATH = "processed_time_series_data.csv" # Вхідний файл з даними
OUTPUT_FILE = "final_export_analysis.csv" # Вихідний файл з результатами аналізу

# --- ФУНКЦІЯ: РЕГРЕСІЙНИЙ АНАЛІЗ (ПРОГНОЗУВАННЯ) ---

def run_regression_analysis(df: pd.DataFrame):
    """
    Виконує лінійну регресію для прогнозування Industry_Profit
    на основі Фокусу на Інноваціях та Інвестицій в Обладнання.
    """

    print("\n" + "="*80)
    print("--- 🔬 ЕТАП DATA MINING: ЛІНІЙНА РЕГРЕСІЯ (ПРОГНОЗУВАННЯ) ---")
    print("="*80)

    # Залежна змінна (y): Industry_Profit (Ваш показник успіху)
    # Незалежні змінні (X): Innovation_Focus та Equipment_Trillion_Yen
    X = df[['Innovation_Focus', 'Equipment_Trillion_Yen']]
    y = df['Industry_Profit']

    # Зважаючи на малий розмір даних (12 точок), train_test_split не завжди доречний.
    # Проте, ми його залишимо для демонстрації методології.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = model.score(X_test, y_test)

    print(f"✅ Модель прогнозування Industry Profit (Тестовий набір):")
    print(f"   R² на тестовому наборі: {r2:.4f}")
    print(f"   Середньоквадратична похибка (MSE): {mse:.2f}")
    print("\n--- КОЕФІЦІЄНТИ МОДЕЛІ (Вплив на Profit) ---")
    print(f"   Коефіцієнт Innovation_Focus: {model.coef_[0]:.4f}")
    print(f"   Коефіцієнт Equipment_Trillion_Yen: {model.coef_[1]:.4f}")
    print("------------------------------------------------")

    # Розраховуємо прогнози та залишки для всього датасету для візуалізації
    y_full_pred = model.predict(X)
    df['Regression_Residuals'] = y - y_full_pred
    df['Predicted_Profit'] = y_full_pred # Додаємо прогнозований профіль для візуалізації

    return df

# --- ФУНКЦІЯ: АНАЛІЗ РОЗПОДІЛУ ТА КОРЕЛЯЦІЇ ---

def run_descriptive_analysis(df: pd.DataFrame):
    """Виводить описові статистики та кореляційну матрицю."""

    print("\n" + "="*80)
    print("--- 📊 ЕТАП DATA MINING: ОПИСОВИЙ АНАЛІЗ ТА КОРЕЛЯЦІЯ ---")
    print("="*80)

    print("\nДинаміка R&D, Equipment та Industry Profit (2015-2026):")
    print(df[['Year', 'R&D_Trillion_Yen', 'Equipment_Trillion_Yen', 'Industry_Profit']].tail(5).to_markdown(index=False))

    print("\nКореляційна матриця ключових числових змінних:")
    corr = df[['R&D_Trillion_Yen', 'Equipment_Trillion_Yen', 'Industry_Profit', 'Innovation_Focus']].corr()
    print(corr.round(3).to_markdown())
    print("------------------------------------------------")

    return df

# --- ОСНОВНИЙ БЛОК ВИКОНАННЯ ---

if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        print("⚠️ ПОМИЛКА: Файл 'processed_time_series_data.csv' не знайдено.")
        print("Будь ласка, запустіть спочатку 'python data_generation.py'.")
    else:
        df_data = pd.read_csv(FILE_PATH)

        df_analysis = run_regression_analysis(df_data.copy())
        df_final = run_descriptive_analysis(df_analysis.copy())

        # Зберігаємо фінальний датасет
        # Для узгодженості з вашим старим файлом, перейменуємо колонки
        df_final['Cluster'] = 1 # Додаємо кластер '1', оскільки у вас немає кластеризації
        df_final = df_final[['Year', 'R&D_Trillion_Yen', 'Equipment_Trillion_Yen', 'Industry_Profit', 
                             'Innovation_Focus', 'Regression_Residuals', 'Cluster']]
        
        df_final.to_csv(OUTPUT_FILE, index=False)

        print("\n" + "="*80)
        print("✅ АНАЛІЗ УСПІШНО ЗАВЕРШЕНО.")
        print(f"Файл '{OUTPUT_FILE}' готовий для візуалізації.")
        print("НАСТУПНИЙ КРОК: streamlit run export_dashboard.py")
        print("="*80)
