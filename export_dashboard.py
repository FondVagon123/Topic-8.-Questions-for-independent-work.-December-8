import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# --- КОНФІГУРАЦІЯ ---
st.set_page_config(
    page_title="Аналіз Інноваційної Стратегії Автопрому Японії",
    layout="wide",
    initial_sidebar_state="expanded"
)
FILE_PATH = "final_export_analysis.csv"

# --- 1. ФУНКЦІЯ: ЛІНІЙНИЙ ГРАФІК ТРЕНДУ ІНВЕСТИЦІЙ ---
def plot_investment_trend(df):
    """Відображає динаміку R&D та Інвестицій в Обладнання."""
    
    fig = px.line(
        df,
        x='Year',
        y=['R&D_Trillion_Yen', 'Equipment_Trillion_Yen'],
        title='**1. Динаміка Інвестицій у R&D та Обладнання (2015-2026)**',
        labels={
            'value': 'Вартість (трлн. єн)',
            'variable': 'Тип Інвестиції'
        },
        template="plotly_dark"
    )
    fig.update_xaxes(tick0=2015, dtick=1) # Забезпечуємо відображення кожного року
    fig.update_layout(legend_title_text='Інвестиції')
    st.plotly_chart(fig, use_container_width=True)

# --- 2. ФУНКЦІЯ: ГРАФІК ФОКУСУ НА ІННОВАЦІЯХ (Ключова метрика) ---
def plot_innovation_focus(df):
    """Відображає частку R&D у загальних інвестиціях (Innovation_Focus)."""
    
    fig = px.bar(
        df,
        x='Year',
        y='Innovation_Focus',
        color='Innovation_Focus',
        color_continuous_scale=px.colors.sequential.Plotly3,
        title='**2. Частка R&D у Загальних Інвестиціях (Innovation Focus)**',
        labels={'Innovation_Focus': 'Частка R&D'},
        template="plotly_dark"
    )
    fig.update_yaxes(tickformat=".0%") # Формат у відсотках
    fig.update_xaxes(tick0=2015, dtick=1)
    st.plotly_chart(fig, use_container_width=True)

# --- 3. ФУНКЦІЯ: SCATTER PLOT З ЛІНІЄЮ ТРЕНДУ (РЕГРЕСІЯ) ---
def plot_regression_scatter(df):
    """Відображає зв'язок Фокусу на Інноваціях та Прибутку."""
    
    fig = px.scatter(
        df,
        x='Innovation_Focus',
        y='Industry_Profit',
        text='Year',
        size='R&D_Trillion_Yen', # Відображаємо розмір бульбашки за R&D
        title='**3. Вплив Інноваційного Фокусу на Прибуток (Regression)**',
        labels={
            'Innovation_Focus': 'Фокус на Інноваціях (Частка R&D)',
            'Industry_Profit': 'Прибуток Галузі (трлн. єн)'
        },
        trendline="ols", # Додаємо лінію регресії (OLS)
        template="plotly_dark"
    )
    # Додавання міток років до точок
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)


# --- ОСНОВНА ФУНКЦІЯ ДАШБОРДУ ---
def run_dashboard():
    """Запускає головний дашборд Streamlit."""
    
    if not os.path.exists(FILE_PATH):
        st.error(f"Файл '{FILE_PATH}' не знайдено. Будь ласка, виконайте `python data_generation.py` та `python export_prediction_model.py` послідовно.")
        return

    df = pd.read_csv(FILE_PATH)
    
    # --- Заголовки ---
    st.title("🇯🇵 Аналіз Інноваційної Стратегії Автопрому Японії (2015-2026)")
    st.header("📊 Результати Data Mining та Візуалізація Часового Ряду")
    
    # --- Розділ 1: Тренд Інвестицій ---
    st.subheader("Розділ 1: Динаміка Стратегічних Інвестицій")
    col1, col2 = st.columns(2)
    with col1:
        plot_investment_trend(df)
    with col2:
        plot_innovation_focus(df)

    # --- Розділ 2: Зв'язки та Регресія ---
    st.subheader("Розділ 2: Моделювання Впливу Інновацій на Прибуток")
    col3, col4 = st.columns(2)
    with col3:
        plot_regression_scatter(df)
    with col4:
        # Для простоти, відобразимо прогнозований vs фактичний профіль
        fig_prof = go.Figure()
        fig_prof.add_trace(go.Scatter(x=df['Year'], y=df['Industry_Profit'], mode='lines+markers', name='Фактичний Profit', line=dict(color='yellow')))
        # Перевіряємо, чи існує Prediction_Error з попереднього запуску
        if 'Predicted_Profit' in df.columns:
            fig_prof.add_trace(go.Scatter(x=df['Year'], y=df['Predicted_Profit'], mode='lines', name='Прогнозований Profit', line=dict(dash='dash', color='blue')))
        
        fig_prof.update_layout(
            title='**4. Фактичний vs Прогнозований Прибуток (Регресія)**',
            xaxis_title='Рік',
            yaxis_title='Прибуток (трлн. єн)',
            template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        fig_prof.update_xaxes(tick0=2015, dtick=1)
        st.plotly_chart(fig_prof, use_container_width=True)

    
    # --- Ключові Висновки ---
    st.markdown("---")
    st.subheader("💡 Ключові Висновки з Аналізу Часового Ряду (2015-2026)")
    st.markdown("""
        1. **Тренд:** Спостерігається стійке зростання інвестицій у **R&D** (Інновації) та **Equipment** (Обладнання), що свідчить про довгострокову стратегію зростання.
        2. **Ключова Модель:** 'Innovation Focus' (Частка R&D) демонструє, що японський автопром зосереджується на **якості та технологіях**, а не лише на розширенні потужностей.
        3. **Вплив:** Графік регресії підтверджує **сильний позитивний зв'язок** між фокусом на інноваціях та прибутком галузі.
        4. **Прогноз:** Лінія прогнозу показує, що при збереженні поточних інвестиційних трендів, очікується подальше зростання прибутку до 2026 року.
    """)

if __name__ == "__main__":
    run_dashboard()
