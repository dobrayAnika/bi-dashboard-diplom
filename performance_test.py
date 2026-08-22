"""
Тест производительности BI-инструмента.
Измеряет время выполнения для каждого этапа работы системы.
"""
import time
import io
import json
import os
import sys
import warnings

# Принудительно ставим UTF-8 для stdout/stderr на Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import plotly
import plotly.graph_objects as go
from sqlalchemy import create_engine

# Импортируем функции из приложения
from app import (
    app, engine, METRICS_LIBRARY,
    calculate_gmv, calculate_order_count, calculate_aov,
    calculate_unique_customers, calculate_arpc,
    find_date_column, load_mapping_config
)

# ============================================================
# КОНФИГУРАЦИЯ МАППИНГА ДЛЯ КАЖДОГО ФАЙЛА
# ============================================================
FILE_CONFIGS = {
    'SuperStoreOrders1.csv': {
        'display_name': 'SuperStoreOrders1.csv',
        'reader': lambda f: pd.read_csv(f),
        'mapping': {
            'gmv':                  {'fields': {'Цена товара': 'sales',         'Количество': 'quantity'}},
            'order_count':          {'fields': {'ID заказа': 'order_id'}},
            'aov':                  {'fields': {'Цена товара': 'sales',         'Количество': 'quantity', 'ID заказа': 'order_id'}},
            'monthly_revenue':      {'fields': {'Цена товара': 'sales',         'Количество': 'quantity', 'Дата заказа': 'order_date'}},
            'revenue_growth':       {'fields': {'Цена товара': 'sales',         'Количество': 'quantity', 'Дата заказа': 'order_date'}},
            'arpc':                 {'fields': {'Цена товара': 'sales',         'Количество': 'quantity', 'Клиент': 'customer_name'}},
            'new_customers_monthly':{'fields': {'Клиент': 'customer_name',      'Дата заказа': 'order_date'}},
            'unique_customers':     {'fields': {'Клиент': 'customer_name'}},
        }
    },
    'data-1774638704531.csv': {
        'display_name': 'data-1774638704531.csv',
        'reader': lambda f: pd.read_csv(f),
        'mapping': {
            'gmv':                  {'fields': {'Цена товара': 'product_price', 'Количество': 'quantity'}},
            'order_count':          {'fields': {'ID заказа': 'order_id'}},
            'aov':                  {'fields': {'Цена товара': 'product_price', 'Количество': 'quantity', 'ID заказа': 'order_id'}},
            'monthly_revenue':      {'fields': {'Цена товара': 'product_price', 'Количество': 'quantity', 'Дата заказа': 'order_date'}},
            'revenue_growth':       {'fields': {'Цена товара': 'product_price', 'Количество': 'quantity', 'Дата заказа': 'order_date'}},
            'arpc':                 {'fields': {'Цена товара': 'product_price', 'Количество': 'quantity', 'Клиент': 'customer_id'}},
            'new_customers_monthly':{'fields': {'Клиент': 'customer_id',        'Дата заказа': 'order_date'}},
            'unique_customers':     {'fields': {'Клиент': 'customer_id'}},
        }
    },
    'train.csv': {
        'display_name': 'train.csv',
        'reader': lambda f: _read_train(f),
        'mapping': {
            'gmv':                  {'fields': {'Цена товара': 'Sales',         'Количество': 'Quantity'}},
            'order_count':          {'fields': {'ID заказа': 'Order ID'}},
            'aov':                  {'fields': {'Цена товара': 'Sales',         'Количество': 'Quantity', 'ID заказа': 'Order ID'}},
            'monthly_revenue':      {'fields': {'Цена товара': 'Sales',         'Количество': 'Quantity', 'Дата заказа': 'Order Date'}},
            'revenue_growth':       {'fields': {'Цена товара': 'Sales',         'Количество': 'Quantity', 'Дата заказа': 'Order Date'}},
            'arpc':                 {'fields': {'Цена товара': 'Sales',         'Количество': 'Quantity', 'Клиент': 'Customer Name'}},
            'new_customers_monthly':{'fields': {'Клиент': 'Customer Name',      'Дата заказа': 'Order Date'}},
            'unique_customers':     {'fields': {'Клиент': 'Customer Name'}},
        }
    },
}


def _read_train(filepath):
    """Читает train.csv и добавляет синтетическую колонку Quantity=1."""
    df = pd.read_csv(filepath)
    df['Quantity'] = 1
    return df


def fmt(seconds):
    """Форматирует время в удобочитаемый вид."""
    if seconds < 1:
        return f"{seconds * 1000:.1f} мс"
    return f"{seconds:.3f} с"


def separator(char='=', width=70):
    print(char * width)


def section(title):
    separator()
    print(f"  {title}")
    separator()


# ============================================================
# ТЕСТ 1: ЗАГРУЗКА ФАЙЛА И СОХРАНЕНИЕ В POSTGRESQL
# ============================================================
def test_upload(filepath, reader):
    """Читает файл и сохраняет в таблицу sales, возвращает (df, время_сек)."""
    t0 = time.perf_counter()
    df = reader(filepath)
    df.to_sql('sales', engine, if_exists='replace', index=False)
    elapsed = time.perf_counter() - t0
    return df, elapsed


# ============================================================
# ТЕСТ 2: РАСЧЁТ ВСЕХ НАСТРОЕННЫХ МЕТРИК
# ============================================================
def test_metrics(df, mapping):
    """Рассчитывает все метрики, возвращает (results_dict, время_сек)."""
    t0 = time.perf_counter()
    results = {}

    if 'gmv' in mapping:
        results['GMV'] = calculate_gmv(df, mapping)

    if 'order_count' in mapping:
        results['Кол-во заказов'] = calculate_order_count(df, mapping)

    if 'aov' in mapping:
        results['AOV'] = calculate_aov(df, mapping)

    if 'unique_customers' in mapping:
        results['Уникальные клиенты'] = calculate_unique_customers(df, mapping)

    if 'arpc' in mapping:
        results['ARPC'] = calculate_arpc(df, mapping)

    elapsed = time.perf_counter() - t0
    return results, elapsed


# ============================================================
# ТЕСТ 3: ПОСТРОЕНИЕ ГРАФИКОВ PLOTLY
# ============================================================
def test_charts(df, mapping):
    """Строит все доступные графики, возвращает (кол-во_графиков, время_сек)."""
    t0 = time.perf_counter()
    charts_built = 0
    date_col = find_date_column(df)

    # Столбчатая диаграмма выручки по месяцам
    if date_col and 'monthly_revenue' in mapping:
        fields = mapping['monthly_revenue'].get('fields', {})
        price_col = fields.get('Цена товара')
        qty_col = fields.get('Количество')

        if price_col and qty_col and price_col in df.columns and qty_col in df.columns:
            df_chart = df.copy()
            df_chart[date_col] = pd.to_datetime(df_chart[date_col], errors='coerce')
            df_chart = df_chart.dropna(subset=[date_col])

            for col_name in [price_col, qty_col]:
                df_chart[col_name] = pd.to_numeric(
                    df_chart[col_name].astype(str).str.replace(r'[^\d.]', '', regex=True),
                    errors='coerce'
                ).fillna(0)

            df_chart['revenue'] = df_chart[price_col] * df_chart[qty_col]
            df_chart['month_label'] = df_chart[date_col].dt.strftime('%Y-%m')

            monthly = df_chart.groupby('month_label', as_index=False)['revenue'].sum().sort_values('month_label')

            if monthly['revenue'].sum() > 0:
                months_list = monthly['month_label'].tolist()
                revenue_list = [float(v) for v in monthly['revenue']]
                text_list = [f"{v:,.0f} ₽".replace(',', ' ') for v in revenue_list]

                fig = go.Figure(data=[go.Bar(
                    x=months_list, y=revenue_list,
                    text=text_list, textposition='outside',
                    marker=dict(color='#2563eb')
                )])
                fig.update_layout(
                    xaxis_title='Месяц', yaxis_title='Выручка (₽)',
                    plot_bgcolor='white', paper_bgcolor='white'
                )
                _ = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                charts_built += 1

    # Диаграмма новых клиентов по месяцам
    if 'new_customers_monthly' in mapping:
        fields_nc = mapping['new_customers_monthly'].get('fields', {})
        customer_col = fields_nc.get('Клиент')
        date_col_nc = fields_nc.get('Дата заказа')

        if customer_col and date_col_nc and customer_col in df.columns and date_col_nc in df.columns:
            df_nc = df.copy()
            df_nc[date_col_nc] = pd.to_datetime(df_nc[date_col_nc], errors='coerce')
            df_nc = df_nc.dropna(subset=[date_col_nc])
            df_nc['customer_clean'] = df_nc[customer_col].astype(str).str.strip()
            df_nc = df_nc[~df_nc['customer_clean'].isin(['', 'nan', 'None', 'NaN'])]

            if not df_nc.empty:
                df_nc['month_label'] = df_nc[date_col_nc].dt.to_period('M').astype(str)
                first_month = df_nc.groupby('customer_clean')['month_label'].min().reset_index()
                new_per_month = (
                    first_month.groupby('month_label').size()
                    .reset_index(name='new_customers')
                    .sort_values('month_label')
                )

                if new_per_month['new_customers'].sum() > 0:
                    fig_nc = go.Figure(data=[go.Bar(
                        x=new_per_month['month_label'].tolist(),
                        y=[int(v) for v in new_per_month['new_customers']],
                        marker=dict(color='#10b981')
                    )])
                    fig_nc.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                    _ = json.dumps(fig_nc, cls=plotly.utils.PlotlyJSONEncoder)
                    charts_built += 1

    elapsed = time.perf_counter() - t0
    return charts_built, elapsed


# ============================================================
# ТЕСТ 4: ПОЛНАЯ ЗАГРУЗКА ДАШБОРДА (GET /)
# ============================================================
def test_dashboard_load(mapping):
    """Выполняет GET / через тестовый клиент Flask, возвращает (статус, время_сек)."""
    # Временно записываем маппинг
    with open('mapping_config.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)

    client = app.test_client()
    t0 = time.perf_counter()
    response = client.get('/')
    elapsed = time.perf_counter() - t0

    return response.status_code, elapsed


# ============================================================
# ТЕСТ 5: ЗАПРОС К GIGACHAT API
# ============================================================
def test_gigachat():
    """Выполняет POST /analyze_ai через тестовый клиент Flask, возвращает (статус, время_сек)."""
    client = app.test_client()
    t0 = time.perf_counter()
    response = client.post('/analyze_ai', content_type='application/json')
    elapsed = time.perf_counter() - t0

    try:
        body = json.loads(response.data)
        status = body.get('status', 'unknown')
    except Exception:
        status = f"HTTP {response.status_code}"

    return status, elapsed


# ============================================================
# ГЛАВНЫЙ БЛОК
# ============================================================
def main():
    app.config['TESTING'] = True

    all_results = {}

    for filename, config in FILE_CONFIGS.items():
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if not os.path.exists(filepath):
            print(f"\n[ПРОПУСК] Файл не найден: {filename}")
            continue

        display = config['display_name']
        mapping = config['mapping']
        reader = config['reader']

        section(f"ФАЙЛ: {display}")

        # --- 1. Загрузка файла и сохранение в PostgreSQL ---
        print(f"\n[1] Загрузка файла и сохранение в PostgreSQL...")
        df, t_upload = test_upload(filepath, reader)
        rows = len(df)
        cols = len(df.columns)
        print(f"    Строк: {rows:,}  |  Столбцов: {cols}  |  Время: {fmt(t_upload)}")

        # --- 2. Расчёт всех настроенных метрик ---
        print(f"\n[2] Расчёт всех настроенных метрик...")
        metrics_results, t_metrics = test_metrics(df, mapping)
        for name, val in metrics_results.items():
            print(f"    {name}: {val}")
        print(f"    Время расчёта метрик: {fmt(t_metrics)}")

        # --- 3. Построение графиков Plotly ---
        print(f"\n[3] Построение графиков Plotly...")
        charts_count, t_charts = test_charts(df, mapping)
        print(f"    Построено графиков: {charts_count}  |  Время: {fmt(t_charts)}")

        # --- 4. Полная загрузка дашборда (GET /) ---
        print(f"\n[4] Полная загрузка дашборда (GET /)...")
        http_status, t_dashboard = test_dashboard_load(mapping)
        print(f"    HTTP статус: {http_status}  |  Время: {fmt(t_dashboard)}")

        # --- 5. Запрос к GigaChat API ---
        print(f"\n[5] Запрос к GigaChat API...")
        gc_status, t_gigachat = test_gigachat()
        print(f"    Статус ответа: {gc_status}  |  Время: {fmt(t_gigachat)}")

        all_results[display] = {
            'rows': rows,
            'cols': cols,
            't_upload': t_upload,
            't_metrics': t_metrics,
            't_charts': t_charts,
            't_dashboard': t_dashboard,
            't_gigachat': t_gigachat,
            'gc_status': gc_status,
        }

    # ============================================================
    # ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
    # ============================================================
    separator()
    print("\n  ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    separator()

    header = f"{'Файл':<30} {'Строк':>7} {'Загрузка':>12} {'Метрики':>12} {'Графики':>12} {'Дашборд':>12} {'GigaChat':>12}"
    print(header)
    print('-' * 90)

    for fname, r in all_results.items():
        name_short = fname[:28] if len(fname) > 28 else fname
        gc_note = '' if r['gc_status'] == 'success' else f" ({r['gc_status']})"
        row = (
            f"{name_short:<30} "
            f"{r['rows']:>7,} "
            f"{fmt(r['t_upload']):>12} "
            f"{fmt(r['t_metrics']):>12} "
            f"{fmt(r['t_charts']):>12} "
            f"{fmt(r['t_dashboard']):>12} "
            f"{fmt(r['t_gigachat']):>12}{gc_note}"
        )
        print(row)

    print()
    separator()
    print("\nТестирование завершено.")


if __name__ == '__main__':
    main()
