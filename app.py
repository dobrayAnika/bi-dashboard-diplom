from flask import Flask, render_template, request, flash, jsonify
from sqlalchemy import create_engine, inspect
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
import warnings
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

load_dotenv()

# Отключаем предупреждения
warnings.simplefilter(action='ignore', category=UserWarning)

app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- НАСТРОЙКА БАЗЫ ДАННЫХ ---
engine = create_engine(
    f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@"
    f"{os.environ['DB_HOST']}/{os.environ['DB_NAME']}"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, 'mapping_config.json')
METADATA_FILE = os.path.join(BASE_DIR, 'metadata.json')

# --- API-КЛЮЧ GIGACHAT ---
GIGACHAT_KEY = os.environ["GIGACHAT_KEY"]

# --- БИБЛИОТЕКА МЕТРИК ---
METRICS_LIBRARY = {
    "gmv": {
        "title": "GMV / Общий объём продаж",
        "desc": "Совокупная стоимость всех проданных товаров.",
        "req": [
            {"name": "Цена товара", "type": "base"},
            {"name": "Количество", "type": "base"}
        ]
    },
    "order_count": {
        "title": "Количество заказов",
        "desc": "Общее число заказов.",
        "req": [
            {"name": "ID заказа", "type": "base"}
        ]
    },
    "aov": {
        "title": "AOV / Средний чек",
        "desc": "Средняя сумма одного заказа.",
        "req": [
            {"name": "Цена товара", "type": "base"},
            {"name": "Количество", "type": "base"},
            {"name": "ID заказа", "type": "base"}
        ]
    },
    "monthly_revenue": {
        "title": "Выручка по месяцам",
        "desc": "Столбчатая диаграмма выручки по каждому месяцу.",
        "req": [
            {"name": "Цена товара", "type": "base"},
            {"name": "Количество", "type": "base"},
            {"name": "Дата заказа", "type": "base"}
        ]
    },
    "revenue_growth": {
        "title": "Revenue Growth Rate / Темп роста выручки",
        "desc": "Процент изменения выручки между периодами.",
        "req": [
            {"name": "Цена товара", "type": "base"},
            {"name": "Количество", "type": "base"},
            {"name": "Дата заказа", "type": "base"}
        ]
    },
    "arpc": {
        "title": "ARPC / Средний доход на клиента",
        "desc": "Средний доход на одного уникального клиента",
        "req": [
            {"name": "Цена товара", "type": "base"},
            {"name": "Количество", "type": "base"},
            {"name": "Клиент", "type": "base"}
        ]
    },
    "new_customers_monthly": {
        "title": "Прирост клиентов в месяц",
        "desc": "Столбчатая диаграмма новых уникальных клиентов по каждому месяцу.",
        "req": [
            {"name": "Клиент", "type": "base"},
            {"name": "Дата заказа", "type": "base"}
        ]
    },
    "unique_customers": {
        "title": "Уникальные клиенты",
        "desc": "Количество уникальных клиентов",
        "req": [
            {"name": "Клиент", "type": "base"}
        ]
    }
}



def safe_convert(df, col):
    """Безопасное преобразование колонки в числовой тип, NaN заменяются на 0."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce').fillna(0)
    return pd.Series(0, index=df.index)


def load_mapping_config():
    """Загрузка конфигурации маппинга из файла."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def clear_mapping_config():
    """Очистка всей конфигурации маппинга (вызывается при загрузке новой базы данных)."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)


def get_fields(mapping, kpi_name):
    """Получение полей для конкретного KPI из маппинга."""
    data = mapping.get(kpi_name, {})
    return data.get('fields', data)


def find_date_column(df):
    """Поиск колонки с датами в DataFrame."""
    for col in df.columns:
        if 'date' in col.lower() or 'дата' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                if df[col].notna().sum() > 0:
                    return col
            except Exception:
                pass
    return None


def calculate_gmv(df, mapping):
    """Расчёт метрики GMV."""
    fields = get_fields(mapping, 'gmv')
    if not fields:
        return None

    price_col = fields.get('Цена товара')
    qty_col = fields.get('Количество')

    if price_col and qty_col:
        price = safe_convert(df, price_col)
        qty = safe_convert(df, qty_col)
        df['calc_gmv'] = price * qty
        return f"{df['calc_gmv'].sum():,.0f} ₽".replace(',', ' ')
    return "0 ₽"


def calculate_order_count(df, mapping):
    """Расчёт количества заказов."""
    fields = get_fields(mapping, 'order_count')
    if not fields:
        return 0

    order_col = fields.get('ID заказа')
    if order_col and order_col in df.columns:
        return df[order_col].nunique()
    return 0


def calculate_aov(df, mapping):
    """Расчёт среднего чека (AOV)."""
    fields = get_fields(mapping, 'aov')
    if not fields:
        return "Нет данных"

    # Вариант 1: Прямой расчет из полей AOV
    price_col = fields.get('Цена товара')
    qty_col = fields.get('Количество')
    order_col = fields.get('ID заказа')

    if price_col and qty_col and order_col:
        if price_col in df.columns and qty_col in df.columns and order_col in df.columns:
            price = safe_convert(df, price_col)
            qty = safe_convert(df, qty_col)
            total_revenue = (price * qty).sum()
            total_orders = df[order_col].nunique()

            if total_orders > 0:
                return f"{(total_revenue / total_orders):,.0f} ₽".replace(',', ' ')

    # Вариант 2: Через зависимости GMV и Order Count
    if 'calc_gmv' in df.columns and 'order_count' in mapping:
        total_revenue = df['calc_gmv'].sum()
        total_orders = calculate_order_count(df, mapping)
        if total_orders > 0:
            return f"{(total_revenue / total_orders):,.0f} ₽".replace(',', ' ')

    return "Нет данных"

def calculate_unique_customers(df, mapping):
    """Расчёт количества уникальных клиентов."""
    fields = get_fields(mapping, 'unique_customers')
    if not fields:
        return 0

    customer_col = fields.get('Клиент')
    if not customer_col or customer_col not in df.columns:
        return 0

    # Приводим к строке чтобы работало и с числами и с текстом
    customers = df[customer_col].astype(str).str.strip()
    # Убираем пустые и nan
    customers = customers[~customers.isin(['', 'nan', 'None', 'NaN'])]
    return customers.nunique()

def calculate_arpc(df, mapping):
    """Расчёт ARPC (средний доход на клиента) = GMV / Уникальные клиенты."""
    fields = get_fields(mapping, 'arpc')
    if not fields:
        return "Нет данных"

    price_col = fields.get('Цена товара')
    qty_col = fields.get('Количество')
    customer_col = fields.get('Клиент')

    if not all([price_col, qty_col, customer_col]):
        return "Заполните все поля"

    if not all(col in df.columns for col in [price_col, qty_col, customer_col]):
        return "Колонки не найдены в БД"

    # GMV
    price = safe_convert(df, price_col)
    qty = safe_convert(df, qty_col)
    total_gmv = (price * qty).sum()

    # Уникальные клиенты (логика как в unique_customers)
    customers = df[customer_col].astype(str).str.strip()
    customers = customers[~customers.isin(['', 'nan', 'None', 'NaN'])]
    unique_count = customers.nunique()

    if unique_count > 0:
        arpc = total_gmv / unique_count
        return f"{arpc:,.0f} ₽".replace(',', ' ')

    return "0 ₽"


@app.route('/')
def dashboard():
    """Главная страница — дашборд с KPI-метриками."""
    try:
        df = pd.read_sql("SELECT * FROM sales", engine)

        mapping = load_mapping_config()

        metric_results = []

        # Расчёт GMV
        if 'gmv' in mapping:
            gmv_value = calculate_gmv(df, mapping)
            if gmv_value:
                metric_results.append({
                    'id': 'gmv',
                    'title': METRICS_LIBRARY['gmv']['title'],
                    'value': gmv_value,
                    'icon': '💰'
                })

        # Расчёт количества заказов
        if 'order_count' in mapping:
            total_orders = calculate_order_count(df, mapping)
            metric_results.append({
                'id': 'order_count',
                'title': METRICS_LIBRARY['order_count']['title'],
                'value': str(total_orders),
                'icon': '📦'
            })

        # Расчёт AOV
        if 'aov' in mapping:
            aov_value = calculate_aov(df, mapping)
            metric_results.append({
                'id': 'aov',
                'title': METRICS_LIBRARY['aov']['title'],
                'value': aov_value,
                'icon': '🛒'
            })

        # Расчёт уникальных клиентов
        if 'unique_customers' in mapping:
            uc_value = calculate_unique_customers(df, mapping)
            metric_results.append({
                'id': 'unique_customers',
                'title': METRICS_LIBRARY['unique_customers']['title'],
                'value': str(uc_value),
                'icon': '👥'
            })

        # Расчёт ARPC
        if 'arpc' in mapping:
            arpc_value = calculate_arpc(df, mapping)
            metric_results.append({
                'id': 'arpc',
                'title': METRICS_LIBRARY['arpc']['title'],
                'value': arpc_value,
                'icon': '💵'
            })


        date_col = find_date_column(df)

        # ============================================
        # СТОЛБЧАТАЯ ДИАГРАММА ВЫРУЧКИ ПО МЕСЯЦАМ
        # ============================================

        revenue_bar_chart = None
        revenue_table_data = None

        # --- 1. Столбчатая диаграмма (monthly_revenue) ---
        if date_col and 'monthly_revenue' in mapping:
            fields = get_fields(mapping, 'monthly_revenue')
            price_col = fields.get('Цена товара')
            qty_col = fields.get('Количество')

            if price_col and qty_col and price_col in df.columns and qty_col in df.columns:
                df_chart = df.copy()
                df_chart[date_col] = pd.to_datetime(df_chart[date_col], errors='coerce')
                df_chart = df_chart.dropna(subset=[date_col])

                for col_name in [price_col, qty_col]:
                    if df_chart[col_name].dtype == object:
                        df_chart[col_name] = (
                            df_chart[col_name]
                            .astype(str)
                            .str.replace(r'[^\d.]', '', regex=True)
                        )
                    df_chart[col_name] = pd.to_numeric(df_chart[col_name], errors='coerce').fillna(0)

                df_chart['revenue'] = df_chart[price_col] * df_chart[qty_col]
                df_chart['month_label'] = df_chart[date_col].dt.strftime('%Y-%m')

                monthly_revenue = (
                    df_chart
                    .groupby('month_label', as_index=False)['revenue']
                    .sum()
                    .sort_values('month_label')
                )

                if monthly_revenue['revenue'].sum() > 0:
                    months_list = monthly_revenue['month_label'].tolist()
                    revenue_list = [float(v) for v in monthly_revenue['revenue']]
                    text_list = [f"{v:,.0f} ₽".replace(',', ' ') for v in revenue_list]

                    fig_bar = go.Figure(
                        data=[
                            go.Bar(
                                x=months_list,
                                y=revenue_list,
                                text=text_list,
                                textposition='outside',
                                textfont=dict(size=13, color='#1e293b'),
                                marker=dict(
                                    color='#2563eb',
                                    line=dict(color='#1e40af', width=1.5)
                                ),
                                hovertemplate='Месяц: %{x}<br>Выручка: %{text}<extra></extra>'
                            )
                        ]
                    )

                    max_val = max(revenue_list) if revenue_list else 0

                    fig_bar.update_layout(
                        xaxis_title='Месяц',
                        yaxis_title='Выручка (₽)',
                        xaxis=dict(type='category', tickangle=-45, gridcolor='#e2e8f0'),
                        yaxis=dict(gridcolor='#e2e8f0', tickformat=',.0f',
                                   range=[0, max_val * 1.25] if max_val > 0 else [0, 1]),
                        bargap=0.3,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=60, b=100, l=100, r=30),
                        font=dict(family='Inter, sans-serif', size=13),
                        hoverlabel=dict(bgcolor='#1e293b', font_size=14, font_color='white')
                    )

                    revenue_bar_chart = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)

        # --- 2. Таблица роста выручки (revenue_growth) ---
        if date_col and 'revenue_growth' in mapping:
            fields_growth = get_fields(mapping, 'revenue_growth')
            price_col_g = fields_growth.get('Цена товара')
            qty_col_g = fields_growth.get('Количество')

            if price_col_g and qty_col_g and price_col_g in df.columns and qty_col_g in df.columns:
                df_growth = df.copy()
                df_growth[date_col] = pd.to_datetime(df_growth[date_col], errors='coerce')
                df_growth = df_growth.dropna(subset=[date_col])

                for col_name in [price_col_g, qty_col_g]:
                    if df_growth[col_name].dtype == object:
                        df_growth[col_name] = (
                            df_growth[col_name]
                            .astype(str)
                            .str.replace(r'[^\d.]', '', regex=True)
                        )
                    df_growth[col_name] = pd.to_numeric(df_growth[col_name], errors='coerce').fillna(0)

                df_growth['revenue'] = df_growth[price_col_g] * df_growth[qty_col_g]
                df_growth['month_label'] = df_growth[date_col].dt.strftime('%Y-%m')

                monthly_rev_table = (
                    df_growth
                    .groupby('month_label', as_index=False)['revenue']
                    .sum()
                    .sort_values('month_label')
                )

                if len(monthly_rev_table) > 0 and monthly_rev_table['revenue'].sum() > 0:
                    revenue_table_data = []
                    prev_revenue = None

                    for _, row in monthly_rev_table.iterrows():
                        month = row['month_label']
                        rev = float(row['revenue'])
                        rev_formatted = f"{rev:,.0f} ₽".replace(',', ' ')

                        if prev_revenue is not None and prev_revenue > 0:
                            growth = ((rev - prev_revenue) / prev_revenue) * 100
                            if growth > 0:
                                growth_str = f"+{growth:.1f}%"
                                growth_class = "positive"
                            elif growth < 0:
                                growth_str = f"{growth:.1f}%"
                                growth_class = "negative"
                            else:
                                growth_str = "0.0%"
                                growth_class = "neutral"
                        elif prev_revenue is not None and prev_revenue == 0:
                            growth_str = "—"
                            growth_class = "neutral"
                        else:
                            growth_str = "—"
                            growth_class = "neutral"

                        revenue_table_data.append({
                            'month': month,
                            'revenue': rev_formatted,
                            'growth': growth_str,
                            'growth_class': growth_class
                        })

                        prev_revenue = rev


        # ============================================
        # СТОЛБЧАТАЯ ДИАГРАММА НОВЫХ КЛИЕНТОВ ПО МЕСЯЦАМ
        # ============================================
        new_customers_bar_chart = None
        new_customers_table_data = None

        if 'new_customers_monthly' in mapping:
            fields_nc = get_fields(mapping, 'new_customers_monthly')
            customer_col_nc = fields_nc.get('Клиент')
            date_col_nc = fields_nc.get('Дата заказа')

            if customer_col_nc and date_col_nc and customer_col_nc in df.columns and date_col_nc in df.columns:
                df_nc = df.copy()
                df_nc[date_col_nc] = pd.to_datetime(df_nc[date_col_nc], errors='coerce')
                df_nc = df_nc.dropna(subset=[date_col_nc])

                # Чистим столбец клиентов
                df_nc['customer_clean'] = df_nc[customer_col_nc].astype(str).str.strip()
                df_nc = df_nc[~df_nc['customer_clean'].isin(['', 'nan', 'None', 'NaN'])]

                if not df_nc.empty:
                    # Для каждого клиента находим месяц первого заказа
                    df_nc['month_label'] = df_nc[date_col_nc].dt.to_period('M').astype(str)
                    first_month = df_nc.groupby('customer_clean')['month_label'].min().reset_index()
                    first_month.columns = ['customer_clean', 'first_month']

                    # Считаем новых клиентов по месяцам
                    new_per_month = (
                        first_month
                        .groupby('first_month')
                        .size()
                        .reset_index(name='new_customers')
                        .rename(columns={'first_month': 'month_label'})
                        .sort_values('month_label')
                    )

                    if new_per_month['new_customers'].sum() > 0:
                        months_list_nc = new_per_month['month_label'].tolist()
                        customers_list_nc = [int(v) for v in new_per_month['new_customers']]
                        text_list_nc = [str(v) for v in customers_list_nc]

                        fig_nc = go.Figure(
                            data=[
                                go.Bar(
                                    x=months_list_nc,
                                    y=customers_list_nc,
                                    text=text_list_nc,
                                    textposition='outside',
                                    textfont=dict(size=13, color='#1e293b'),
                                    marker=dict(
                                        color='#10b981',
                                        line=dict(color='#059669', width=1.5)
                                    ),
                                    hovertemplate='Месяц: %{x}<br>Новых клиентов: %{text}<extra></extra>'
                                )
                            ]
                        )

                        max_val_nc = max(customers_list_nc) if customers_list_nc else 0
                        fig_nc.update_layout(
                            xaxis_title='Месяц',
                            yaxis_title='Количество новых клиентов',
                            xaxis=dict(type='category', tickangle=-45, gridcolor='#e2e8f0'),
                            yaxis=dict(gridcolor='#e2e8f0', tickformat='d',
                                       range=[0, max_val_nc * 1.25] if max_val_nc > 0 else [0, 1]),
                            bargap=0.3,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(t=60, b=100, l=100, r=30),
                            font=dict(family='Inter, sans-serif', size=13),
                            hoverlabel=dict(bgcolor='#1e293b', font_size=14, font_color='white')
                        )

                        new_customers_bar_chart = json.dumps(fig_nc, cls=plotly.utils.PlotlyJSONEncoder)

                        # Таблица роста новых клиентов
                        new_customers_table_data = []
                        prev_count = None
                        for _, row in new_per_month.iterrows():
                            month = row['month_label']
                            count = int(row['new_customers'])

                            if prev_count is not None and prev_count > 0:
                                growth = ((count - prev_count) / prev_count) * 100
                                if growth > 0:
                                    growth_str = f"+{growth:.1f}%"
                                    growth_class = "positive"
                                elif growth < 0:
                                    growth_str = f"{growth:.1f}%"
                                    growth_class = "negative"
                                else:
                                    growth_str = "0.0%"
                                    growth_class = "neutral"
                            elif prev_count is not None and prev_count == 0:
                                growth_str = "—"
                                growth_class = "neutral"
                            else:
                                growth_str = "—"
                                growth_class = "neutral"

                            new_customers_table_data.append({
                                'month': month,
                                'count': str(count),
                                'growth': growth_str,
                                'growth_class': growth_class
                            })
                            prev_count = count


        return render_template(
            'index.html',
            metric_results=metric_results,
            revenue_bar_chart=revenue_bar_chart,
            revenue_table_data=revenue_table_data,
            new_customers_bar_chart=new_customers_bar_chart,
            new_customers_table_data=new_customers_table_data,
            no_data=False
        )

    except Exception as e:
        logging.error(f"Ошибка загрузки дашборда: {e}")
        return render_template(
            'index.html',
            metric_results=[],
            revenue_bar_chart=None,
            revenue_table_data=None,
            new_customers_bar_chart=None,
            new_customers_table_data=None,
            no_data=True
        )

@app.route('/bi_tool', methods=['GET', 'POST'])
def bi_tool():
    """Страница BI-конструктора — настройка маппинга полей и метрик."""
    columns = []
    db_name = None
    saved_mapping = load_mapping_config()

    # Проверяем, загружены ли данные в базу
    try:
        inspector = inspect(engine)
        if 'sales' in inspector.get_table_names():
            df_sample = pd.read_sql("SELECT * FROM sales LIMIT 1", engine)
            columns = df_sample.columns.tolist()

            if os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    db_name = meta.get('filename')
            else:
                db_name = "Текущая база данных"
    except Exception as e:
        logging.error(f"Ошибка проверки базы: {e}")

    # Обработка загрузки файла
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            fname_lower = file.filename.lower()
            if not (fname_lower.endswith('.csv') or fname_lower.endswith('.xlsx') or fname_lower.endswith('.xls')):
                flash('Неподдерживаемый формат файла. Загрузите файл CSV или Excel (.xlsx, .xls).', 'error')
            else:
                try:
                    if fname_lower.endswith('.csv'):
                        try:
                            df = pd.read_csv(file, encoding='utf-8')
                        except UnicodeDecodeError:
                            file.seek(0)
                            df = pd.read_csv(file, encoding='cp1251')
                    else:
                        df = pd.read_excel(file)

                    if df.empty:
                        flash('Файл не содержит данных. Проверьте содержимое файла.', 'error')
                    else:
                        empty_cols = [col for col in df.columns if df[col].isna().all()]
                        if empty_cols:
                            flash(
                                f'Внимание: следующие колонки полностью пусты — '
                                f'{", ".join(str(c) for c in empty_cols)}. '
                                f'Метрики, использующие их, будут равны 0.',
                                'warning'
                            )

                        df.to_sql('sales', engine, if_exists='replace', index=False)
                        columns = df.columns.tolist()
                        db_name = file.filename

                        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                            json.dump({'filename': db_name}, f, ensure_ascii=False)

                        clear_mapping_config()
                        saved_mapping = {}

                        flash(f'База данных "{db_name}" загружена. Настройте метрики заново.', 'success')

                except Exception as e:
                    logging.error(f"Ошибка загрузки файла: {e}")
                    flash(
                        'Не удалось прочитать файл. Убедитесь, что файл не повреждён '
                        'и соответствует формату CSV или Excel.',
                        'error'
                    )

    return render_template(
        'bi_tool.html',
        columns=columns,
        metrics_library=METRICS_LIBRARY,
        db_name=db_name,
        saved_mapping=saved_mapping
    )


@app.route('/save_config', methods=['POST'])
def save_config():
    """Сохранение конфигурации KPI в файл маппинга."""
    data = request.json
    config = load_mapping_config()

    kpi_name = data.get('kpi_name')
    if kpi_name:
        config[kpi_name] = {
            'fields': data.get('fields', {})
        }

        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        return jsonify({'status': 'success'})

    return jsonify({'status': 'error', 'message': 'KPI name missing'}), 400


@app.route('/clear_field', methods=['POST'])
def clear_field():
    """Удаление маппинга для конкретного поля KPI."""
    data = request.json
    config = load_mapping_config()

    kpi_name = data.get('kpi_name')
    field_name = data.get('field_name')

    if kpi_name and kpi_name in config:
        if 'fields' in config[kpi_name]:
            if field_name in config[kpi_name]['fields']:
                del config[kpi_name]['fields'][field_name]

                if len(config[kpi_name]['fields']) == 0:
                    del config[kpi_name]

        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        return jsonify({'status': 'success', 'message': f'Поле "{field_name}" очищено'})

    return jsonify({'status': 'error', 'message': 'Неверная метрика или поле'}), 400


@app.route('/calculate_metric', methods=['POST'])
def calculate_metric():
    """Расчёт и возврат значения конкретной метрики."""
    data = request.json
    kpi_id = data.get('kpi_id')
    fields = data.get('fields', {})

    if not kpi_id:
        return jsonify({'status': 'error', 'message': 'Не указана метрика'}), 400

    try:
        df = pd.read_sql("SELECT * FROM sales", engine)

        value = "Нет данных"
        title = METRICS_LIBRARY.get(kpi_id, {}).get('title', 'Метрика')

        if kpi_id == 'gmv':
            if 'Цена товара' not in fields or 'Количество' not in fields:
                return jsonify({
                    'status': 'error',
                    'message': 'Заполните все поля (Цена товара и Количество)'
                }), 400

            price_col = fields.get('Цена товара')
            qty_col = fields.get('Количество')

            if price_col not in df.columns or qty_col not in df.columns:
                return jsonify({
                    'status': 'error',
                    'message': 'Колонки не найдены в базе данных'
                }), 400

            price = safe_convert(df, price_col)
            qty = safe_convert(df, qty_col)
            gmv = (price * qty).sum()
            value = f"{gmv:,.0f} ₽".replace(',', ' ')

        elif kpi_id == 'order_count':
            if 'ID заказа' not in fields:
                return jsonify({
                    'status': 'error',
                    'message': 'Заполните поле (ID заказа)'
                }), 400

            order_col = fields.get('ID заказа')

            if order_col not in df.columns:
                return jsonify({
                    'status': 'error',
                    'message': f'Колонка "{order_col}" не найдена в базе данных'
                }), 400

            order_count = df[order_col].nunique()
            value = f"{order_count}"

        elif kpi_id == 'aov':
            main_mapping = load_mapping_config()

            if 'gmv' not in main_mapping or 'order_count' not in main_mapping:
                return jsonify({
                    'status': 'error',
                    'message': 'Для AOV сначала настройте GMV и Количество заказов'
                }), 400

            gmv_fields = main_mapping.get('gmv', {}).get('fields', {})
            price_col = gmv_fields.get('Цена товара')
            qty_col = gmv_fields.get('Количество')

            if price_col and qty_col and price_col in df.columns and qty_col in df.columns:
                price = safe_convert(df, price_col)
                qty = safe_convert(df, qty_col)
                total_revenue = (price * qty).sum()
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Настройте корректно поля для GMV'
                }), 400

            order_fields = main_mapping.get('order_count', {}).get('fields', {})
            order_col = order_fields.get('ID заказа')

            if order_col and order_col in df.columns:
                total_orders = df[order_col].nunique()
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Настройте корректно поля для Количество заказов'
                }), 400

            if total_orders > 0:
                aov = total_revenue / total_orders
                value = f"{aov:,.0f} ₽".replace(',', ' ')
            else:
                value = "0 ₽"

        elif kpi_id == 'arpc':
            if 'Цена товара' not in fields or 'Количество' not in fields or 'Клиент' not in fields:
                return jsonify({
                    'status': 'error',
                    'message': 'Заполните все поля (Цена товара, Количество и Клиент)'
                }), 400

            price_col = fields.get('Цена товара')
            qty_col = fields.get('Количество')
            customer_col = fields.get('Клиент')

            if not all(col in df.columns for col in [price_col, qty_col, customer_col]):
                return jsonify({
                    'status': 'error',
                    'message': 'Колонки не найдены в базе данных'
                }), 400

            price = safe_convert(df, price_col)
            qty = safe_convert(df, qty_col)
            total_gmv = (price * qty).sum()

            customers = df[customer_col].astype(str).str.strip()
            customers = customers[~customers.isin(['', 'nan', 'None', 'NaN'])]
            unique_count = customers.nunique()

            if unique_count > 0:
                arpc = total_gmv / unique_count
                value = f"{arpc:,.0f} ₽".replace(',', ' ')
            else:
                value = "0 ₽"

        return jsonify({
            'status': 'success',
            'value': value,
            'title': title
        })

    except Exception as e:
        logging.error(f"Ошибка расчёта метрики: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Не удалось вычислить метрику. Проверьте, что данные загружены корректно.'
        }), 500


@app.route('/get_metric_result', methods=['POST'])
def get_metric_result():
    """Получение значения метрики для отображения на дашборде."""
    data = request.json
    kpi_id = data.get('kpi_id')

    if not kpi_id:
        return jsonify({'status': 'error', 'message': 'Не указана метрика'}), 400

    try:
        df = pd.read_sql("SELECT * FROM sales", engine)

        mapping = load_mapping_config()
        title = METRICS_LIBRARY.get(kpi_id, {}).get('title', 'Метрика')
        value = "Нет данных"

        if kpi_id == 'gmv' and 'gmv' in mapping:
            result = calculate_gmv(df, mapping)
            value = result if result else "0 ₽"

        elif kpi_id == 'order_count' and 'order_count' in mapping:
            result = calculate_order_count(df, mapping)
            value = f"{result}"

        elif kpi_id == 'aov' and 'aov' in mapping:
            result = calculate_aov(df, mapping)
            value = result if result else "0 ₽"

        elif kpi_id == 'arpc' and 'arpc' in mapping:
            result = calculate_arpc(df, mapping)
            value = result if result else "0 ₽"

        elif kpi_id == 'unique_customers' and 'unique_customers' in mapping:
            result = calculate_unique_customers(df, mapping)
            value = str(result)

        return jsonify({
            'status': 'success',
            'value': value,
            'title': title
        })

    except Exception as e:
        logging.error(f"Ошибка получения метрики: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Не удалось получить значение метрики. Проверьте, что данные загружены.'
        }), 500


# --- AI ANALYSIS ENDPOINT (GigaChat) ---
@app.route('/analyze_ai', methods=['POST'])
def analyze_ai():
    """AI анализ — серверная логика (как в рабочей версии)."""
    try:
        df = pd.read_sql("SELECT * FROM sales", engine)

        mapping = load_mapping_config()

        # Собираем только настроенные метрики
        lines = []

        if 'gmv' in mapping:
            val = calculate_gmv(df, mapping)
            if val: lines.append(f"- GMV: {val}")

        if 'order_count' in mapping:
            val = calculate_order_count(df, mapping)
            lines.append(f"- Заказы: {val}")

        if 'aov' in mapping:
            val = calculate_aov(df, mapping)
            if val != "Нет данных": lines.append(f"- Средний чек: {val}")

        if 'arpc' in mapping:
            val = calculate_arpc(df, mapping)
            if val != "Нет данных": lines.append(f"- ARPC: {val}")

        if 'unique_customers' in mapping:
            val = calculate_unique_customers(df, mapping)
            lines.append(f"- Уникальные клиенты: {val}")

        # Период данных
        date_col = find_date_column(df)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df_sorted = df.dropna(subset=[date_col]).sort_values(by=date_col)
            if len(df_sorted) >= 2:
                first = df_sorted.iloc[0][date_col].strftime('%d.%m.%Y')
                last = df_sorted.iloc[-1][date_col].strftime('%d.%m.%Y')
                lines.append(f"- Период: {first} — {last}")

        if not lines:
            return jsonify({
                'status': 'success',
                'answer': 'На дашборде нет настроенных метрик. Перейдите в BI-Конструктор и настройте хотя бы одну метрику.'
            })

        prompt = (
                f"Ты опытный бизнес-аналитик. Проанализируй метрики:\n\n"
                f"Данные:\n" + "\n".join(lines) + "\n\n"
                                                    f"Задание:\n"
                                                    f"1. Оцени состояние бизнеса (1-2 предложения)\n"
                                                    f"2. Дай 2-3 конкретные рекомендации\n"
                                                    f"3. На что обратить внимание в первую очередь\n"
                                                    f"Пиши кратко, по делу (макс. 150 слов)."
        )

        with GigaChat(credentials=GIGACHAT_KEY, verify_ssl_certs=False) as giga:
            payload = Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM,
                             content="Ты бизнес-аналитик. Анализируй только данные из запроса."),
                    Messages(role=MessagesRole.USER, content=prompt)
                ],
                temperature=0.7,
                max_tokens=1500
            )
            response = giga.chat(payload)
            ai_answer = response.choices[0].message.content

        return jsonify({'status': 'success', 'answer': ai_answer})

    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ['connection', 'timeout', 'network', 'refused', 'unreachable', 'ssl']):
            message = 'Сервис GigaChat временно недоступен. Проверьте подключение к интернету и попробуйте снова.'
        elif any(k in err for k in ['401', 'unauthorized', 'forbidden', '403']):
            message = 'Ошибка авторизации GigaChat. Проверьте правильность API-ключа в файле .env.'
        elif 'sales' in err or 'relation' in err:
            message = 'Нет загруженных данных. Перейдите в Конструктор и загрузите файл.'
        else:
            message = 'Не удалось получить ответ от GigaChat. Попробуйте позже.'
        return jsonify({'status': 'error', 'answer': message}), 500


if __name__ == '__main__':
    app.run(debug=False)
