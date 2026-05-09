import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from datetime import datetime


def fig_to_base64(fig) -> str:
    """Конвертирует matplotlib фигуру в base64 строку для HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130,
                bbox_inches='tight', facecolor='#0f172a')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def build_html_report(
    df: pd.DataFrame,
    final_report: str,
    figures: list,
    filename: str = "dataset",
    user_instruction: str = ""
) -> str:
    """Генерирует полный HTML-отчёт."""

    now        = datetime.now().strftime('%d.%m.%Y %H:%M')
    rows       = len(df)
    cols       = len(df.columns)
    nulls      = df.isnull().sum().sum()
    null_pct   = round(nulls / (rows * cols) * 100, 1)
    num_cols   = len(df.select_dtypes(include='number').columns)
    cat_cols   = len(df.select_dtypes(include='object').columns)
    duplicates = df.duplicated().sum()

    # Конвертируем графики
    imgs_html = ""
    for i, fig in enumerate(figures[:6]):
        b64  = fig_to_base64(fig)
        imgs_html += f"""
        <div class="chart-card">
            <img src="data:image/png;base64,{b64}"
                 alt="График {i+1}" style="width:100%;border-radius:8px;">
        </div>"""

    # Превью данных
    preview_html = df.head(8).to_html(
        classes='data-table',
        border=0,
        index=False
    )

    # Статистика числовых колонок
    num_stats_html = ""
    num_df = df.select_dtypes(include='number')
    if not num_df.empty:
        num_stats_html = num_df.describe().round(2).to_html(
            classes='data-table',
            border=0
        )

    # Форматируем отчёт LLM
    report_formatted = final_report.replace('\n', '<br>') if final_report else \
        "Отчёт не был сформирован."

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Analyst Agent — Отчёт</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0a0f1e;
    color: #e2e8f0;
    line-height: 1.7;
  }}

  .header {{
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
    padding: 3rem 2rem;
    text-align: center;
    border-bottom: 1px solid #4c1d95;
  }}

  .header h1 {{
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
  }}

  .header .meta {{
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 0.5rem;
  }}

  .badge {{
    display: inline-block;
    background: #1e293b;
    border: 1px solid #4c1d95;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.82rem;
    color: #a78bfa;
    margin: 0.2rem;
  }}

  .container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
  }}

  .section {{
    margin-bottom: 2.5rem;
  }}

  .section-title {{
    font-size: 1.35rem;
    font-weight: 700;
    color: #a78bfa;
    border-left: 4px solid #7c3aed;
    padding-left: 1rem;
    margin-bottom: 1.2rem;
  }}

  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }}

  .metric-card {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
  }}

  .metric-card:hover {{
    border-color: #7c3aed;
  }}

  .metric-value {{
    font-size: 1.8rem;
    font-weight: 800;
    color: #a78bfa;
  }}

  .metric-label {{
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.3rem;
  }}

  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
    gap: 1.2rem;
  }}

  .chart-card {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
    overflow: hidden;
  }}

  .report-content {{
    background: #1e293b;
    border: 1px solid #4c1d95;
    border-radius: 12px;
    padding: 2rem;
    line-height: 1.9;
  }}

  .report-content h2 {{
    color: #a78bfa;
    font-size: 1.15rem;
    margin: 1.5rem 0 0.5rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #334155;
  }}

  .report-content strong {{
    color: #60a5fa;
  }}

  .data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    overflow-x: auto;
  }}

  .data-table th {{
    background: #312e81;
    color: #a78bfa;
    padding: 0.6rem 0.8rem;
    text-align: left;
    font-weight: 600;
  }}

  .data-table td {{
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid #1e293b;
    color: #cbd5e1;
  }}

  .data-table tr:nth-child(even) td {{
    background: #0f172a;
  }}

  .table-wrapper {{
    overflow-x: auto;
    border-radius: 8px;
    border: 1px solid #334155;
  }}

  .instruction-box {{
    background: #0f2027;
    border: 1px solid #0891b2;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #7dd3fc;
    font-style: italic;
    margin-bottom: 1rem;
  }}

  .footer {{
    text-align: center;
    padding: 2rem;
    color: #475569;
    font-size: 0.85rem;
    border-top: 1px solid #1e293b;
    margin-top: 3rem;
  }}

  @media (max-width: 768px) {{
    .header h1 {{ font-size: 1.8rem; }}
    .charts-grid {{ grid-template-columns: 1fr; }}
    .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>

<div class="header">
  <div style="font-size:2.5rem;margin-bottom:0.5rem;">🛸</div>
  <h1>Data Analyst Agent</h1>
  <div class="meta">Автоматический анализ данных с помощью ИИ-агента</div>
  <div style="margin-top:1rem;">
    <span class="badge">📁 {filename}</span>
    <span class="badge">📅 {now}</span>
    <span class="badge">🤖 llama-3.3-70b-versatile</span>
    <span class="badge">⚡ Groq API</span>
  </div>
</div>

<div class="container">

  <!-- Метрики -->
  <div class="section">
    <div class="section-title">📊 Общие метрики датасета</div>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value">{rows:,}</div>
        <div class="metric-label">Строк</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{cols}</div>
        <div class="metric-label">Столбцов</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{num_cols}</div>
        <div class="metric-label">Числовых колонок</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{cat_cols}</div>
        <div class="metric-label">Категориальных колонок</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{nulls:,}</div>
        <div class="metric-label">Пропусков ({null_pct}%)</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{duplicates:,}</div>
        <div class="metric-label">Дубликатов</div>
      </div>
    </div>
  </div>

  <!-- Инструкция пользователя -->
  {"" if not user_instruction else f'''
  <div class="section">
    <div class="section-title">💬 Инструкция пользователя</div>
    <div class="instruction-box">{user_instruction}</div>
  </div>
  '''}

  <!-- Графики -->
  {"" if not imgs_html else f'''
  <div class="section">
    <div class="section-title">📈 Визуализация</div>
    <div class="charts-grid">{imgs_html}</div>
  </div>
  '''}

  <!-- Отчёт агента -->
  <div class="section">
    <div class="section-title">🤖 Отчёт ИИ-агента</div>
    <div class="report-content">
      {report_formatted}
    </div>
  </div>

  <!-- Превью данных -->
  <div class="section">
    <div class="section-title">👀 Превью данных (первые 8 строк)</div>
    <div class="table-wrapper">
      {preview_html}
    </div>
  </div>

  <!-- Статистика -->
  {"" if num_stats_html == "" else f'''
  <div class="section">
    <div class="section-title">📐 Описательная статистика</div>
    <div class="table-wrapper">
      {num_stats_html}
    </div>
  </div>
  '''}

</div>

<div class="footer">
  Сгенерировано Data Analyst Agent • {now} •
  Модель: llama-3.3-70b-versatile via Groq API
</div>

</body>
</html>"""

    return html