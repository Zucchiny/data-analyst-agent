import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import re
import time
import math
import warnings
from datetime import datetime
from collections import Counter
from groq import Groq
from agent_tools import smart_visualize
from report_builder import build_html_report

warnings.filterwarnings('ignore')

# ── настройки страницы ─────────────────────────────────────
st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="🛸",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0 0.2rem 0;
    }
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .report-box {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid #7c3aed;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .injection-warning {
        background: #450a0a;
        border: 1px solid #991b1b;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #fca5a5;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── константы ──────────────────────────────────────────────
API_KEY        = "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"
MODEL          = "llama-3.1-8b-instant"
MAX_ITERATIONS = 6
MAX_ROWS       = 3000
SLEEP_BETWEEN  = 3

client = Groq(api_key=API_KEY)

# ── безопасность ───────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(a\s+)?(?!analyst|data)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"disregard\s+(all\s+)?",
    r"override\s+(all\s+)?",
    r"new\s+instructions?\s*:",
    r"system\s*prompt\s*:",
    r"<\s*script",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"\[INST\]",
    r"<<SYS>>",
]

# Только реально опасное — без import, numpy и т.д.
DANGEROUS_CODE = [
    "subprocess",
    "os.system",
    "os.popen",
    "os.remove",
    "os.rmdir",
    "os.mkdir",
    "shutil",
    "socket",
    "requests.get",
    "requests.post",
    "urllib.request",
    "open(",
    "file(",
    "compile(",
    "globals()",
    "locals()",
    "vars(",
]

def check_injection(text: str) -> bool:
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

def sanitize_code(code: str) -> tuple[bool, str]:
    for danger in DANGEROUS_CODE:
        if danger in code:
            return False, danger
    return True, ""

# ── безопасный sandbox ─────────────────────────────────────
def make_safe_globals(df: pd.DataFrame) -> dict:
    """Создаёт безопасное окружение с нужными библиотеками."""
    return {
        # Основные библиотеки
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
        "plt": plt,
        "matplotlib": matplotlib,
        "sns": sns,
        "seaborn": sns,
        # Данные — копия чтобы агент не испортил оригинал
        "df": df.copy(),
        # Стандартные утилиты
        "Counter": Counter,
        "datetime": datetime,
        "math": math,
        "re": re,
        # Встроенные функции Python
        "print": print,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "round": round,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "sorted": sorted,
        "reversed": reversed,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "type": type,
        "any": any,
        "all": all,
        "map": map,
        "filter": filter,
        "hasattr": hasattr,
        "getattr": getattr,
        "vars": vars,
        "dir": dir,
        "help": help,
        "None": None,
        "True": True,
        "False": False,
        # Отключаем опасные встроенные
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "round": round,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "sorted": sorted,
            "isinstance": isinstance,
            "type": type,
            "any": any,
            "all": all,
            "None": None,
            "True": True,
            "False": False,
        },
    }

# ── выполнение кода ────────────────────────────────────────
def execute_code(code: str, df: pd.DataFrame) -> dict:
    # Проверка безопасности
    is_safe, reason = sanitize_code(code)
    if not is_safe:
        return {
            "success": False,
            "output": f"⛔ Заблокировано: `{reason}`",
            "figures": []
        }

    # Закрываем старые графики перед выполнением
    plt.close('all')

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    figures    = []

    try:
        safe_globals = make_safe_globals(df)
        exec(code, safe_globals)
        output = sys.stdout.getvalue()

        # Собираем новые графики
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            figures.append(fig)

        return {
            "success": True,
            "output": output.strip()[:2000] if output.strip() else "✓ Выполнено",
            "figures": figures
        }

    except Exception as e:
        output = sys.stdout.getvalue()
        return {
            "success": False,
            "output": f"❌ Ошибка: {str(e)[:500]}\n{output[:200]}",
            "figures": []
        }
    finally:
        sys.stdout = old_stdout

# ── извлечение кода ────────────────────────────────────────
def extract_code(text: str):
    for pattern in [
        r"```python\n(.*?)```",
        r"```py\n(.*?)```",
        r"```\n(.*?)```",
        r"```python(.*?)```",
    ]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Убираем строки с import os, import sys и т.д.
            safe_lines = []
            for line in code.split('\n'):
                stripped = line.strip()
                if stripped.startswith('import os') or \
                   stripped.startswith('import sys') or \
                   stripped.startswith('import subprocess') or \
                   stripped.startswith('import socket') or \
                   stripped.startswith('from os') or \
                   stripped.startswith('from sys') or \
                   stripped.startswith('from subprocess'):
                    continue
                safe_lines.append(line)
            return '\n'.join(safe_lines)
    return None

# ── компактный контекст ────────────────────────────────────
def build_df_info(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    stats = ""
    for col in num_cols[:4]:
        s = df[col].dropna()
        if len(s) > 0:
            stats += (
                f"  {col}: "
                f"mean={s.mean():.2f}, "
                f"median={s.median():.2f}, "
                f"std={s.std():.2f}, "
                f"max={s.max():.2f}\n"
            )

    top_vals = ""
    for col in cat_cols[:3]:
        top = df[col].value_counts().head(3).to_dict()
        top_vals += f"  {col}: {top}\n"

    nulls = dict(df.isnull().sum()[df.isnull().sum() > 0])

    return (
        f"Строк: {len(df):,} | Столбцов: {len(df.columns)}\n"
        f"Колонки: {', '.join(df.columns.tolist())}\n"
        f"Числовые: {num_cols}\n"
        f"Категориальные: {cat_cols}\n"
        f"Пропуски: {nulls}\n"
        f"Числовая статистика:\n{stats}"
        f"Топ категорий:\n{top_vals}"
    )

# ── системный промпт ───────────────────────────────────────
def build_system_prompt(df_info: str) -> str:
    return (
        "Ты профессиональный аналитик данных. "
        "Анализируй датасет пошагово через Python-код.\n\n"
        f"ДАТАСЕТ:\n{df_info}\n\n"
        "ДОСТУПНЫЕ БИБЛИОТЕКИ (уже импортированы):\n"
        "pd (pandas), np (numpy), plt (matplotlib.pyplot), "
        "sns (seaborn), df (датасет)\n\n"
        "ВАЖНО:\n"
        "- НЕ пиши import в коде — всё уже доступно\n"
        "- Используй print() для вывода результатов\n"
        "- Один блок ```python ... ``` за шаг\n"
        "- df уже загружен как переменная\n\n"
        "ПЛАН (5 шагов):\n"
        "1 → структура: типы, пропуски, уникальные значения\n"
        "2 → статистика: describe(), топ значений\n"
        "3 → тренды: группировки, динамика + 1-2 графика\n"
        "4 → аномалии: IQR-метод, выбросы\n"
        "5 → FINAL_REPORT: подробный текстовый отчёт на русском\n\n"
        "НА ШАГЕ 5:\n"
        "- НЕ пиши код\n"
        "- Напиши: FINAL_REPORT:\n"
        "- После — подробный отчёт минимум 400 слов\n"
        "- Разделы: характеристика, находки, тренды, аномалии, рекомендации"
    )

# ── агентный цикл ──────────────────────────────────────────
def run_agent(df: pd.DataFrame, user_instruction: str):
    # Сэмплируем если большой
    if len(df) > MAX_ROWS:
        df_work = df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)
        st.info(
            f"ℹ️ Датасет большой — используется выборка {MAX_ROWS:,} строк"
        )
    else:
        df_work = df.copy()

    df_info       = build_df_info(df_work)
    system_prompt = build_system_prompt(df_info)

    messages = [{
        "role": "user",
        "content": (
            f"Проанализируй датасет за 5 шагов.\n"
            f"Инструкция: {user_instruction[:200]}\n"
            f"Помни: не пиши import — всё уже доступно (pd, np, plt, sns, df).\n"
            f"Начинай с шага 1."
        )
    }]

    all_figures  = []
    final_report = None
    steps_log    = []
    all_outputs  = []

    progress_bar = st.progress(0)
    status_text  = st.empty()
    steps_area   = st.container()

    for iteration in range(MAX_ITERATIONS):
        progress_bar.progress(int((iteration / MAX_ITERATIONS) * 100))
        status_text.info(
            f"🤖 Шаг {iteration + 1}/{MAX_ITERATIONS} — агент думает..."
        )

        if iteration > 0:
            time.sleep(SLEEP_BETWEEN)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt}
                ] + messages,
                max_tokens=1500,
                temperature=0.1,
            )
        except Exception as e:
            error_msg = str(e)
            if "413" in error_msg or "too large" in error_msg.lower():
                status_text.error(
                    "❌ Запрос слишком большой. "
                    "Попробуй датасет с меньшим количеством столбцов."
                )
            elif "429" in error_msg:
                status_text.warning("⏳ Лимит запросов. Жду 30 секунд...")
                time.sleep(30)
                continue
            else:
                status_text.error(f"❌ Ошибка API: {error_msg[:200]}")
            break

        assistant_msg = response.choices[0].message.content

        # Финальный отчёт
        if "FINAL_REPORT:" in assistant_msg:
            final_report = assistant_msg.split("FINAL_REPORT:")[1].strip()
            progress_bar.progress(100)
            status_text.success("✅ Анализ завершён!")
            with steps_area:
                with st.expander(
                    f"📝 Шаг {iteration+1} — финальный отчёт",
                    expanded=False
                ):
                    st.markdown(assistant_msg)
            break

        with steps_area:
            with st.expander(
                f"🔍 Шаг {iteration+1} — агент думает",
                expanded=False
            ):
                st.markdown(assistant_msg)

        code = extract_code(assistant_msg)
        exec_result = {
            "success": True,
            "output": "Код не найден в ответе",
            "figures": []
        }

        if code:
            exec_result = execute_code(code, df_work)
            if exec_result["figures"]:
                all_figures.extend(exec_result["figures"])
            if exec_result["output"] and exec_result["success"]:
                all_outputs.append(exec_result["output"])

            with steps_area:
                with st.expander(
                    f"⚙️ Шаг {iteration+1} — результат",
                    expanded=False
                ):
                    if exec_result["success"]:
                        st.code(exec_result["output"], language="text")
                    else:
                        st.error(exec_result["output"])

        steps_log.append({
            "step":          iteration + 1,
            "code_executed": code is not None,
            "success":       exec_result["success"]
        })

        messages.append({"role": "assistant", "content": assistant_msg})

        next_step    = iteration + 2
        output_short = exec_result["output"][:400]
        feedback     = (
            f"Результат шага {iteration+1}:\n{output_short}\n\n"
            f"Переходи к шагу {next_step}.\n"
            f"Не пиши import — pd, np, plt, sns, df уже доступны."
        )

        if next_step >= 5:
            feedback += (
                "\n\nЭТО ПОСЛЕДНИЙ ШАГ. "
                "Напиши FINAL_REPORT: и подробный отчёт на русском. "
                "Минимум 400 слов. Конкретные цифры. Без кода."
            )

        messages.append({"role": "user", "content": feedback})

    # Fallback
    if not final_report:
        if all_outputs:
            final_report = (
                "## Результаты анализа\n\n"
                + "\n\n---\n\n".join(all_outputs[:5])
            )
        else:
            final_report = (
                "## Результаты анализа\n\n"
                f"Датасет: {len(df_work):,} строк, "
                f"{len(df_work.columns)} столбцов.\n\n"
                f"Колонки: {', '.join(df_work.columns.tolist())}"
            )

    # Умные графики
    try:
        plt.close('all')
        smart_figs  = smart_visualize(df_work)
        all_figures = smart_figs + all_figures
    except Exception as e:
        st.warning(f"⚠️ Автоматические графики: {e}")

    return final_report, all_figures, steps_log

# ══════════════════════════════════════════════════════════
# ИНТЕРФЕЙС
# ══════════════════════════════════════════════════════════
st.markdown(
    '<div class="main-header">🛸 Data Analyst Agent</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">'
    'Загрузи любой CSV — агент проведёт полный анализ с графиками и выводами'
    '</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### ⚙️ О приложении")
    st.markdown(f"**Модель:** `{MODEL}`")
    st.markdown("**API:** Groq Cloud")
    st.markdown("**Агент:** ReAct (думает → код → результат → вывод)")
    st.divider()
    st.markdown("### 📋 Инструкция")
    st.markdown("""
1. 📂 Загрузи CSV-файл
2. ✍️ Напиши инструкцию (опционально)
3. 🚀 Нажми **Запустить анализ**
4. 👀 Наблюдай за шагами агента
5. 📥 Скачай HTML-отчёт
""")
    st.divider()
    st.markdown("### 🛡️ Безопасность")
    st.markdown("""
- ✅ Защита от prompt injection
- ✅ Изолированное выполнение кода
- ✅ Блокировка опасных операций
- ✅ Лимит строк датасета (3000)
""")
    st.divider()
    st.markdown("### 🤖 Telegram-бот")
    st.markdown("[@data_analyst_ufo_bot](https://t.me/data_analyst_ufo_bot)")
    st.divider()
    st.markdown("### 💡 Примеры датасетов")
    st.markdown("""
- 🛸 UFO Sightings (NUFORC)
- 🚢 Titanic
- 🌸 Iris
- 📈 Любой CSV!
""")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📂 Загрузка данных")
    uploaded_file = st.file_uploader(
        "Выбери CSV-файл",
        type=["csv"],
        help="Максимальный размер: 200MB"
    )

with col2:
    st.markdown("### 💬 Инструкция для агента")
    user_instruction = st.text_area(
        "Что нужно проанализировать? (опционально)",
        placeholder=(
            "Например: обрати внимание на сезонность, "
            "найди аномалии в длительности, "
            "сделай акцент на географии..."
        ),
        height=130
    )

    if user_instruction and check_injection(user_instruction):
        st.markdown(
            '<div class="injection-warning">'
            '⚠️ <b>Обнаружена подозрительная инструкция.</b> '
            'Переформулируй запрос.</div>',
            unsafe_allow_html=True
        )
        user_instruction = "Проведи стандартный разведочный анализ данных."

if uploaded_file:
    try:
        df = pd.read_csv(
            uploaded_file, encoding="utf-8", on_bad_lines="skip"
        )
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(
                uploaded_file, encoding="cp1251", on_bad_lines="skip"
            )
        except Exception as e:
            st.error(f"Не удалось прочитать файл: {e}")
            st.stop()
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        st.stop()

    st.divider()
    st.markdown("### 👀 Предпросмотр датасета")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📊 Строк",     f"{len(df):,}")
    m2.metric("📋 Столбцов",  len(df.columns))
    m3.metric("❌ Пропусков", f"{df.isnull().sum().sum():,}")
    m4.metric("🔢 Числовых",  len(df.select_dtypes(include='number').columns))
    m5.metric("📁 Размер",    f"{uploaded_file.size // 1024} KB")

    st.dataframe(df.head(8), use_container_width=True)

    if not user_instruction:
        user_instruction = (
            "Проведи полный разведочный анализ. "
            "Найди тренды, аномалии, визуализации и бизнес-выводы на русском."
        )

    st.divider()
    run_btn = st.button(
        "🚀 Запустить анализ",
        type="primary",
        use_container_width=True
    )

    if run_btn:
        st.divider()
        st.markdown("### 🤖 Работа агента")

        final_report, figures, steps_log = run_agent(df, user_instruction)

        if figures:
            st.divider()
            st.markdown("### 📊 Графики")
            cols = st.columns(2)
            for i, fig in enumerate(figures):
                try:
                    with cols[i % 2]:
                        st.pyplot(fig)
                except Exception:
                    pass
                finally:
                    plt.close(fig)

        if final_report:
            st.divider()
            st.markdown("### 📋 Итоговый отчёт агента")
            st.markdown(
                f'<div class="report-box">{final_report}</div>',
                unsafe_allow_html=True
            )

            st.divider()
            try:
                html_report = build_html_report(
                    df=df,
                    final_report=final_report,
                    figures=[],
                    filename=uploaded_file.name,
                    user_instruction=user_instruction
                )
                st.download_button(
                    label="📥 Скачать полный отчёт (HTML)",
                    data=html_report.encode('utf-8'),
                    file_name=f"analysis_{uploaded_file.name.replace('.csv','')}.html",
                    mime="text/html",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"HTML отчёт недоступен: {e}")

            st.divider()
            st.markdown("### 📈 Статистика работы агента")
            s1, s2, s3 = st.columns(3)
            s1.metric("Шагов выполнено",   len(steps_log))
            s2.metric("Успешных",           sum(1 for s in steps_log if s["success"]))
            s3.metric("Графиков построено", len(figures))
        else:
            st.error(
                "❌ Не удалось получить результат. "
                "Попробуй запустить снова."
            )
else:
    st.divider()
    st.markdown("""
    <div style='text-align:center; padding: 3rem; color: #475569;'>
        <div style='font-size: 4rem;'>📂</div>
        <div style='font-size: 1.2rem; margin-top: 1rem;'>
            Загрузи CSV-файл чтобы начать
        </div>
        <div style='font-size: 0.9rem; margin-top: 0.5rem;'>
            Агент сам проведёт полный анализ данных
        </div>
    </div>
    """, unsafe_allow_html=True)
