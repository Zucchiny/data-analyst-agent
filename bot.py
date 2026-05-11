import telebot
from telebot import types
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

warnings.filterwarnings('ignore')

BOT_TOKEN     = "8607006455:AAEXQLLIhD5hows2GFnztsH3fS_xH09hYuU"
API_KEY       = "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"
MODEL         = "llama-3.1-8b-instant"
MAX_ITER      = 6
MAX_ROWS      = 3000
SLEEP_BETWEEN = 3

bot    = telebot.TeleBot(BOT_TOKEN)
client = Groq(api_key=API_KEY)

user_states = {}

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
]

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

def make_safe_globals(df: pd.DataFrame) -> dict:
    return {
        "pd": pd, "pandas": pd,
        "np": np, "numpy": np,
        "plt": plt, "matplotlib": matplotlib,
        "sns": sns, "seaborn": sns,
        "df": df.copy(),
        "Counter": Counter,
        "datetime": datetime,
        "math": math,
        "re": re,
        "print": print,
        "len": len, "range": range,
        "enumerate": enumerate,
        "zip": zip, "list": list,
        "dict": dict, "set": set,
        "tuple": tuple, "str": str,
        "int": int, "float": float,
        "bool": bool, "round": round,
        "sum": sum, "min": min,
        "max": max, "abs": abs,
        "sorted": sorted, "reversed": reversed,
        "isinstance": isinstance, "type": type,
        "any": any, "all": all,
        "map": map, "filter": filter,
        "hasattr": hasattr, "getattr": getattr,
        "None": None, "True": True, "False": False,
        "__builtins__": {
            "print": print, "len": len,
            "range": range, "str": str,
            "int": int, "float": float,
            "list": list, "dict": dict,
            "set": set, "tuple": tuple,
            "bool": bool, "round": round,
            "sum": sum, "min": min, "max": max,
            "abs": abs, "sorted": sorted,
            "isinstance": isinstance, "type": type,
            "any": any, "all": all,
            "None": None, "True": True, "False": False,
        },
    }

def execute_code(code: str, df: pd.DataFrame) -> dict:
    is_safe, reason = sanitize_code(code)
    if not is_safe:
        return {"success": False, "output": f"Заблокировано: {reason}", "figures": []}

    plt.close('all')
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    figures    = []

    try:
        safe_globals = make_safe_globals(df)
        exec(code, safe_globals)
        output = sys.stdout.getvalue()

        for fig_num in plt.get_fignums():
            figures.append(plt.figure(fig_num))

        return {
            "success": True,
            "output": output.strip()[:1500] if output.strip() else "Выполнено",
            "figures": figures
        }
    except Exception as e:
        return {
            "success": False,
            "output": f"Ошибка: {str(e)[:300]}",
            "figures": []
        }
    finally:
        sys.stdout = old_stdout

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
            safe_lines = []
            for line in code.split('\n'):
                s = line.strip()
                if any(s.startswith(bad) for bad in [
                    'import os', 'import sys', 'import subprocess',
                    'import socket', 'from os', 'from sys',
                    'from subprocess', 'from socket'
                ]):
                    continue
                safe_lines.append(line)
            return '\n'.join(safe_lines)
    return None

def build_df_info(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    stats = ""
    for col in num_cols[:4]:
        s = df[col].dropna()
        if len(s) > 0:
            stats += (
                f"  {col}: mean={s.mean():.2f}, "
                f"median={s.median():.2f}, max={s.max():.2f}\n"
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
        f"Статистика:\n{stats}"
        f"Топ категорий:\n{top_vals}"
    )

def build_system_prompt(df_info: str) -> str:
    return (
        "Ты профессиональный аналитик данных. "
        "Анализируй датасет пошагово через Python-код.\n\n"
        f"ДАТАСЕТ:\n{df_info}\n\n"
        "ДОСТУПНЫЕ БИБЛИОТЕКИ (уже импортированы, не пиши import):\n"
        "pd, np, plt, sns, df, math, re, Counter, datetime\n\n"
        "ПРАВИЛА:\n"
        "- НЕ пиши import — всё уже доступно\n"
        "- Один блок ```python``` за шаг\n"
        "- Используй print() для вывода\n"
        "- df уже загружен\n\n"
        "ШАГИ:\n"
        "1 → структура данных\n"
        "2 → статистика\n"
        "3 → тренды + графики\n"
        "4 → аномалии\n"
        "5 → FINAL_REPORT: подробный отчёт на русском (без кода)"
    )

def safe_send(chat_id: int, text: str, **kwargs):
    try:
        bot.send_message(chat_id, text, parse_mode="Markdown", **kwargs)
    except Exception:
        try:
            clean = re.sub(r'(?<!\\)[*_`\[\]]', '', text)
            bot.send_message(chat_id, clean, **kwargs)
        except Exception as e:
            try:
                bot.send_message(chat_id, text[:4000], **kwargs)
            except Exception:
                pass

def main_keyboard():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    kb.add(
        types.KeyboardButton("📊 Стандартный анализ"),
        types.KeyboardButton("📈 Найти тренды"),
        types.KeyboardButton("⚠️ Найти аномалии"),
        types.KeyboardButton("💡 Бизнес-выводы"),
        types.KeyboardButton("❓ Помощь"),
        types.KeyboardButton("🔄 Сброс")
    )
    return kb

def run_agent_bot(df: pd.DataFrame, instruction: str, chat_id: int):
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)
        bot.send_message(
            chat_id,
            f"ℹ️ Датасет большой — используется выборка {MAX_ROWS:,} строк"
        )

    df_info       = build_df_info(df)
    system_prompt = build_system_prompt(df_info)
    messages      = [{
        "role": "user",
        "content": (
            f"Анализируй датасет за 5 шагов.\n"
            f"Инструкция: {instruction[:200]}\n"
            f"Не пиши import — pd, np, plt, sns, df уже доступны.\n"
            f"Начинай с шага 1."
        )
    }]

    all_figures  = []
    final_report = None
    all_outputs  = []

    steps_labels = [
        "▰▱▱▱▱ 20% — Изучаю структуру",
        "▰▰▱▱▱ 40% — Считаю статистику",
        "▰▰▰▱▱ 60% — Ищу тренды",
        "▰▰▰▰▱ 80% — Анализирую аномалии",
        "▰▰▰▰▰ 100% — Формирую отчёт",
    ]

    progress_msg = bot.send_message(
        chat_id,
        "🤖 *Агент начал анализ...*\n\n▱▱▱▱▱ 0% — Подготовка",
        parse_mode="Markdown"
    )

    for iteration in range(MAX_ITER):
        if iteration > 0:
            time.sleep(SLEEP_BETWEEN)

        if iteration < len(steps_labels):
            try:
                bot.edit_message_text(
                    f"🤖 *Агент работает...*\n\n{steps_labels[iteration]}",
                    chat_id=chat_id,
                    message_id=progress_msg.message_id,
                    parse_mode="Markdown"
                )
            except Exception:
                pass

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
            error_str = str(e)
            if "429" in error_str:
                bot.send_message(chat_id, "⏳ Лимит запросов. Жду 30 секунд...")
                time.sleep(30)
                continue
            elif "413" in error_str:
                bot.send_message(
                    chat_id,
                    "❌ Запрос слишком большой. "
                    "Попробуй датасет с меньшим количеством столбцов."
                )
                break
            else:
                bot.send_message(chat_id, f"❌ Ошибка API: {error_str[:150]}")
                break

        assistant_msg = response.choices[0].message.content

        if "FINAL_REPORT:" in assistant_msg:
            final_report = assistant_msg.split("FINAL_REPORT:")[1].strip()
            try:
                bot.edit_message_text(
                    "✅ *Анализ завершён!*\n\n▰▰▰▰▰ 100%",
                    chat_id=chat_id,
                    message_id=progress_msg.message_id,
                    parse_mode="Markdown"
                )
            except Exception:
                pass
            break

        code = extract_code(assistant_msg)
        exec_result = {"success": True, "output": "Нет кода", "figures": []}

        if code:
            exec_result = execute_code(code, df)
            if exec_result["figures"]:
                all_figures.extend(exec_result["figures"])
            if exec_result["output"] and exec_result["success"]:
                all_outputs.append(exec_result["output"])

        messages.append({"role": "assistant", "content": assistant_msg})

        next_step    = iteration + 2
        output_short = exec_result["output"][:400]
        feedback     = (
            f"Результат: {output_short}\n"
            f"Шаг {next_step}. "
            f"Не пиши import — pd, np, plt, sns, df уже доступны."
        )

        if next_step >= 5:
            feedback += (
                " ПОСЛЕДНИЙ ШАГ! "
                "Напиши FINAL_REPORT: и отчёт на русском. "
                "400+ слов. Без кода."
            )

        messages.append({"role": "user", "content": feedback})

    # Если агент не дал FINAL_REPORT — просим LLM написать отчёт отдельно
    if not final_report:
        bot.send_message(
            chat_id,
            "📝 Формирую итоговый отчёт...",
        )
        time.sleep(SLEEP_BETWEEN)
        final_report = generate_interpretation(df, all_outputs)

        # Если и это не сработало — делаем базовый отчёт
        if not final_report:
            num_cols = df.select_dtypes(include='number').columns.tolist()
            cat_cols = df.select_dtypes(include='object').columns.tolist()
            nulls    = df.isnull().sum()
            nulls    = nulls[nulls > 0]

            stats_text = ""
            for col in num_cols:
                s = df[col].dropna()
                stats_text += (
                    f"• {col}: среднее={s.mean():.1f}, "
                    f"медиана={s.median():.1f}, "
                    f"макс={s.max():.1f}\n"
                )

            top_text = ""
            for col in cat_cols[:4]:
                top = df[col].value_counts().head(3)
                top_text += f"\n*{col}:*\n"
                for val, cnt in top.items():
                    pct = round(cnt / len(df) * 100, 1)
                    top_text += f"  — {val}: {cnt:,} ({pct}%)\n"

            nulls_text = ""
            for col, cnt in nulls.items():
                pct = round(cnt / len(df) * 100, 1)
                nulls_text += f"  — {col}: {cnt:,} ({pct}%)\n"

            final_report = (
                f"📊 *Общая характеристика:*\n"
                f"Датасет содержит {len(df):,} строк "
                f"и {len(df.columns)} столбцов.\n"
                f"Числовых: {len(num_cols)}, "
                f"категориальных: {len(cat_cols)}.\n"
                f"Пропусков всего: {df.isnull().sum().sum():,}\n\n"
                f"📉 *Числовая статистика:*\n{stats_text}\n"
                f"🔍 *Топ значений:*{top_text}\n"
                f"❌ *Пропуски:*\n"
                + (nulls_text if nulls_text else "  — нет\n") +
                f"\n💡 *Рекомендации:*\n"
                f"• Проверить выбросы в числовых колонках\n"
                f"• Заполнить или удалить пропущенные значения\n"
                f"• Провести дополнительный анализ по времени и географии"
            )

    # Умные графики
    try:
        plt.close('all')
        smart_figs  = smart_visualize(df)
        all_figures = smart_figs + all_figures
    except Exception:
        pass

    return final_report, all_figures

@bot.message_handler(commands=['start'])
def start(message):
    chat_id = message.chat.id
    user_states[chat_id] = {}
    safe_send(
        chat_id,
        "🛸 *Data Analyst Agent*\n\n"
        "Привет! Я ИИ-агент для анализа данных.\n\n"
        "📊 Структура и качество данных\n"
        "📈 Тренды и закономерности\n"
        "⚠️ Аномалии и выбросы\n"
        "📉 Визуализации\n"
        "💡 Бизнес-рекомендации\n\n"
        "👇 *Отправь CSV-файл чтобы начать*",
        reply_markup=main_keyboard()
    )

@bot.message_handler(commands=['help'])
def help_cmd(message):
    safe_send(
        message.chat.id,
        "📖 *Помощь*\n\n"
        "*Как пользоваться:*\n"
        "1. Отправь CSV-файл\n"
        "2. Выбери тип анализа\n"
        "3. Получи отчёт и графики\n\n"
        "*Кнопки:*\n"
        "📊 Стандартный — полный EDA\n"
        "📈 Тренды — динамика и паттерны\n"
        "⚠️ Аномалии — выбросы и ошибки\n"
        "💡 Бизнес — рекомендации\n"
        "🔄 Сброс — новый файл\n\n"
        "*Ограничения:*\n"
        "— Только CSV файлы\n"
        "— До 3000 строк (остальные сэмплируются)\n"
        "— Анализ 1-2 минуты",
        reply_markup=main_keyboard()
    )

@bot.message_handler(content_types=['document'])
def handle_document(message):
    chat_id = message.chat.id

    if not message.document.file_name.endswith('.csv'):
        bot.send_message(chat_id, "⚠️ Отправь файл в формате CSV")
        return

    msg = bot.send_message(chat_id, "📥 Загружаю файл...")

    try:
        file_info  = bot.get_file(message.document.file_id)
        downloaded = bot.download_file(file_info.file_path)

        try:
            df = pd.read_csv(
                io.BytesIO(downloaded),
                encoding="utf-8",
                on_bad_lines="skip"
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                io.BytesIO(downloaded),
                encoding="cp1251",
                on_bad_lines="skip"
            )

    except Exception as e:
        try:
            bot.edit_message_text(
                f"❌ Ошибка: {e}",
                chat_id=chat_id,
                message_id=msg.message_id
            )
        except Exception:
            bot.send_message(chat_id, f"❌ Ошибка: {e}")
        return

    user_states[chat_id] = {
        "df": df,
        "filename": message.document.file_name
    }

    num_cols     = df.select_dtypes(include='number').columns.tolist()
    cat_cols     = df.select_dtypes(include='object').columns.tolist()
    cols_preview = ', '.join(df.columns.tolist()[:8])
    if len(df.columns) > 8:
        cols_preview += '...'

    preview = (
        f"✅ *Файл загружен!*\n\n"
        f"📁 `{message.document.file_name}`\n"
        f"📊 Строк: `{len(df):,}`\n"
        f"📋 Столбцов: `{len(df.columns)}`\n"
        f"🔢 Числовых: `{len(num_cols)}`\n"
        f"🔤 Категориальных: `{len(cat_cols)}`\n"
        f"❌ Пропусков: `{df.isnull().sum().sum():,}`\n\n"
        f"*Столбцы:* `{cols_preview}`\n\n"
        f"👇 *Выбери тип анализа*"
    )

    try:
        bot.edit_message_text(
            preview,
            chat_id=chat_id,
            message_id=msg.message_id,
            parse_mode="Markdown"
        )
    except Exception:
        safe_send(chat_id, preview)

@bot.message_handler(content_types=['text'])
def handle_text(message):
    chat_id = message.chat.id
    text    = message.text.strip()

    if text == "🔄 Сброс":
        user_states[chat_id] = {}
        bot.send_message(
            chat_id,
            "🔄 Сброс! Отправь новый CSV-файл.",
            reply_markup=main_keyboard()
        )
        return

    if text == "❓ Помощь":
        help_cmd(message)
        return

    if chat_id not in user_states or "df" not in user_states.get(chat_id, {}):
        bot.send_message(
            chat_id,
            "👆 Сначала отправь CSV-файл!",
            reply_markup=main_keyboard()
        )
        return

    button_map = {
        "📊 Стандартный анализ": (
            "Полный разведочный анализ: "
            "тренды, аномалии, визуализации, бизнес-выводы."
        ),
        "📈 Найти тренды": (
            "Тренды и закономерности: "
            "динамика по времени, сезонность, паттерны."
        ),
        "⚠️ Найти аномалии": (
            "Аномалии и выбросы через IQR-метод. "
            "Объясни причины каждой аномалии."
        ),
        "💡 Бизнес-выводы": (
            "Конкретные бизнес-выводы и рекомендации "
            "на основе данных."
        ),
    }

    instruction = button_map.get(text, text)

    if check_injection(instruction):
        bot.send_message(
            chat_id,
            "⚠️ Подозрительная инструкция. Переформулируй.",
            reply_markup=main_keyboard()
        )
        return

    run_analysis(chat_id, instruction)

def generate_interpretation(df: pd.DataFrame, outputs: list) -> str:
    """Просит LLM написать человеческий отчёт на основе собранных данных."""
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    stats_text = ""
    for col in num_cols:
        s = df[col].dropna()
        stats_text += (
            f"{col}: mean={s.mean():.1f}, "
            f"median={s.median():.1f}, "
            f"max={s.max():.1f}, "
            f"min={s.min():.1f}\n"
        )

    top_text = ""
    for col in cat_cols[:4]:
        top = df[col].value_counts().head(5).to_dict()
        top_text += f"{col}: {top}\n"

    nulls = {
        col: int(cnt)
        for col, cnt in df.isnull().sum().items()
        if cnt > 0
    }

    code_results = "\n".join(outputs[:3]) if outputs else "нет данных"

    prompt = (
        f"Ты аналитик данных. На основе данных о датасете напиши "
        f"подробный аналитический отчёт на русском языке.\n\n"
        f"ДАННЫЕ О ДАТАСЕТЕ:\n"
        f"Строк: {len(df):,} | Столбцов: {len(df.columns)}\n"
        f"Колонки: {', '.join(df.columns.tolist())}\n"
        f"Числовая статистика:\n{stats_text}\n"
        f"Топ значений:\n{top_text}\n"
        f"Пропуски: {nulls}\n\n"
        f"РЕЗУЛЬТАТЫ АНАЛИЗА КОДА:\n{code_results[:1000]}\n\n"
        f"ТРЕБОВАНИЯ К ОТЧЁТУ:\n"
        f"1. Начни с описания что за датасет и о чём он\n"
        f"2. Опиши ключевые находки с конкретными цифрами\n"
        f"3. Расскажи о трендах и закономерностях\n"
        f"4. Опиши аномалии и проблемы данных\n"
        f"5. Дай практические рекомендации\n"
        f"Минимум 300 слов. Пиши живым языком, не сухо."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return None
    
def run_analysis(chat_id: int, instruction: str):
    df = user_states[chat_id]["df"]

    try:
        final_report, figures = run_agent_bot(df, instruction, chat_id)

        if figures:
            bot.send_message(chat_id, "📊 *Графики:*", parse_mode="Markdown")
            for i, fig in enumerate(figures[:5]):
                try:
                    buf = io.BytesIO()
                    fig.savefig(
                        buf, format='png', dpi=120,
                        bbox_inches='tight', facecolor='#0f172a'
                    )
                    buf.seek(0)
                    bot.send_photo(
                        chat_id, buf,
                        caption=f"График {i+1}/{min(len(figures), 5)}"
                    )
                except Exception:
                    pass
                finally:
                    plt.close(fig)

        if final_report:
            max_len = 3500
            if len(final_report) > max_len:
                chunks  = []
                current = ""
                for line in final_report.split('\n'):
                    if len(current) + len(line) + 1 > max_len:
                        chunks.append(current)
                        current = line
                    else:
                        current += '\n' + line
                if current:
                    chunks.append(current)

                for i, chunk in enumerate(chunks):
                    header = "📋 *Итоговый отчёт:*\n\n" if i == 0 else ""
                    safe_send(chat_id, header + chunk)
            else:
                safe_send(chat_id, "📋 *Итоговый отчёт:*\n\n" + final_report)

            bot.send_message(
                chat_id,
                "✅ *Готово!* Выбери следующее действие:",
                parse_mode="Markdown",
                reply_markup=main_keyboard()
            )
        else:
            bot.send_message(
                chat_id,
                "⚠️ Не удалось получить отчёт. Попробуй снова.",
                reply_markup=main_keyboard()
            )

    except Exception as e:
        bot.send_message(
            chat_id,
            f"❌ Ошибка: {str(e)[:150]}\n\nПопробуй ещё раз.",
            reply_markup=main_keyboard()
        )

if __name__ == "__main__":
    print("🛸 Data Analyst Bot запущен!")
    print(f"   Модель : {MODEL}")
    print("   Бот    : @data_analyst_ufo_bot")
    print("   Жду сообщений...")
    bot.infinity_polling(timeout=30, long_polling_timeout=30)
