import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── палитра и стиль ────────────────────────────────────────
PALETTE = ["#7c3aed", "#2563eb", "#059669", "#dc2626", "#d97706",
           "#0891b2", "#db2777", "#65a30d", "#ea580c", "#0f766e"]

BG_DARK  = '#0f172a'
BG_CARD  = '#1e293b'
BORDER   = '#334155'
TEXT     = '#e2e8f0'
SUBTEXT  = '#94a3b8'

def set_style():
    plt.style.use('dark_background')
    sns.set_palette(PALETTE)

def style_ax(ax, title=""):
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors=SUBTEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11,
                     fontweight='bold', pad=10)

# ══════════════════════════════════════════════════════════
# ИНСТРУМЕНТ 1 — Полная сводка
# ══════════════════════════════════════════════════════════
def full_summary(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 55)
    lines.append("ПОЛНАЯ СВОДКА ПО ДАТАСЕТУ")
    lines.append("=" * 55)
    lines.append(f"Строк        : {len(df):,}")
    lines.append(f"Столбцов     : {len(df.columns)}")
    lines.append(f"Дубликатов   : {df.duplicated().sum():,}")
    lines.append(f"Пустых ячеек : {df.isnull().sum().sum():,}")
    lines.append(f"Память       : {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    lines.append("")

    lines.append("── Столбцы ──")
    for col in df.columns:
        nulls  = df[col].isnull().sum()
        pct    = round(nulls / len(df) * 100, 1)
        dtype  = str(df[col].dtype)
        unique = df[col].nunique()
        lines.append(
            f"  {col:<30} {dtype:<12} "
            f"nulls={nulls}({pct}%)  unique={unique}"
        )

    lines.append("")
    lines.append("── Числовая статистика ──")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().round(2)
        lines.append(desc.to_string())

    lines.append("")
    lines.append("── Категориальные столбцы (топ-5) ──")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols[:8]:
        top = df[col].value_counts().head(5)
        lines.append(f"\n  [{col}]")
        for val, cnt in top.items():
            pct = round(cnt / len(df) * 100, 1)
            lines.append(f"    {str(val):<25} {cnt:>7,}  ({pct}%)")

    return "\n".join(lines)

# ══════════════════════════════════════════════════════════
# ИНСТРУМЕНТ 2 — Аномалии
# ══════════════════════════════════════════════════════════
def detect_anomalies(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 55)
    lines.append("ОБНАРУЖЕНИЕ АНОМАЛИЙ (IQR-метод)")
    lines.append("=" * 55)

    num_cols = df.select_dtypes(include='number').columns.tolist()

    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        Q1  = series.quantile(0.25)
        Q3  = series.quantile(0.75)
        IQR = Q3 - Q1
        lower   = Q1 - 1.5 * IQR
        upper   = Q3 + 1.5 * IQR
        outliers = series[(series < lower) | (series > upper)]
        pct = round(len(outliers) / len(series) * 100, 1)

        skew = round(series.skew(), 2)
        kurt = round(series.kurt(), 2)

        lines.append(f"\n[{col}]")
        lines.append(f"  Медиана      : {series.median():>15,.2f}")
        lines.append(f"  Среднее      : {series.mean():>15,.2f}")
        lines.append(f"  Станд. откл. : {series.std():>15,.2f}")
        lines.append(f"  Асимметрия   : {skew:>15}")
        lines.append(f"  Эксцесс      : {kurt:>15}")
        lines.append(f"  IQR-граница  : [{lower:,.2f} ; {upper:,.2f}]")
        lines.append(f"  Выбросов     : {len(outliers):,} ({pct}%)")
        if len(outliers) > 0:
            lines.append(f"  Макс значение: {series.max():,.2f}")
            lines.append(f"  Мин значение : {series.min():,.2f}")

    return "\n".join(lines)

# ══════════════════════════════════════════════════════════
# ИНСТРУМЕНТ 3 — Временные тренды
# ══════════════════════════════════════════════════════════
def analyze_time_trends(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 55)
    lines.append("АНАЛИЗ ВРЕМЕННЫХ ТРЕНДОВ")
    lines.append("=" * 55)

    date_cols = [
        col for col in df.columns
        if any(w in col.lower()
               for w in ['date', 'time', 'year', 'month', 'дата', 'год'])
    ]

    if not date_cols:
        return "Временные столбцы не обнаружены"

    month_names = {
        1:'Январь', 2:'Февраль', 3:'Март', 4:'Апрель',
        5:'Май', 6:'Июнь', 7:'Июль', 8:'Август',
        9:'Сентябрь', 10:'Октябрь', 11:'Ноябрь', 12:'Декабрь'
    }

    for col in date_cols[:2]:
        lines.append(f"\n[{col}]")
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            valid  = parsed.dropna()
            if len(valid) < 10:
                continue

            lines.append(f"  Период    : {valid.min().year} — {valid.max().year}")
            lines.append(f"  Дат всего : {len(valid):,}")

            by_year = valid.dt.year.value_counts().sort_index()
            peak_y  = int(by_year.idxmax())
            lines.append(f"  Пик (год) : {peak_y} ({int(by_year.max()):,} записей)")
            lines.append(f"  Топ-5 лет :")
            for yr, cnt in by_year.sort_values(ascending=False).head(5).items():
                bar = '█' * int(cnt / by_year.max() * 20)
                lines.append(f"    {int(yr)}: {bar} {cnt:,}")

            by_month  = valid.dt.month.value_counts().sort_index()
            peak_m    = int(by_month.idxmax())
            low_m     = int(by_month.idxmin())
            lines.append(f"  Пик (мес) : {month_names.get(peak_m, peak_m)} ({int(by_month.max()):,})")
            lines.append(f"  Спад (мес): {month_names.get(low_m, low_m)} ({int(by_month.min()):,})")

            by_hour = valid.dt.hour.value_counts().sort_index()
            if len(by_hour) > 1:
                peak_h = int(by_hour.idxmax())
                lines.append(f"  Пик (час) : {peak_h}:00 ({int(by_hour.max()):,})")

        except Exception as e:
            lines.append(f"  Ошибка: {e}")

    return "\n".join(lines)

# ══════════════════════════════════════════════════════════
# ИНСТРУМЕНТ 4 — Корреляции
# ══════════════════════════════════════════════════════════
def correlation_analysis(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 55)
    lines.append("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
    lines.append("=" * 55)

    num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) < 2:
        return "Недостаточно числовых столбцов"

    corr = df[num_cols].corr().round(3)
    lines.append("\nМатрица корреляций:")
    lines.append(corr.to_string())

    lines.append("\nСильные корреляции (|r| > 0.4):")
    found = False
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            val = corr.iloc[i, j]
            if abs(val) > 0.4:
                strength  = "сильная" if abs(val) > 0.7 else "умеренная"
                direction = "положительная" if val > 0 else "отрицательная"
                lines.append(
                    f"  {num_cols[i]} ↔ {num_cols[j]}: "
                    f"r={val} ({strength} {direction})"
                )
                found = True
    if not found:
        lines.append("  Значимых корреляций не обнаружено (|r| < 0.4)")

    return "\n".join(lines)

# ══════════════════════════════════════════════════════════
# ИНСТРУМЕНТ 5 — Умная визуализация
# ══════════════════════════════════════════════════════════
def smart_visualize(df: pd.DataFrame) -> list:
    set_style()
    figures = []

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # ── График 1: Распределения числовых переменных ────────
    if num_cols:
        cols_plot = num_cols[:4]
        fig, axes = plt.subplots(2, len(cols_plot),
                                 figsize=(5 * len(cols_plot), 8))
        fig.patch.set_facecolor(BG_DARK)

        if len(cols_plot) == 1:
            axes = [[axes[0]], [axes[1]]]

        for idx, col in enumerate(cols_plot):
            data    = df[col].dropna()
            p1, p99 = data.quantile(0.01), data.quantile(0.99)
            clean   = data[(data >= p1) & (data <= p99)]

            # Гистограмма
            ax_hist = axes[0][idx]
            ax_hist.hist(clean, bins=40, color=PALETTE[idx % len(PALETTE)],
                        alpha=0.85, edgecolor='none')
            ax_hist.axvline(data.median(), color='#f59e0b',
                           linestyle='--', linewidth=1.5,
                           label=f'Медиана: {data.median():.1f}')
            ax_hist.axvline(data.mean(), color='#ef4444',
                           linestyle='--', linewidth=1.5,
                           label=f'Среднее: {data.mean():.1f}')
            style_ax(ax_hist, col)
            ax_hist.legend(fontsize=8, labelcolor=TEXT,
                          facecolor=BG_CARD, edgecolor=BORDER)

            # Boxplot
            ax_box = axes[1][idx]
            bp = ax_box.boxplot(clean, vert=True, patch_artist=True,
                               medianprops=dict(color='#f59e0b', linewidth=2),
                               boxprops=dict(facecolor=PALETTE[idx % len(PALETTE)],
                                            alpha=0.7),
                               whiskerprops=dict(color=SUBTEXT),
                               capprops=dict(color=SUBTEXT),
                               flierprops=dict(marker='o', color=PALETTE[3],
                                              alpha=0.5, markersize=3))
            style_ax(ax_box, f'{col} (boxplot)')

        plt.suptitle('Распределение числовых переменных',
                     color=TEXT, fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        figures.append(fig)

    # ── График 2: Топ категорий ────────────────────────────
    good_cats = [c for c in cat_cols if 2 < df[c].nunique() < 25][:4]
    if good_cats:
        rows = (len(good_cats) + 1) // 2
        cols = min(2, len(good_cats))
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(8 * cols, 5 * rows))
        fig.patch.set_facecolor(BG_DARK)

        if len(good_cats) == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]

        for idx, col in enumerate(good_cats):
            r, c   = divmod(idx, cols)
            ax     = axes[r][c] if rows > 1 else axes[0][idx % cols]
            top    = df[col].value_counts().head(10)
            colors = PALETTE[:len(top)]

            bars = ax.barh(top.index.astype(str)[::-1],
                          top.values[::-1],
                          color=colors[::-1],
                          edgecolor='none', height=0.65)

            for bar, val in zip(bars, top.values[::-1]):
                ax.text(bar.get_width() + max(top.values) * 0.01,
                       bar.get_y() + bar.get_height() / 2,
                       f'{val:,}', va='center',
                       color=SUBTEXT, fontsize=9)

            style_ax(ax, f'Топ значений: {col}')

        # Скрываем лишние оси
        for idx in range(len(good_cats), rows * cols):
            r, c = divmod(idx, cols)
            if rows > 1:
                axes[r][c].set_visible(False)

        plt.suptitle('Распределение категориальных переменных',
                     color=TEXT, fontsize=14, fontweight='bold')
        plt.tight_layout()
        figures.append(fig)

    # ── График 3: Тренд по времени ────────────────────────
    date_col = next(
        (col for col in df.columns
         if any(w in col.lower() for w in ['date', 'time', 'year'])),
        None
    )
    if date_col:
        try:
            parsed = pd.to_datetime(df[date_col], errors='coerce')
            years  = parsed.dt.year.dropna()
            years  = years[(years >= 1900) & (years <= 2030)]

            if len(years) > 50:
                by_year = years.value_counts().sort_index()

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                fig.patch.set_facecolor(BG_DARK)

                # Линейный тренд
                ax = axes[0]
                ax.set_facecolor(BG_CARD)
                ax.fill_between(by_year.index, by_year.values,
                               alpha=0.25, color=PALETTE[0])
                ax.plot(by_year.index, by_year.values,
                       color=PALETTE[0], linewidth=2.5)

                peak_y = int(by_year.idxmax())
                peak_v = int(by_year.max())
                ax.annotate(
                    f'Пик: {peak_y}\n({peak_v:,})',
                    xy=(peak_y, peak_v),
                    xytext=(peak_y - len(by_year) // 8, peak_v * 0.80),
                    arrowprops=dict(arrowstyle='->', color=PALETTE[3],
                                  lw=1.5),
                    color=TEXT, fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor=BG_CARD, edgecolor=PALETTE[3])
                )
                style_ax(ax, 'Динамика записей по годам')
                ax.set_xlabel('Год', color=SUBTEXT)
                ax.set_ylabel('Количество записей', color=SUBTEXT)

                # Сезонность по месяцам
                ax2 = axes[1]
                ax2.set_facecolor(BG_CARD)
                months = parsed.dt.month.dropna().value_counts().sort_index()
                month_labels = ['Янв','Фев','Мар','Апр','Май','Июн',
                               'Июл','Авг','Сен','Окт','Ноя','Дек']
                bar_colors = [PALETTE[3] if v == months.max()
                             else PALETTE[0] for v in months.values]

                bars = ax2.bar(range(1, len(months)+1), months.values,
                              color=bar_colors, edgecolor='none', width=0.7)
                ax2.set_xticks(range(1, 13))
                ax2.set_xticklabels(month_labels, color=SUBTEXT, fontsize=9)
                for bar, val in zip(bars, months.values):
                    ax2.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + months.max() * 0.01,
                            f'{val:,}', ha='center',
                            color=SUBTEXT, fontsize=8)
                style_ax(ax2, 'Сезонность (по месяцам)')
                ax2.set_ylabel('Количество записей', color=SUBTEXT)

                plt.suptitle('Временной анализ',
                            color=TEXT, fontsize=14, fontweight='bold')
                plt.tight_layout()
                figures.append(fig)
        except Exception:
            pass

    # ── График 4: Тепловая карта корреляций ───────────────
    num_cols_corr = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols_corr) >= 2:
        try:
            corr_matrix = df[num_cols_corr].corr()
            fig, ax     = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor(BG_DARK)
            ax.set_facecolor(BG_CARD)

            mask = np.zeros_like(corr_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = True

            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0,
                ax=ax,
                linewidths=0.5,
                linecolor=BG_DARK,
                annot_kws={"size": 10, "color": TEXT},
                cbar_kws={"shrink": 0.8}
            )
            ax.set_title('Тепловая карта корреляций',
                        color=TEXT, fontsize=13, fontweight='bold', pad=15)
            ax.tick_params(colors=SUBTEXT, labelsize=9)
            plt.tight_layout()
            figures.append(fig)
        except Exception:
            pass

    # ── График 5: Пропущенные значения ────────────────────
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=True)
    if len(nulls) > 0:
        fig, ax = plt.subplots(figsize=(8, max(3, len(nulls) * 0.6)))
        fig.patch.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_CARD)

        pcts   = (nulls / len(df) * 100).round(1)
        colors = [PALETTE[3] if p > 20 else PALETTE[1]
                 if p > 5 else PALETTE[2] for p in pcts]

        bars = ax.barh(nulls.index, pcts.values,
                      color=colors, edgecolor='none', height=0.6)
        for bar, val, cnt in zip(bars, pcts.values, nulls.values):
            ax.text(bar.get_width() + 0.3,
                   bar.get_y() + bar.get_height()/2,
                   f'{val}% ({cnt:,})',
                   va='center', color=SUBTEXT, fontsize=9)

        ax.axvline(5,  color=PALETTE[1], linestyle='--',
                  alpha=0.5, linewidth=1, label='5% порог')
        ax.axvline(20, color=PALETTE[3], linestyle='--',
                  alpha=0.5, linewidth=1, label='20% порог')

        style_ax(ax, 'Пропущенные значения по столбцам (%)')
        ax.set_xlabel('Доля пропусков (%)', color=SUBTEXT)
        ax.legend(fontsize=9, labelcolor=TEXT,
                 facecolor=BG_CARD, edgecolor=BORDER)
        plt.tight_layout()
        figures.append(fig)

    return figures

# ══════════════════════════════════════════════════════════
# ИНСТРУМЕНТ 6 — Богатый контекст для LLM
# ══════════════════════════════════════════════════════════
def build_rich_context(df: pd.DataFrame) -> str:
    parts = [
        full_summary(df),
        "",
        detect_anomalies(df),
        "",
        analyze_time_trends(df),
        "",
        correlation_analysis(df),
    ]
    return "\n".join(parts)