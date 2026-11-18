"""
Бэктест трендовой стратегии "пробой канала Дончиана" на данных BingX.

Что делает стратегия (4h свечи):
  - ЛОНГ:  закрытие выше максимума последних 55 свечей и выше EMA200
  - ШОРТ:  закрытие ниже минимума последних 55 свечей и ниже EMA200
  - ВЫХОД: закрытие пробило противоположный 20-свечной канал
           или сработал стоп-лосс (3 ATR от цены входа)
  - РАЗМЕР: рискуем 1% капитала монеты на сделку, плечо не выше 3x

Честность бэктеста:
  - сигнал считается по ЗАКРЫТОЙ свече, вход происходит по ОТКРЫТИЮ следующей
    (нет подглядывания в будущее)
  - учитываются комиссия и проскальзывание на входе и выходе
  - стоп проверяется внутри свечи по high/low
  - не учитывается: фандинг, ликвидации, частичные исполнения

Запуск из корня проекта:
    python scripts/backtest.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # рисуем в файл, без окна
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]
TIMEFRAME = "4h"
BARS_PER_YEAR = 6 * 365  # 6 четырёхчасовых свечей в сутки

# ---- Настройки стратегии: можно менять и смотреть, что будет ----
CONFIG = {
    "initial_capital": 10_000.0,  # общий стартовый капитал, делится поровну на монеты
    "entry_period": 55,           # канал для входа (свечей)
    "exit_period": 20,            # канал для выхода (свечей)
    "trend_ema": 200,             # фильтр тренда
    "atr_period": 14,
    "atr_stop_mult": 3.0,         # стоп = 3 ATR от входа
    "risk_per_trade": 0.01,       # рискуем 1% капитала на сделку
    "max_leverage": 3.0,          # потолок плеча
    "fee": 0.0005,                # комиссия 0.05% за сторону (проверь тариф BingX!)
    "slippage": 0.0002,           # проскальзывание 0.02% за сторону
    "allow_short": True,
}


# ------------------------------------------------------------------
# Данные и индикаторы
# ------------------------------------------------------------------
def load(name):
    df = pd.read_csv(DATA_DIR / f"{name}_{TIMEFRAME}.csv", parse_dates=["datetime"])
    df = df.iloc[:-1]  # последняя свеча ещё не закрыта - выбрасываем
    return df.reset_index(drop=True)


def add_indicators(df, cfg):
    df = df.copy()
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    p = cfg["atr_period"]
    df["atr"] = true_range.ewm(alpha=1 / p, adjust=False, min_periods=p).mean()
    e = cfg["trend_ema"]
    df["ema"] = df["close"].ewm(span=e, adjust=False, min_periods=e).mean()

    n, m = cfg["entry_period"], cfg["exit_period"]
    # shift(1): канал строится по ПРЕДЫДУЩИМ свечам, текущая в него не входит
    df["entry_high"] = df["high"].rolling(n).max().shift(1)
    df["entry_low"] = df["low"].rolling(n).min().shift(1)
    df["exit_high"] = df["high"].rolling(m).max().shift(1)
    df["exit_low"] = df["low"].rolling(m).min().shift(1)
    return df


def warmup_bars(cfg):
    return max(cfg["trend_ema"], cfg["entry_period"], cfg["atr_period"]) + 1


# ------------------------------------------------------------------
# Движок бэктеста для одной монеты
# ------------------------------------------------------------------
def backtest_symbol(symbol, df, cfg, capital, start=None, close_at_end=True):
    """start - с какой свечи начинать торговать (по умолчанию после прогрева).
    close_at_end=False нужен боту: открытая позиция остаётся открытой."""
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    ema = df["ema"].to_numpy()
    entry_hi = df["entry_high"].to_numpy()
    entry_lo = df["entry_low"].to_numpy()
    exit_hi = df["exit_high"].to_numpy()
    exit_lo = df["exit_low"].to_numpy()
    times = df["datetime"].to_numpy()

    n = len(df)
    start = warmup_bars(cfg) if start is None else start
    fee, slip = cfg["fee"], cfg["slippage"]

    cash = capital
    pos = 0            # 1 = лонг, -1 = шорт, 0 = нет позиции
    entry_price = 0.0
    entry_fee = 0.0
    entry_i = 0
    size = 0.0         # размер позиции в монетах
    stop = 0.0
    pending = None     # приказ, который исполним на открытии следующей свечи
    pending_atr = 0.0

    equity = np.zeros(n)
    trades = []

    def close_position(i, price, reason):
        nonlocal cash, pos, size
        pnl = pos * (price - entry_price) * size
        exit_fee = size * price * fee
        cash += pnl - exit_fee
        net = pnl - entry_fee - exit_fee
        trades.append(
            {
                "symbol": symbol,
                "side": "long" if pos == 1 else "short",
                "entry_time": times[entry_i],
                "exit_time": times[i],
                "entry_price": entry_price,
                "exit_price": price,
                "size": size,
                "net_pnl": net,
                "return_on_notional": net / (size * entry_price),
                "reason": reason,
            }
        )
        pos = 0
        size = 0.0

    for i in range(n):
        if i < start:
            equity[i] = cash
            continue

        # 1) исполняем приказ, поставленный на закрытии прошлой свечи
        if pending is not None:
            if pending == "close" and pos != 0:
                px = o[i] * (1 - slip * pos)
                close_position(i, px, "signal")
            elif pending in ("long", "short") and pos == 0:
                side = 1 if pending == "long" else -1
                px = o[i] * (1 + slip * side)
                stop_dist = pending_atr * cfg["atr_stop_mult"]
                if stop_dist > 0:
                    by_risk = cash * cfg["risk_per_trade"] / stop_dist
                    by_leverage = cash * cfg["max_leverage"] / px
                    size = min(by_risk, by_leverage)
                    pos = side
                    entry_price = px
                    entry_i = i
                    stop = px - side * stop_dist
                    entry_fee = size * px * fee
                    cash -= entry_fee
            pending = None

        # 2) проверяем стоп внутри свечи
        if pos == 1 and lo[i] <= stop:
            close_position(i, min(stop, o[i]) * (1 - slip), "stop")
        elif pos == -1 and h[i] >= stop:
            close_position(i, max(stop, o[i]) * (1 + slip), "stop")

        # 3) считаем капитал по цене закрытия
        unrealized = pos * (c[i] - entry_price) * size if pos != 0 else 0.0
        equity[i] = cash + unrealized

        # 4) смотрим сигнал по закрытию свечи (исполнится на следующей)
        if pos == 0:
            if c[i] > entry_hi[i] and c[i] > ema[i]:
                pending, pending_atr = "long", atr[i]
            elif cfg["allow_short"] and c[i] < entry_lo[i] and c[i] < ema[i]:
                pending, pending_atr = "short", atr[i]
        elif pos == 1 and c[i] < exit_lo[i]:
            pending = "close"
        elif pos == -1 and c[i] > exit_hi[i]:
            pending = "close"

    state = {
        "pos": pos,
        "entry_price": entry_price,
        "entry_time": times[entry_i] if pos != 0 else None,
        "size": size,
        "stop": stop,
        "unrealized": pos * (c[-1] - entry_price) * size if pos != 0 else 0.0,
        "pending": pending,
        "last_close": c[-1],
    }

    # в конце бэктеста закрываем открытую позицию по последней цене
    if close_at_end and pos != 0:
        close_position(n - 1, c[-1], "end")
        equity[-1] = cash

    return pd.Series(equity, index=df["datetime"]), trades, state


# ------------------------------------------------------------------
# Портфель: несколько монет, капитал делится поровну
# ------------------------------------------------------------------
def run_backtest(frames, cfg):
    capital_each = cfg["initial_capital"] / len(frames)
    curves, all_trades = {}, []
    for name, df in frames.items():
        eq, trades, _ = backtest_symbol(
            name, add_indicators(df, cfg), cfg, capital_each
        )
        curves[name] = eq
        all_trades += trades

    w = warmup_bars(cfg)
    equity = pd.concat(curves, axis=1, join="inner").sum(axis=1).iloc[w:]
    return equity, pd.DataFrame(all_trades)


def buy_and_hold(frames, cfg, index):
    """Для сравнения: поровну купили монеты в начале и держим."""
    capital_each = cfg["initial_capital"] / len(frames)
    parts = []
    for df in frames.values():
        close = df.set_index("datetime")["close"].reindex(index)
        parts.append(close / close.iloc[0] * capital_each)
    return pd.concat(parts, axis=1).sum(axis=1)


# ------------------------------------------------------------------
# Метрики
# ------------------------------------------------------------------
def equity_stats(eq):
    ret = eq.pct_change().dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1
    years = len(eq) / BARS_PER_YEAR
    growth = eq.iloc[-1] / eq.iloc[0]
    cagr = growth ** (1 / years) - 1 if growth > 0 and years > 0 else float("nan")
    max_dd = (eq / eq.cummax() - 1).min()
    sharpe = ret.mean() / ret.std() * np.sqrt(BARS_PER_YEAR) if ret.std() > 0 else 0.0
    return {"total": total, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe}


def trade_stats(trades):
    if trades.empty:
        return None
    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]
    gross_win = wins["net_pnl"].sum()
    gross_loss = -losses["net_pnl"].sum()
    return {
        "n": len(trades),
        "win_rate": len(wins) / len(trades),
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "avg_win": wins["net_pnl"].mean() if len(wins) else 0.0,
        "avg_loss": losses["net_pnl"].mean() if len(losses) else 0.0,
        "longs": int((trades["side"] == "long").sum()),
        "shorts": int((trades["side"] == "short").sum()),
    }


def fmt_stats(s):
    return (
        f"доходность {s['total']:+.1%} | в год {s['cagr']:+.1%} | "
        f"макс. просадка {s['max_dd']:.1%} | Sharpe {s['sharpe']:.2f}"
    )


# ------------------------------------------------------------------
# Отчёт
# ------------------------------------------------------------------
def main():
    frames = {name: load(name) for name in SYMBOLS}
    cfg = CONFIG

    equity, trades = run_backtest(frames, cfg)
    bench = buy_and_hold(frames, cfg, equity.index)

    print("=" * 70)
    print("СТРАТЕГИЯ: пробой канала Дончиана + фильтр EMA, лонг и шорт")
    print(f"Период: {equity.index[0]:%Y-%m-%d} -> {equity.index[-1]:%Y-%m-%d}")
    print(f"Старт: {cfg['initial_capital']:,.0f} USDT, монеты: {', '.join(SYMBOLS)}")
    print("=" * 70)

    print("\nИТОГ ЗА ВЕСЬ ПЕРИОД")
    print(f"  Стратегия:     {fmt_stats(equity_stats(equity))}")
    print(f"  Купил и держу: {fmt_stats(equity_stats(bench))}")
    print(f"  Конечный капитал: {equity.iloc[-1]:,.0f} USDT")

    # Две половины периода: стабильна ли стратегия, или повезло в одном куске
    mid = len(equity) // 2
    print("\nПО ПОЛОВИНАМ ПЕРИОДА (устойчивость)")
    print(f"  1-я ({equity.index[0]:%Y-%m} .. {equity.index[mid - 1]:%Y-%m}): "
          f"{fmt_stats(equity_stats(equity.iloc[:mid]))}")
    print(f"  2-я ({equity.index[mid]:%Y-%m} .. {equity.index[-1]:%Y-%m}): "
          f"{fmt_stats(equity_stats(equity.iloc[mid:]))}")

    ts = trade_stats(trades)
    print("\nСДЕЛКИ")
    if ts is None:
        print("  Сделок не было")
    else:
        print(f"  Всего: {ts['n']} (лонг {ts['longs']}, шорт {ts['shorts']})")
        print(f"  Доля прибыльных: {ts['win_rate']:.1%}")
        print(f"  Profit factor: {ts['profit_factor']:.2f} (выше 1 = в плюсе)")
        print(f"  Средняя прибыль: {ts['avg_win']:+.1f} USDT, "
              f"средний убыток: {ts['avg_loss']:+.1f} USDT")
        print("  По монетам (чистый результат, USDT):")
        for sym, val in trades.groupby("symbol")["net_pnl"].sum().items():
            print(f"    {sym}: {val:+,.0f}")

    # Чувствительность: если результат держится только на одном значении - это подгонка
    print("\nЧУВСТВИТЕЛЬНОСТЬ К ДЛИНЕ КАНАЛА ВХОДА")
    for period in (20, 40, 55, 80, 120):
        test_cfg = {**cfg, "entry_period": period}
        eq, tr = run_backtest(frames, test_cfg)
        s = equity_stats(eq)
        print(f"  канал {period:>3}: {s['total']:+8.1%} | "
              f"просадка {s['max_dd']:6.1%} | сделок {len(tr)}")

    # Сохраняем сделки и график
    if not trades.empty:
        trades.to_csv(DATA_DIR / "trades.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(equity.index, equity, label="Стратегия", linewidth=1.3)
    axes[0].plot(bench.index, bench, label="Купил и держу", linewidth=1.0, alpha=0.7)
    axes[0].set_title("Капитал, USDT")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    drawdown = equity / equity.cummax() - 1
    axes[1].fill_between(drawdown.index, drawdown * 100, 0, color="tab:red", alpha=0.4)
    axes[1].set_title("Просадка стратегии, %")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = DATA_DIR / "backtest_result.png"
    fig.savefig(out, dpi=120)
    print(f"\nГрафик: {out}")
    print(f"Сделки: {DATA_DIR / 'trades.csv'}")


if __name__ == "__main__":
    main()
