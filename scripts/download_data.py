"""
Скачивает историю свечей с BingX (фьючерсы) и сохраняет в папку data/.
Качает от сегодняшнего дня назад в прошлое, пока не дойдёт до START
или пока биржа не перестанет отдавать данные.

Запуск из корня проекта:
    python scripts/download_data.py
"""

from pathlib import Path

import ccxt
import pandas as pd

# ---- Настройки: можно менять ----
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
TIMEFRAME = "4h"
START = "2022-01-01T00:00:00Z"  # как далеко в прошлое хотим дойти
LIMIT = 1000                    # сколько свечей просить за один запрос
# ----------------------------------

# Папка data рядом с папкой scripts
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_symbol(exchange, symbol):
    """Качает историю одной монеты порциями (от новых к старым)."""
    start_ms = exchange.parse8601(START)
    end_ms = exchange.milliseconds()
    rows = []
    oldest_seen = None

    while True:
        batch = exchange.fetch_ohlcv(
            symbol, TIMEFRAME, limit=LIMIT, params={"until": end_ms}
        )
        if not batch:
            print("  биржа вернула пустой ответ, останавливаюсь")
            break

        oldest = min(c[0] for c in batch)
        # Защита от зацикливания: если дальше в прошлое не продвинулись
        if oldest_seen is not None and oldest >= oldest_seen:
            print("  новых (более старых) свечей нет, останавливаюсь")
            break

        rows.extend(batch)
        oldest_seen = oldest
        end_ms = oldest - 1  # следующая порция - всё, что старше

        print(
            f"  {symbol}: скачано {len(rows)} свечей, "
            f"дошли до {exchange.iso8601(oldest)[:10]}"
        )

        if oldest <= start_ms:
            break

    df = pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df = df[df["timestamp"] >= start_ms]
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def main():
    exchange = ccxt.bingx({"options": {"defaultType": "swap"}})

    for symbol in SYMBOLS:
        print(f"Качаю {symbol}...")
        df = download_symbol(exchange, symbol)

        if df.empty:
            print(f"Для {symbol} данных не получено, пропускаю\n")
            continue

        # BTC/USDT:USDT -> BTC_USDT_4h.csv
        name = symbol.split(":")[0].replace("/", "_")
        path = DATA_DIR / f"{name}_{TIMEFRAME}.csv"
        df.to_csv(path, index=False)

        print(
            f"Готово: {len(df)} свечей, "
            f"с {df['datetime'].iloc[0]} по {df['datetime'].iloc[-1]}"
        )
        print(f"Сохранено в {path}\n")


if __name__ == "__main__":
    main()
