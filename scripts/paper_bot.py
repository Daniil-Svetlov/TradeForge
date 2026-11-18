"""
Бумажный бот: торгует по стратегии из backtest.py на ЖИВЫХ ценах BingX,
но сделки виртуальные. Реальных денег и ключей биржи не нужно.

Раз в свечу он скачивает свежие данные, прогоняет стратегию с момента старта,
показывает позиции и результат и шлёт уведомления в Telegram.

Запуск из корня проекта:
    python scripts/paper_bot.py                  # один проход (таймфрейм 4h)
    python scripts/paper_bot.py --loop           # работает постоянно
    python scripts/paper_bot.py --timeframe 15m  # быстрый тест механики на 15-минутках
    python scripts/paper_bot.py --reset          # начать бумажную торговлю заново
    python scripts/paper_bot.py --chat-id        # узнать свой chat id в Telegram
    python scripts/paper_bot.py --test-telegram  # отправить тестовое сообщение

Настройки Telegram лежат в файле .env в корне проекта:
    TELEGRAM_BOT_TOKEN=123456:ABC...
    TELEGRAM_CHAT_ID=123456789

Ограничение: бот держит окно в 1000 свечей. На 4h это ~166 дней, на 15m ~10 дней,
на 5m ~3.5 дня. Если тест идёт дольше, бот предупредит.
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import ccxt
import pandas as pd

from backtest import (
    CONFIG,
    DATA_DIR,
    SYMBOLS,
    add_indicators,
    backtest_symbol,
    warmup_bars,
)

PROJECT_DIR = DATA_DIR.parent
FETCH_LIMIT = 1000
SUMMARY_EVERY_SEC = 24 * 3600  # сводка "бот жив" раз в сутки


# ------------------------------------------------------------------
# Telegram
# ------------------------------------------------------------------
def load_env():
    """Читает файл .env (строки вида KEY=value)."""
    env = {}
    path = PROJECT_DIR / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def telegram_credentials():
    env = load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or env.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or env.get("TELEGRAM_CHAT_ID")
    return token, chat_id


def telegram_request(token, method, params):
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = urllib.parse.urlencode(params).encode()
    request = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode())


def send_telegram(text):
    token, chat_id = telegram_credentials()
    if not token or not chat_id:
        return False
    try:
        telegram_request(token, "sendMessage", {"chat_id": chat_id, "text": text})
        return True
    except Exception as error:  # сообщения об ошибке не содержат токен
        print(f"Telegram: не удалось отправить ({error})")
        return False


def show_chat_ids():
    token, _ = telegram_credentials()
    if not token:
        print("В файле .env нет TELEGRAM_BOT_TOKEN.")
        return
    try:
        result = telegram_request(token, "getUpdates", {})
    except Exception as error:
        print(f"Не удалось связаться с Telegram ({error}). Проверь токен.")
        return
    chats = {}
    for update in result.get("result", []):
        message = update.get("message") or update.get("channel_post") or {}
        chat = message.get("chat")
        if chat:
            chats[chat["id"]] = chat.get("username") or chat.get("first_name") or ""
    if not chats:
        print("Сообщений боту пока нет. Открой своего бота в Telegram, нажми "
              "Start, отправь любое сообщение и запусти команду ещё раз.")
        return
    for chat_id, name in chats.items():
        print(f"chat id: {chat_id}  ({name})")


# ------------------------------------------------------------------
# Данные
# ------------------------------------------------------------------
def to_ccxt_symbol(name):
    base, quote = name.split("_")  # BTC_USDT -> BTC/USDT:USDT
    return f"{base}/{quote}:{quote}"


def fetch_closed_candles(exchange, name, timeframe, tf_ms):
    """Скачивает свежие свечи и выбрасывает ещё не закрытую."""
    raw = exchange.fetch_ohlcv(to_ccxt_symbol(name), timeframe, limit=FETCH_LIMIT)
    df = pd.DataFrame(
        raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df[df["timestamp"] + tf_ms <= exchange.milliseconds()]
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df = df.reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def state_paths(timeframe):
    # для 4h оставляем прежние имена файлов, для других таймфреймов - с суффиксом
    suffix = "" if timeframe == "4h" else f"_{timeframe}"
    return (
        DATA_DIR / f"paper_config{suffix}.json",
        DATA_DIR / f"paper_trades{suffix}.csv",
        DATA_DIR / f"paper_notify{suffix}.json",
    )


def load_or_create_paper_config(config_file, first_candle_ts):
    if config_file.exists():
        return json.loads(config_file.read_text())
    paper = {
        "start_ts": int(first_candle_ts),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config_file.write_text(json.dumps(paper, indent=2))
    start = pd.to_datetime(first_candle_ts, unit="ms", utc=True)
    print(f"Бумажная торговля начата с {start}")
    return paper


# ------------------------------------------------------------------
# Один проход бота
# ------------------------------------------------------------------
def run_once(exchange, timeframe):
    cfg = CONFIG
    tf_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    config_file, trades_file, notify_file = state_paths(timeframe)

    frames = {
        name: fetch_closed_candles(exchange, name, timeframe, tf_ms)
        for name in SYMBOLS
    }
    # Выравниваем монеты по общей последней закрытой свече
    last_ts = min(int(df["timestamp"].iloc[-1]) for df in frames.values())
    frames = {
        name: df[df["timestamp"] <= last_ts].reset_index(drop=True)
        for name, df in frames.items()
    }

    paper = load_or_create_paper_config(config_file, last_ts)
    start_ts = paper["start_ts"]
    capital_each = cfg["initial_capital"] / len(SYMBOLS)

    prev = json.loads(notify_file.read_text()) if notify_file.exists() else {}
    prev_open = prev.get("open", {})
    prev_plan = prev.get("plan", {})
    new_open, new_plan = {}, {}
    events, status_lines = [], []

    last_time = pd.to_datetime(last_ts, unit="ms", utc=True)
    header = f"Последняя закрытая {timeframe}-свеча: {last_time:%Y-%m-%d %H:%M} UTC"

    total_equity = 0.0
    all_trades = []
    for name, df in frames.items():
        d = add_indicators(df, cfg)
        start_i = int((d["timestamp"] >= start_ts).idxmax())
        if start_i < warmup_bars(cfg):
            status_lines.append(
                f"! {name}: тест идёт дольше окна истории, результат неточный"
            )
            start_i = warmup_bars(cfg)

        eq, trades, st = backtest_symbol(
            name, d, cfg, capital_each, start=start_i, close_at_end=False
        )
        total_equity += eq.iloc[-1]
        all_trades += trades
        price = st["last_close"]

        if st["pos"] != 0:
            side = "ЛОНГ" if st["pos"] == 1 else "ШОРТ"
            entry_time = pd.Timestamp(st["entry_time"])
            status_lines.append(
                f"{name}: {side} с {entry_time:%Y-%m-%d %H:%M}, "
                f"вход {st['entry_price']:.2f}, стоп {st['stop']:.2f}, "
                f"сейчас {price:.2f}, нереализ. {st['unrealized']:+.1f} USDT"
            )
            entry_key = str(entry_time)
            new_open[name] = entry_key
            if prev_open.get(name) != entry_key:
                events.append(
                    f"🟢 ОТКРЫТ {side} {name} (бумажный)\n"
                    f"Вход {st['entry_price']:.2f}, стоп {st['stop']:.2f}, "
                    f"размер {st['size']:.4f}"
                )
        else:
            status_lines.append(f"{name}: позиции нет, цена {price:.2f}")

        plan = {"long": "ОТКРЫТЬ ЛОНГ", "short": "ОТКРЫТЬ ШОРТ",
                "close": "ЗАКРЫТЬ ПОЗИЦИЮ"}.get(st["pending"])
        if plan:
            status_lines.append(f"   -> ПЛАН на открытие следующей свечи: {plan}")
            plan_key = f"{st['pending']}@{last_ts}"
            new_plan[name] = plan_key
            if prev_plan.get(name) != plan_key:
                events.append(
                    f"📈 СИГНАЛ {name}: {plan} на открытии следующей свечи"
                )

    # Закрытые сделки: новые показываем и отправляем
    old_count = len(pd.read_csv(trades_file)) if trades_file.exists() else 0
    trades_df = pd.DataFrame(all_trades)
    new_trade_lines = []
    if not trades_df.empty:
        trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)
        trades_df.to_csv(trades_file, index=False)
        for _, t in trades_df.iloc[old_count:].iterrows():
            mark = "✅" if t["net_pnl"] > 0 else "🔻"
            line = (f"{t['symbol']} {t['side']}: {t['net_pnl']:+.1f} USDT "
                    f"({t['reason']}), закрыта {pd.Timestamp(t['exit_time']):%Y-%m-%d %H:%M}")
            new_trade_lines.append(line)
            events.append(f"{mark} ЗАКРЫТА сделка {line}")

    start_cap = cfg["initial_capital"]
    equity_line = (f"Бумажный капитал: {total_equity:,.0f} USDT "
                   f"({total_equity / start_cap - 1:+.2%} от {start_cap:,.0f}), "
                   f"закрытых сделок: {len(trades_df)}")

    # Вывод в терминал
    print("=" * 66)
    print(header)
    print("=" * 66)
    print("\n".join(status_lines))
    if new_trade_lines:
        print("\nНовые закрытые сделки:")
        print("\n".join(f"  {line}" for line in new_trade_lines))
    print(f"\n{equity_line}\n")

    # Telegram: события сразу, сводка раз в сутки
    messages = list(events)
    summary_due = time.time() - prev.get("last_summary", 0) > SUMMARY_EVERY_SEC
    if summary_due:
        messages.append(
            f"📊 Статус бота ({timeframe}, бумажный)\n{header}\n\n"
            + "\n".join(status_lines) + f"\n\n{equity_line}"
        )
    if messages:
        token, chat_id = telegram_credentials()
        if token and chat_id:
            send_telegram("\n\n".join(messages))
        else:
            print("(Telegram не настроен: заполни .env, чтобы получать уведомления)\n")

    notify_file.write_text(json.dumps({
        "open": new_open,
        "plan": new_plan,
        "last_summary": time.time() if summary_due else prev.get("last_summary", 0),
    }))


def sleep_until_next_candle(tf_seconds):
    now = time.time()
    next_boundary = (int(now // tf_seconds) + 1) * tf_seconds + 30
    wait = max(1, next_boundary - now)
    resume = datetime.fromtimestamp(next_boundary, tz=timezone.utc)
    print(f"Сплю до {resume:%Y-%m-%d %H:%M:%S} UTC...\n")
    time.sleep(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="работать постоянно")
    parser.add_argument("--reset", action="store_true", help="начать заново")
    parser.add_argument("--timeframe", default="4h", help="например 4h, 15m, 5m")
    parser.add_argument("--chat-id", action="store_true", help="показать chat id")
    parser.add_argument("--test-telegram", action="store_true",
                        help="отправить тестовое сообщение")
    args = parser.parse_args()

    if args.chat_id:
        show_chat_ids()
        return
    if args.test_telegram:
        ok = send_telegram("✅ TradeForge: Telegram подключён, уведомления работают.")
        print("Сообщение отправлено." if ok else
              "Не получилось отправить. Проверь токен и chat id в .env.")
        return

    if args.reset:
        for f in state_paths(args.timeframe):
            f.unlink(missing_ok=True)
        print(f"Бумажная торговля ({args.timeframe}) сброшена.")

    exchange = ccxt.bingx({"options": {"defaultType": "swap"}})

    if not args.loop:
        run_once(exchange, args.timeframe)
        return

    tf_seconds = exchange.parse_timeframe(args.timeframe)
    had_error = False
    while True:
        try:
            run_once(exchange, args.timeframe)
            if had_error:
                send_telegram("✅ Бот снова работает после ошибки.")
                had_error = False
        except Exception as error:  # сеть, биржа: не падаем, пробуем в следующий раз
            print(f"Ошибка: {error}")
            if not had_error:
                send_telegram(f"⚠️ Ошибка бота: {error}")
                had_error = True
        sleep_until_next_candle(tf_seconds)


if __name__ == "__main__":
    main()
