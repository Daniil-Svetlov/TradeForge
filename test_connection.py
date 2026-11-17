import ccxt

exchange = ccxt.bingx({"options": {"defaultType": "swap"}})
candles = exchange.fetch_ohlcv("BTC/USDT:USDT", "4h", limit=10)

for candle in candles:
    print(candle)  # [время, open, high, low, close, объём]
