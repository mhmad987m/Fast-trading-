"""مصادر البيانات: CSV، باينانس (بيانات سوق عامة بدون مفاتيح)، ومولّد بيانات اصطناعية."""
import json
import time
import urllib.request

import numpy as np
import pandas as pd

TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
BASE = "https://data-api.binance.vision/api/v3/klines"
COLS = ["open", "high", "low", "close", "volume"]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    tcol = next(c for c in ("timestamp", "time", "open_time", "datetime", "date") if c in df.columns)
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t):
        t = pd.to_datetime(t, unit="ms" if t.iloc[0] > 1e11 else "s", utc=True)
    else:
        t = pd.to_datetime(t, utc=True)
    out = df[COLS].astype(float)
    out.index = t
    out = out[~out.index.duplicated()].sort_index()
    out.index.name = "time"
    return out


def _fetch(symbol, tf, start_ms=None, limit=1000):
    url = f"{BASE}?symbol={symbol}&interval={tf}&limit={limit}"
    if start_ms:
        url += f"&startTime={start_ms}"
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())


def _to_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["time"] + COLS
    df.index = pd.to_datetime(df.pop("time"), unit="ms", utc=True)
    return df.astype(float)


def fetch_klines(symbol="BTCUSDT", tf="5m", limit=800) -> pd.DataFrame:
    return _to_df(_fetch(symbol, tf, limit=min(limit, 1000)))


def fetch_history(symbol="BTCUSDT", tf="5m", days=30) -> pd.DataFrame:
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    parts = []
    while start < end:
        rows = _fetch(symbol, tf, start_ms=start)
        if not rows:
            break
        parts.append(_to_df(rows))
        start = int(rows[-1][0]) + TF_MIN[tf] * 60_000
        if len(rows) < 1000:
            break
    df = pd.concat(parts)
    return df[~df.index.duplicated()].sort_index()


def synthetic(days=60, tf_min=5, seed=7, structured=True, start_price=60_000.0,
              start="2026-01-01") -> pd.DataFrame:
    """
    structured=True : أنظمة سوق متبدلة (ترند صاعد/هابط، عرضي مع ارتداد، تذبذب عالٍ).
    structured=False: مشي عشوائي صرف بلا أي أفضلية => أي ربح فيه = وهم (اختبار سلامة).
    """
    rng = np.random.default_rng(seed)
    n = days * 1440 // tf_min
    m = 6
    sig0 = 0.0012 * np.sqrt(tf_min / 5)
    regime = rng.integers(0, 4)
    logvol = 0.0
    x = np.log(start_price)
    anchor = x
    r = np.empty(n)
    sg = np.empty(n)
    for t in range(n):
        if rng.random() < 1 / 120:
            regime = rng.integers(0, 4) if structured else 3
            anchor = x
        logvol = 0.97 * logvol + 0.12 * rng.standard_normal()
        vmult = np.exp(0.35 * logvol)
        if structured:
            vmult *= (1.0, 1.0, 0.6, 1.7)[regime]
        s = sig0 * vmult
        mu = 0.0
        if structured:
            if regime == 0:
                mu = 0.15 * s
            elif regime == 1:
                mu = -0.15 * s
            elif regime == 2:
                mu = 0.12 * (anchor - x)
        rt = mu + s * rng.standard_normal()
        r[t], sg[t] = rt, s
        x += rt
    logp = np.log(start_price) + np.cumsum(r)
    prev = np.concatenate([[np.log(start_price)], logp[:-1]])
    w = np.cumsum(rng.normal(0, 1 / np.sqrt(m), (n, m)), axis=1)
    k = np.arange(1, m + 1) / m
    bridge = w - k * w[:, -1:]
    path = prev[:, None] + r[:, None] * k + sg[:, None] * bridge
    allp = np.exp(np.concatenate([prev[:, None], path], axis=1))
    o, c = allp[:, 0], allp[:, -1]
    hi, lo = allp.max(axis=1), allp.min(axis=1)
    idx = pd.date_range(start, periods=n, freq=f"{tf_min}min", tz="UTC")
    hours = idx.hour.to_numpy()
    season = 1 + 0.6 * np.exp(-((hours - 14) % 24) ** 2 / 18)
    vol = rng.lognormal(0, 0.35, n) * season * (1 + 120 * np.abs(r) / sig0 / 100 * 3 + 3 * (sg / sig0 > 1.4))
    vol *= 50
    return pd.DataFrame(dict(open=o, high=hi, low=lo, close=c, volume=vol), index=idx)
