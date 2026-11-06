"""مؤشرات فنية (سببية فقط، بدون أي نظر للمستقبل)."""
import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return mid, mid + k * sd, mid - k * sd


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP يعاد حسابه كل يوم UTC."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.normalize()
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return (pv / vv).fillna(df["close"])


def htf_bias(df: pd.DataFrame, rule: str, f: int, s: int):
    """اتجاه الإطار الأكبر باستخدام الشموع المكتملة فقط (shift(1) = لا تسريب مستقبل)."""
    hc = df["close"].resample(rule).last().dropna()
    hf, hs = ema(hc, f), ema(hc, s)
    up = ((hf > hs) & (hc > hs)).shift(1, fill_value=False)
    dn = ((hf < hs) & (hc < hs)).shift(1, fill_value=False)
    up = up.reindex(df.index, method="ffill").fillna(False).astype(bool)
    dn = dn.reindex(df.index, method="ffill").fillna(False).astype(bool)
    return up, dn
