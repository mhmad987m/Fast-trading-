"""
توليد الإشارات. ثلاث استراتيجيات سكالبينج تعمل حسب حالة السوق:
  1) Pullback : ارتداد من EMA داخل ترند قوي + اتجاه الإطار الأكبر
  2) Breakout : اختراق نطاق صغير بحجم تداول مرتفع وشمعة قوية
  3) MeanRev  : ارتداد من خارج بولنجر في سوق عرضي (ADX منخفض) مع انحراف عن VWAP
الإشارة تُحسب عند إغلاق الشمعة ويُنفَّذ الدخول عند افتتاح الشمعة التالية.
"""
import numpy as np
import pandas as pd

from indicators import adx, atr, bollinger, ema, htf_bias, rsi, session_vwap
from params import Params

SETUP_NAMES = {1: "pullback", 2: "breakout", 3: "meanrev"}
WARMUP = 200


def generate_signals(df: pd.DataFrame, p: Params):
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    a = atr(df, p.atr_n)
    r = rsi(c, p.rsi_n)
    ad = adx(df, p.adx_n)
    ef, em, es = ema(c, p.ema_f), ema(c, p.ema_m), ema(c, p.ema_s)
    _, bb_up, bb_lo = bollinger(c, p.bb_n, p.bb_k)
    vr = v / v.rolling(p.vol_n).mean()
    vw = session_vwap(df)
    dev = c - vw
    zv = dev / dev.rolling(96).std()
    hh = h.rolling(p.brk_n).max().shift(1)
    ll = l.rolling(p.brk_n).min().shift(1)
    body = (c - o) / (h - l).replace(0, np.nan)

    slope = es.diff(5)
    up_t = (ef > em) & (em > es) & (slope > 0)
    dn_t = (ef < em) & (em < es) & (slope < 0)
    trending = ad > p.adx_trend
    ranging = ad < p.adx_range

    if p.use_htf:
        hu, hd = htf_bias(df, p.htf_rule, p.htf_f, p.htf_s)
    else:
        hu = hd = pd.Series(True, index=df.index)

    # 1) Pullback
    turn_up = (r > r.shift(1)) & (r.rolling(4).min() < p.rsi_pb)
    turn_dn = (r < r.shift(1)) & (r.rolling(4).max() > 100 - p.rsi_pb)
    pb_l = up_t & (ad > p.adx_trend * 0.8) & hu & (l <= em + 0.25 * a) & (c > ef) & (c > o) & turn_up
    pb_s = dn_t & (ad > p.adx_trend * 0.8) & hd & (h >= em - 0.25 * a) & (c < ef) & (c < o) & turn_dn
    # 2) Breakout
    brk_l = up_t & (ad > p.adx_trend * 0.8) & hu & (c > hh) & (vr > p.vol_spike) & (body > 0.6)
    brk_s = dn_t & (ad > p.adx_trend * 0.8) & hd & (c < ll) & (vr > p.vol_spike) & (body < -0.6)
    # 3) Mean reversion
    calm = ad < p.adx_trend                      # سوق غير متجه بقوة
    mr_l = calm & (l < bb_lo) & (c > bb_lo) & (c > o) & (zv.rolling(3).min() < -p.z_vwap * 0.6) & (r.rolling(3).min() < p.rsi_os + 8)
    mr_s = calm & (h > bb_up) & (c < bb_up) & (c < o) & (zv.rolling(3).max() > p.z_vwap * 0.6) & (r.rolling(3).max() > 100 - p.rsi_os - 8)

    f = lambda x: x.fillna(False).to_numpy(dtype=bool, copy=True)
    pb_l, pb_s, brk_l, brk_s, mr_l, mr_s = map(f, (pb_l, pb_s, brk_l, brk_s, mr_l, mr_s))
    pb_l &= p.use_pullback; pb_s &= p.use_pullback
    brk_l &= p.use_breakout; brk_s &= p.use_breakout
    mr_l &= p.use_meanrev; mr_s &= p.use_meanrev

    long_ = pb_l | brk_l | mr_l
    short_ = (pb_s | brk_s | mr_s) & p.allow_short
    setup = np.where(pb_l | pb_s, 1, np.where(brk_l | brk_s, 2, 3)).astype(np.int8)

    # فلتر التكلفة: الهدف لازم يغطي التكاليف بهامش
    edge_ok = (p.tp_atr * a / c).to_numpy(copy=True) >= p.min_edge_mult * p.cost_frac
    # فلتر الساعات
    if p.hours:
        hr = df.index.hour
        edge_ok &= (hr >= p.hours[0]) & (hr < p.hours[1])

    valid = edge_ok & ~np.isnan(a.to_numpy())
    valid[:WARMUP] = False
    both = long_ & short_
    sig = np.zeros(len(df), dtype=np.int8)
    sig[long_ & valid & ~both] = 1
    sig[short_ & valid & ~both] = -1
    return sig, a.to_numpy(), setup
