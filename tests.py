"""اختبارات صحة: لا تسريب مستقبل + حسابات الربح/العمولة + سقف الرافعة."""
from dataclasses import replace

import numpy as np
import pandas as pd

from data import synthetic
from engine import Engine
from params import Params
from strategy import generate_signals

TS = pd.Timestamp("2026-01-01", tz="UTC")


def test_no_lookahead():
    df = synthetic(days=40, tf_min=15, seed=5)
    p = replace(Params(), use_pullback=True, use_meanrev=True)
    full, *_ = generate_signals(df, p)
    for k in (1500, 2500, 3500):
        part, *_ = generate_signals(df.iloc[:k], p)
        assert np.array_equal(full[:k - 1], part[:k - 1]), f"تسريب مستقبل عند {k}"
    print("OK  لا يوجد تسريب للمستقبل في الإشارات")


def _eng():
    p = replace(Params(), slippage=0.0, risk_pct=0.01, max_leverage=5.0, sl_atr=1.0, tp_atr=2.0)
    return Engine(p, 10_000.0)


def test_tp_math():
    e = _eng()
    e.step(TS, 100, 121, 99, 120, (1, 10.0, 1))
    t = e.trades[0]
    assert t["reason"] == "tp" and abs(t["pnl"] - 199.36) < 1e-6, t
    print("OK  حساب الهدف والعمولات (maker)")


def test_sl_math_and_conservative_order():
    e = _eng()
    e.step(TS, 100, 121, 89, 100, (1, 10.0, 1))   # الشمعة لمست الستوب والهدف معاً
    t = e.trades[0]
    assert t["reason"] == "sl" and abs(t["pnl"] + 100.76) < 1e-6, t
    print("OK  الستوب يُحتسب أولاً عند لمس الطرفين (محافظ)")


def test_short_and_leverage_cap():
    p = replace(Params(), slippage=0.0, risk_pct=0.5, max_leverage=2.0)
    e = Engine(p, 10_000.0)
    e.step(TS, 100, 101, 99, 100, (-1, 1.0, 1))
    assert e.pos["qty"] * 100 <= 20_000 + 1e-6, e.pos
    print("OK  سقف الرافعة يعمل")


def test_daily_halt_and_kill():
    p = replace(Params(), slippage=0.0, risk_pct=0.02, daily_loss_pct=0.03, max_leverage=50)
    e = Engine(p, 10_000.0)
    for i in range(4):
        e.step(TS + pd.Timedelta(minutes=15 * i), 100, 101, 80, 85, (1, 5.0, 1))
    assert e.day_halt and len(e.trades) < 4, (e.day_halt, len(e.trades))
    print("OK  إيقاف الخسارة اليومية يعمل")


if __name__ == "__main__":
    test_no_lookahead(); test_tp_math(); test_sl_math_and_conservative_order()
    test_short_and_leverage_cap(); test_daily_halt_and_kill()
    print("كل الاختبارات نجحت")
