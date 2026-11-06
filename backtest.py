"""باكتيست + مقاييس الأداء + مونت كارلو."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine import Engine
from params import Params
from strategy import generate_signals, SETUP_NAMES


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict
    final_equity: float


def run_backtest(df: pd.DataFrame, p: Params, equity0: float = 10_000.0, start: int = 0) -> Result:
    sig, atr_, setup = generate_signals(df, p)
    idx = df.index
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(df)
    eng = Engine(p, equity0)
    eq = np.full(n, equity0, dtype=float)
    pending = None
    for t in range(start, n):
        eq[t] = eng.step(idx[t], o[t], h[t], l[t], c[t], pending)
        pending = (int(sig[t]), float(atr_[t]), int(setup[t])) if (sig[t] != 0 and eng.pos is None) else None
    eng.force_close(idx[-1], c[-1])
    eq[-1] = eng.equity
    trades = pd.DataFrame(eng.trades)
    equity = pd.Series(eq, index=idx)
    return Result(trades, equity, compute_metrics(trades, equity.iloc[start:], equity0), eng.equity)


def compute_metrics(trades: pd.DataFrame, equity: pd.Series, equity0: float) -> dict:
    m = dict(n_trades=0, win_rate=0.0, profit_factor=0.0, expectancy_R=0.0, net_pct=0.0,
             max_dd_pct=0.0, sharpe=0.0, tstat=0.0, fees=0.0, trades_per_day=0.0,
             avg_bars=0.0, final_equity=float(equity.iloc[-1]))
    peak = np.maximum.accumulate(equity.to_numpy())
    m["max_dd_pct"] = float(((equity.to_numpy() - peak) / peak).min() * 100)
    m["net_pct"] = float((equity.iloc[-1] / equity0 - 1) * 100)
    if trades.empty:
        return m
    w, lo = trades[trades.pnl > 0].pnl, trades[trades.pnl <= 0].pnl
    m["n_trades"] = len(trades)
    m["win_rate"] = float(len(w) / len(trades) * 100)
    m["profit_factor"] = float(w.sum() / abs(lo.sum())) if lo.sum() != 0 else float("inf")
    m["expectancy_R"] = float(trades.R.mean())
    m["fees"] = float(trades.fees.sum())
    m["avg_bars"] = float(trades.bars.mean())
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    m["trades_per_day"] = len(trades) / days
    if len(trades) > 2 and trades.ret.std() > 0:
        m["tstat"] = float(trades.ret.mean() / trades.ret.std() * np.sqrt(len(trades)))
    d = equity.resample("1D").last().dropna().pct_change().dropna()
    if len(d) > 2 and d.std() > 0:
        m["sharpe"] = float(d.mean() / d.std() * np.sqrt(365))
    return m


def breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["setup"] = t["setup"].map(SETUP_NAMES)
    t["side"] = t["dir"].map({1: "long", -1: "short"})
    g = t.groupby("setup").agg(n=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean() * 100),
                               avgR=("R", "mean"), pnl=("pnl", "sum"))
    return g.round(2)


def monte_carlo(trades: pd.DataFrame, n_sims: int = 3000, seed: int = 1, ruin_dd: float = 0.30):
    """إعادة خلط الصفقات لتقدير توزيع العائد وأقصى تراجع واحتمال الانهيار."""
    if len(trades) < 20:
        return None
    rng = np.random.default_rng(seed)
    r = trades.ret.to_numpy()
    k = len(r)
    sims = rng.choice(r, size=(n_sims, k), replace=True)
    curves = np.cumprod(1 + sims, axis=1)
    peaks = np.maximum.accumulate(curves, axis=1)
    dd = ((curves - peaks) / peaks).min(axis=1)
    final = curves[:, -1] - 1
    return dict(median_return=float(np.median(final) * 100), p5_return=float(np.percentile(final, 5) * 100),
                p95_return=float(np.percentile(final, 95) * 100), median_dd=float(np.median(dd) * 100),
                p95_dd=float(np.percentile(dd, 5) * 100), prob_loss=float((final < 0).mean() * 100),
                prob_ruin=float((dd < -ruin_dd).mean() * 100))


def print_report(res: Result, title: str = ""):
    m = res.metrics
    print(f"\n===== {title} =====")
    print(f"الصفقات: {m['n_trades']} ({m['trades_per_day']:.1f}/يوم) | نسبة الربح: {m['win_rate']:.1f}% | PF: {m['profit_factor']:.2f}")
    print(f"العائد الصافي: {m['net_pct']:+.2f}% | أقصى تراجع: {m['max_dd_pct']:.2f}% | Sharpe: {m['sharpe']:.2f} | t-stat: {m['tstat']:.2f}")
    print(f"توقع الصفقة: {m['expectancy_R']:+.3f}R | عمولات مدفوعة: {m['fees']:.0f} | متوسط المدة: {m['avg_bars']:.1f} شمعة")
    b = breakdown(res.trades)
    if not b.empty:
        print(b.to_string())
