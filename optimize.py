"""
تحسين المعاملات بطريقة Walk-Forward:
نحسّن على نافذة تدريب، ثم نختبر النتيجة على فترة لم يرها المحسّن أبداً (Out-of-Sample).
الرقم الوحيد الذي يُعتد به هو رقم OOS المجمّع.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest import compute_metrics, run_backtest
from params import Params

SPACE = {
    "sl_atr": ("f", 0.8, 2.0), "tp_atr": ("f", 1.2, 3.5), "trail_atr": ("f", 0.6, 1.6),
    "be_atr": ("f", 0.6, 1.5), "adx_trend": ("f", 16, 30), "adx_range": ("f", 14, 22),
    "vol_spike": ("f", 1.3, 2.8), "bb_k": ("f", 1.8, 2.8), "rsi_pb": ("f", 38, 50),
    "rsi_os": ("f", 20, 35), "brk_n": ("i", 6, 24), "max_bars": ("i", 12, 60),
    "use_pullback": ("b",), "use_breakout": ("b",), "use_meanrev": ("b",),
}
WARM = 600


def sample(base: Params, rng) -> Params:
    kw = {}
    for k, spec in SPACE.items():
        if spec[0] == "f":
            kw[k] = float(rng.uniform(spec[1], spec[2]))
        elif spec[0] == "i":
            kw[k] = int(rng.integers(spec[1], spec[2] + 1))
        else:
            kw[k] = bool(rng.integers(0, 2))
    if not (kw["use_pullback"] or kw["use_breakout"] or kw["use_meanrev"]):
        kw["use_pullback"] = True
    return replace(base, **kw)


def objective(m: dict, min_trades: int) -> float:
    if m["n_trades"] < min_trades:
        return -1e9
    dd_pen = 1 - min(abs(m["max_dd_pct"]) / 30, 1)
    return m["tstat"] * dd_pen


def search(train: pd.DataFrame, base: Params, n_iter: int, rng, min_trades: int):
    best, best_s = base, -1e18
    cands = [base] + [sample(base, rng) for _ in range(n_iter)]
    for p in cands:
        res = run_backtest(train, p, start=WARM)
        s = objective(res.metrics, min_trades)
        if s > best_s:
            best, best_s = p, s
    return best, best_s


def walk_forward(df, base: Params, tf_min: int, train_days=21, test_days=7,
                 n_iter=40, seed=1, equity0=10_000.0, verbose=True):
    rng = np.random.default_rng(seed)
    bpd = 1440 // tf_min
    tr, te = train_days * bpd, test_days * bpd
    pos, equity = WARM, equity0
    all_trades, curves, folds = [], [], []
    while pos + tr + te <= len(df):
        train = df.iloc[pos - WARM: pos + tr]
        best, score = search(train, base, n_iter, rng, min_trades=max(15, train_days))
        test = df.iloc[pos + tr - WARM: pos + tr + te]
        res = run_backtest(test, best, equity0=equity, start=WARM)
        folds.append(dict(fold=len(folds) + 1, train_score=round(score, 2), n=res.metrics["n_trades"],
                          oos_pct=round(res.metrics["net_pct"], 2), dd=round(res.metrics["max_dd_pct"], 2)))
        if verbose:
            print(f"  fold {folds[-1]['fold']}: score_train={folds[-1]['train_score']:>6} "
                  f"| OOS trades={folds[-1]['n']:>3} ret={folds[-1]['oos_pct']:+.2f}% dd={folds[-1]['dd']:.2f}%")
        if not res.trades.empty:
            all_trades.append(res.trades)
        curves.append(res.equity.iloc[WARM:])
        equity = res.final_equity
        pos += te
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    curve = pd.concat(curves) if curves else pd.Series(dtype=float)
    metrics = compute_metrics(trades, curve, equity0) if len(curve) else {}
    return trades, curve, metrics, pd.DataFrame(folds), best
