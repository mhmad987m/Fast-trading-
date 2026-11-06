"""
Scalper-Pro  — نظام مضاربة سريعة: استراتيجيات + إدارة مخاطر + باكتيست + Walk-Forward + Monte Carlo + Paper.
أمثلة:
  python main.py demo
  python main.py fetch --symbol BTCUSDT --tf 5m --days 60 --out btc5m.csv
  python main.py backtest --csv btc5m.csv --tf 5m
  python main.py optimize --csv btc5m.csv --tf 5m
  python main.py paper --symbol BTCUSDT --tf 5m
"""
import argparse
from dataclasses import asdict

import pandas as pd

from backtest import monte_carlo, print_report, run_backtest
from data import TF_MIN, fetch_history, load_csv, synthetic
from optimize import walk_forward
from params import Params


def plot_equity(eq: pd.Series, path: str, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq.index, eq.values, lw=1.2)
    ax.set_title(title); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print("تم حفظ الرسم:", path)


def print_mc(trades):
    mc = monte_carlo(trades)
    if not mc:
        print("Monte Carlo: صفقات قليلة جداً")
        return
    print(f"Monte Carlo (3000 محاكاة): وسيط العائد {mc['median_return']:+.1f}% | "
          f"[5%..95%] = [{mc['p5_return']:+.1f}%, {mc['p95_return']:+.1f}%] | "
          f"أسوأ تراجع (95%) {mc['p95_dd']:.1f}% | احتمال الخسارة {mc['prob_loss']:.0f}% | احتمال تراجع>30% {mc['prob_ruin']:.0f}%")


def cmd_demo(a):
    p = Params()
    print("1) اختبار سلامة: مشي عشوائي بلا أي أفضلية (المتوقع: خسارة بسيطة = التكاليف)")
    rw = synthetic(days=a.days, tf_min=15, structured=False, seed=3)
    print_report(run_backtest(rw, p), "Random Walk 15m (لا أفضلية)")
    print("\n2) بيانات اصطناعية بأنظمة سوق (ترند/عرضي/تذبذب) — للتحقق من عمل المنطق فقط")
    df = synthetic(days=a.days, tf_min=15, structured=True, seed=11)
    res = run_backtest(df, p)
    print_report(res, "Structured synthetic 15m — معاملات افتراضية")
    print_mc(res.trades)
    print("\n3) Walk-Forward (تدريب 21 يوم / اختبار 7 أيام، OOS فقط)")
    trades, curve, m, folds, best = walk_forward(df, p, 15, n_iter=a.iters)
    if m:
        print(f"OOS المجمّع: صفقات={m['n_trades']} عائد={m['net_pct']:+.2f}% تراجع={m['max_dd_pct']:.2f}% "
              f"PF={m['profit_factor']:.2f} t-stat={m['tstat']:.2f}")
        print_mc(trades)
        plot_equity(curve, "equity_oos.png", "Out-of-sample equity (synthetic)")


def cmd_fetch(a):
    df = fetch_history(a.symbol, a.tf, a.days)
    df.to_csv(a.out)
    print(f"تم حفظ {len(df)} شمعة في {a.out} ({df.index[0]} → {df.index[-1]})")


def cmd_backtest(a):
    df = load_csv(a.csv)
    res = run_backtest(df, Params())
    print_report(res, f"Backtest {a.csv}")
    print_mc(res.trades)
    plot_equity(res.equity, "equity.png", "Equity")
    res.trades.to_csv("trades.csv", index=False)


def cmd_optimize(a):
    df = load_csv(a.csv)
    trades, curve, m, folds, best = walk_forward(df, Params(), TF_MIN[a.tf], n_iter=a.iters)
    if m:
        print(f"\nOOS المجمّع: صفقات={m['n_trades']} عائد={m['net_pct']:+.2f}% تراجع={m['max_dd_pct']:.2f}% "
              f"PF={m['profit_factor']:.2f} Sharpe={m['sharpe']:.2f} t-stat={m['tstat']:.2f}")
        print_mc(trades)
        plot_equity(curve, "equity_oos.png", "Out-of-sample equity")
    print("\nآخر معاملات مختارة:\n", asdict(best))


def cmd_paper(a):
    from live import run_paper
    run_paper(a.symbol, a.tf, Params(), a.equity)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("demo"); s.add_argument("--days", type=int, default=120); s.add_argument("--iters", type=int, default=25); s.set_defaults(f=cmd_demo)
    s = sub.add_parser("fetch"); s.add_argument("--symbol", default="BTCUSDT"); s.add_argument("--tf", default="5m"); s.add_argument("--days", type=int, default=60); s.add_argument("--out", default="data.csv"); s.set_defaults(f=cmd_fetch)
    s = sub.add_parser("backtest"); s.add_argument("--csv", required=True); s.add_argument("--tf", default="5m"); s.set_defaults(f=cmd_backtest)
    s = sub.add_parser("optimize"); s.add_argument("--csv", required=True); s.add_argument("--tf", default="5m"); s.add_argument("--iters", type=int, default=40); s.set_defaults(f=cmd_optimize)
    s = sub.add_parser("paper"); s.add_argument("--symbol", default="BTCUSDT"); s.add_argument("--tf", default="5m"); s.add_argument("--equity", type=float, default=10000); s.set_defaults(f=cmd_paper)
    a = ap.parse_args()
    a.f(a)
