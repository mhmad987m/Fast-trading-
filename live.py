"""
تداول ورقي (Paper) + تنبيهات إشارات على بيانات باينانس الحية.
لا يرسل أي أوامر حقيقية. يحاكي حساباً وهمياً بنفس محرك الباكتيست.
شغّله أسابيع قبل التفكير بأي مال حقيقي.
"""
import time

import pandas as pd

from data import TF_MIN, fetch_klines
from engine import Engine
from params import Params
from strategy import SETUP_NAMES, generate_signals


def run_paper(symbol="BTCUSDT", tf="5m", p: Params = Params(), equity0=10_000.0, poll=5, log="paper_trades.csv"):
    eng = Engine(p, equity0)
    pending, last_ts = None, None
    print(f"[paper] {symbol} {tf} | رصيد وهمي {equity0:.0f} | Ctrl+C للإيقاف")
    while True:
        try:
            df = fetch_klines(symbol, tf, 800)
        except Exception as e:  # شبكة/حظر مؤقت
            print("خطأ شبكة:", e)
            time.sleep(15)
            continue
        closed = df.iloc[:-1]
        newest = closed.index[-1]
        if last_ts is None:
            last_ts = newest
            print(f"جاهز. آخر شمعة مغلقة: {newest}")
        elif newest > last_ts:
            for ts in closed.index[closed.index > last_ts]:
                r = closed.loc[ts]
                eng.step(ts, r.open, r.high, r.low, r.close, pending)
                pending = None
            last_ts = newest
            sig, atr_, setup = generate_signals(closed, p)
            if eng.pos is None and sig[-1] != 0 and eng.can_trade():
                d, a = int(sig[-1]), float(atr_[-1])
                nxt = float(df.iloc[-1].open)
                pending = (d, a, int(setup[-1]))
                print(f"\n>>> إشارة {'LONG' if d == 1 else 'SHORT'} [{SETUP_NAMES[int(setup[-1])]}] "
                      f"دخول≈{nxt:.2f} SL={nxt - d * p.sl_atr * a:.2f} TP={nxt + d * p.tp_atr * a:.2f}")
            st = "في صفقة" if eng.pos else "بلا صفقة"
            print(f"{newest} | رصيد={eng.equity:.2f} | صفقات={len(eng.trades)} | {st}"
                  f"{' | KILL' if eng.killed else ''}{' | وقف يومي' if eng.day_halt else ''}")
            if eng.trades:
                pd.DataFrame(eng.trades).to_csv(log, index=False)
        time.sleep(poll)
