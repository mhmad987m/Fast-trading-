"""
محرك التنفيذ وإدارة المخاطر. نفس الكلاس يُستخدم في الباكتيست والتداول الورقي،
فالنتائج متطابقة المنطق.
قواعد محافظة: لو لمست الشمعة الستوب والهدف معاً => نعتبر الستوب ضُرب أولاً.
"""
from params import Params


class Engine:
    def __init__(self, p: Params, equity0: float = 10_000.0):
        self.p = p
        self.equity = equity0
        self.peak = equity0
        self.pos = None
        self.trades = []
        self.bar = 0
        self.day = None
        self.day_start_eq = equity0
        self.day_trades = 0
        self.day_halt = False
        self.consec = 0
        self.cool_until = -1
        self.killed = False

    # ---------- المخاطر ----------
    def _roll_day(self, ts):
        d = ts.date()
        if d != self.day:
            self.day = d
            self.day_start_eq = self.equity
            self.day_trades = 0
            self.day_halt = False

    def can_trade(self) -> bool:
        return (not self.killed and not self.day_halt
                and self.day_trades < self.p.max_trades_day
                and self.bar >= self.cool_until)

    def _risk_pct(self) -> float:
        r = self.p.risk_pct
        if self.consec >= 2:
            r *= 0.5                       # بعد خسارتين متتاليتين نصف المخاطرة
        if self.equity < self.peak * 0.90:
            r *= 0.5                       # تحت القمة بـ10% نصف المخاطرة
        return r

    # ---------- الدخول ----------
    def _open(self, ts, o, sig):
        d, a, setup = sig
        p = self.p
        entry = o * (1 + d * p.slippage)
        stop_dist = p.sl_atr * a
        if stop_dist <= 0:
            return
        risk_amt = self.equity * self._risk_pct()
        qty = risk_amt / stop_dist
        qty = min(qty, self.equity * p.max_leverage / entry)
        if qty <= 0:
            return
        self.pos = dict(d=d, entry=entry, qty=qty, sl=entry - d * stop_dist,
                        sl0=entry - d * stop_dist, tp=entry + d * p.tp_atr * a,
                        atr=a, best=entry, held=0, setup=setup, ts=ts, risk=risk_amt)
        self.day_trades += 1

    # ---------- الخروج ----------
    def _close(self, ts, px, reason, maker=False):
        p, pos = self.p, self.pos
        fee_x = p.maker_fee if maker else p.taker_fee
        gross = pos["d"] * (px - pos["entry"]) * pos["qty"]
        fees = pos["entry"] * pos["qty"] * p.taker_fee + px * pos["qty"] * fee_x
        net = gross - fees
        eq_before = self.equity
        self.equity += net
        self.peak = max(self.peak, self.equity)
        self.trades.append(dict(entry_ts=pos["ts"], exit_ts=ts, dir=pos["d"], setup=pos["setup"],
                                entry=pos["entry"], exit=px, qty=pos["qty"], pnl=net, fees=fees,
                                ret=net / eq_before, R=net / pos["risk"], reason=reason, bars=pos["held"]))
        self.pos = None
        if net < 0:
            self.consec += 1
            if self.consec >= p.max_consec_losses:
                self.cool_until = self.bar + p.cooldown_bars
                self.consec = 0
        else:
            self.consec = 0
        if self.equity <= self.day_start_eq * (1 - p.daily_loss_pct):
            self.day_halt = True
        if self.equity <= self.peak * (1 - p.kill_dd):
            self.killed = True

    def _manage(self, ts, o, h, l, c):
        p, pos = self.p, self.pos
        d = pos["d"]
        moved = pos["sl"] != pos["sl0"]
        sl_label = "trail" if moved else "sl"
        if d == 1:
            if o <= pos["sl"]:
                return self._close(ts, o * (1 - p.slippage), sl_label)
            if l <= pos["sl"]:
                return self._close(ts, pos["sl"] * (1 - p.slippage), sl_label)
            if h >= pos["tp"]:
                return self._close(ts, max(pos["tp"], o), "tp", maker=p.tp_is_maker)
            pos["best"] = max(pos["best"], h)
            if pos["best"] - pos["entry"] >= p.be_atr * pos["atr"]:
                pos["sl"] = max(pos["sl"], pos["entry"] + 2 * p.slippage * pos["entry"],
                                pos["best"] - p.trail_atr * pos["atr"])
        else:
            if o >= pos["sl"]:
                return self._close(ts, o * (1 + p.slippage), sl_label)
            if h >= pos["sl"]:
                return self._close(ts, pos["sl"] * (1 + p.slippage), sl_label)
            if l <= pos["tp"]:
                return self._close(ts, min(pos["tp"], o), "tp", maker=p.tp_is_maker)
            pos["best"] = min(pos["best"], l)
            if pos["entry"] - pos["best"] >= p.be_atr * pos["atr"]:
                pos["sl"] = min(pos["sl"], pos["entry"] - 2 * p.slippage * pos["entry"],
                                pos["best"] + p.trail_atr * pos["atr"])
        pos["held"] += 1
        if pos["held"] >= p.max_bars:
            self._close(ts, c * (1 - d * p.slippage), "time")

    # ---------- خطوة واحدة = شمعة واحدة ----------
    def step(self, ts, o, h, l, c, sig=None):
        """sig = (dir, atr, setup) من إغلاق الشمعة السابقة أو None."""
        self.bar += 1
        self._roll_day(ts)
        if self.pos is None and sig is not None and self.can_trade():
            self._open(ts, o, sig)
        if self.pos is not None:
            self._manage(ts, o, h, l, c)
        if self.killed and self.pos is not None:
            self._close(ts, c, "kill")
        return self.mtm(c)

    def mtm(self, c) -> float:
        if self.pos is None:
            return self.equity
        pos = self.pos
        return self.equity + pos["d"] * (c - pos["entry"]) * pos["qty"]

    def force_close(self, ts, c):
        if self.pos is not None:
            self._close(ts, c, "end")
