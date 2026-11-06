"""كل إعدادات النظام في مكان واحد."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # ---- مؤشرات ----
    ema_f: int = 9
    ema_m: int = 21
    ema_s: int = 55
    rsi_n: int = 14
    atr_n: int = 14
    adx_n: int = 14
    bb_n: int = 20
    bb_k: float = 2.2
    vol_n: int = 30
    vol_spike: float = 1.8
    brk_n: int = 12

    # ---- فلتر النظام (Regime) ----
    adx_trend: float = 22.0     # فوقه = سوق ترند
    adx_range: float = 18.0     # تحته = سوق عرضي
    use_htf: bool = True        # فلتر اتجاه الإطار الأكبر
    htf_rule: str = "1h"
    htf_f: int = 9
    htf_s: int = 21

    # ---- أنواع الصفقات ----
    use_pullback: bool = False  # ارتداد داخل الترند (فعّله/دع المحسّن يقرر على بيانات حقيقية)
    use_breakout: bool = True   # اختراق بحجم عالٍ (الوحيد الذي فصل عن الضجيج في الاختبارات)
    use_meanrev: bool = False   # ارتداد للمتوسط في السوق العرضي
    allow_short: bool = True
    rsi_pb: float = 45.0
    rsi_os: float = 28.0
    z_vwap: float = 2.0

    # ---- الخروج ----
    sl_atr: float = 2.0
    tp_atr: float = 4.0
    be_atr: float = 1.5         # بعد هذه المسافة يتحرك الستوب لنقطة التعادل
    trail_atr: float = 1.5      # ستوب متحرك
    max_bars: int = 40          # خروج زمني

    # ---- فلاتر ----
    min_edge_mult: float = 4.0  # الهدف لازم يكون >= 4 أضعاف تكلفة الصفقة
    hours: tuple | None = None  # مثال (7, 21) بتوقيت UTC

    # ---- إدارة المخاطر ----
    risk_pct: float = 0.005     # 0.5% من الرصيد لكل صفقة
    max_leverage: float = 5.0
    daily_loss_pct: float = 0.03
    max_trades_day: int = 30
    max_consec_losses: int = 4
    cooldown_bars: int = 12
    kill_dd: float = 0.20       # إيقاف كامل عند خسارة 20% من القمة

    # ---- التكاليف (واقعية) ----
    taker_fee: float = 0.0004
    maker_fee: float = 0.0002
    slippage: float = 0.0001
    tp_is_maker: bool = True

    @property
    def cost_frac(self) -> float:
        exit_fee = self.maker_fee if self.tp_is_maker else self.taker_fee
        return self.taker_fee + exit_fee + 2 * self.slippage
