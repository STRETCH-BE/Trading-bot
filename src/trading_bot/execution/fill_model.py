"""The single source of truth for fee, slippage and minimum-size arithmetic.

Audit finding 3: this arithmetic used to live inline inside the backtest
loop, which meant the paper and live brokers would each have to reimplement
it, and the three would drift apart the first time any of them was touched.
Extracting it here makes divergence structurally impossible — the engine and
both brokers consume the SAME object.

Nothing outside this module may compute a fee, apply slippage, or decide
whether an order clears an exchange minimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from trading_bot.data import schema

Liquidity = Literal["maker", "taker"]


@dataclass(frozen=True)
class MinimumCheck:
    ok: bool
    reason: str = ""
    failed_limit: str = ""  # "ordermin" | "costmin" | ""


@dataclass(frozen=True)
class FillModel:
    """Costs and exchange minimums. Constructed from config, shared everywhere."""

    maker_fee_bps: float
    taker_fee_bps: float
    slippage_bps: float
    default_liquidity: Liquidity = "taker"
    # Explicit overrides for tests/what-if runs. None => per-pair from schema.
    min_order_units: float | None = None
    costmin: float | None = None
    # Lot/tick rounding is OFF by default so that enabling it is a deliberate,
    # measured behaviour change rather than a silent one. See round_units().
    apply_rounding: bool = False

    @classmethod
    def from_config(cls, config) -> FillModel:
        return cls(
            maker_fee_bps=config.maker_fee_bps,
            taker_fee_bps=config.taker_fee_bps,
            slippage_bps=config.slippage_bps,
            default_liquidity=config.fee_mode,
            min_order_units=config.min_order_units,
            costmin=config.costmin,
        )

    # --- rates ---------------------------------------------------------------

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps / 10_000.0

    def fee_rate(self, liquidity: Liquidity | None = None) -> float:
        which = liquidity or self.default_liquidity
        bps = self.taker_fee_bps if which == "taker" else self.maker_fee_bps
        return bps / 10_000.0

    # --- the three operations ------------------------------------------------

    def fill_price(self, side: str, reference_price: float) -> float:
        """Reference price moved AGAINST the trader by the slippage rate."""
        rate = self.slippage_rate
        return reference_price * (1 + rate) if side == "buy" else reference_price * (1 - rate)

    def fee(self, notional: float, liquidity: Liquidity | None = None) -> float:
        return notional * self.fee_rate(liquidity)

    def clears_minimums(
        self, pair: schema.Pair | str | None, units: float, notional: float
    ) -> MinimumCheck:
        """Both exchange floors: size in base units AND value in quote currency.

        Order matters and is deliberate: ordermin is checked first so that a
        costmin failure always means "big enough to submit, too cheap to be
        allowed", which is what the diagnostics downstream assume.
        """
        min_units, cost_min = self.limits_for(pair)
        if units < min_units:
            return MinimumCheck(
                ok=False,
                reason=f"{units:.10f} units below ordermin {min_units:.10f}",
                failed_limit="ordermin",
            )
        if notional < cost_min:
            return MinimumCheck(
                ok=False,
                reason=f"notional {notional:.4f} below costmin {cost_min:.4f}",
                failed_limit="costmin",
            )
        return MinimumCheck(ok=True)

    def limits_for(self, pair: schema.Pair | str | None) -> tuple[float, float]:
        """(ordermin units, costmin quote) — per pair, or explicit overrides."""
        if self.min_order_units is not None:
            return self.min_order_units, (self.costmin if self.costmin is not None else 0.0)
        if pair is None:
            raise ValueError(
                "FillModel needs a `pair` to look up exchange minimums, or an "
                "explicit min_order_units override. Refusing to assume a "
                "default: the wrong ordermin fills orders the exchange rejects."
            )
        cost_min = self.costmin if self.costmin is not None else schema.cost_minimum(pair)
        return schema.min_order_units(pair), cost_min

    # --- lot / tick rounding -------------------------------------------------

    def round_units(self, pair: schema.Pair | str | None, units: float) -> float:
        """Round an order size DOWN to the exchange's lot precision.

        Down, never nearest: rounding up could push an order past a balance
        the account does not have. Disabled unless ``apply_rounding`` is set,
        so that turning it on is an explicit, measurable change rather than a
        silent shift in every historical number.
        """
        if not self.apply_rounding or pair is None:
            return units
        decimals = schema.lot_decimals(pair)
        factor = 10.0**decimals
        return int(units * factor) / factor

    def round_price(self, pair: schema.Pair | str | None, price: float) -> float:
        """Round a price to the exchange's tick precision (nearest)."""
        if not self.apply_rounding or pair is None:
            return price
        return round(price, schema.price_decimals(pair))
