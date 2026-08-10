"""Stage 3: strategies. Each is a pure signal function usable by the backtester."""

from trading_bot.strategies.donchian import DonchianParams, donchian_breakout

__all__ = ["DonchianParams", "donchian_breakout"]
