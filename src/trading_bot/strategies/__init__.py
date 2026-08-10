"""Stage 3: strategies. Each is a pure signal function usable by the backtester."""

from trading_bot.strategies.donchian import DonchianParams, donchian_breakout
from trading_bot.strategies.voltrend import VolTrendParams, voltrend

__all__ = ["DonchianParams", "VolTrendParams", "donchian_breakout", "voltrend"]
