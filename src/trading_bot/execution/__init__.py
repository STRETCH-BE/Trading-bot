"""Execution primitives: the order seam, the shared fill model, translation.

Deliberately contains no strategy logic and, as of Stage 5a, no broker and no
risk gate. It exists so that later stages have a submission boundary to put a
mandatory gate in front of.
"""

from trading_bot.execution.fill_model import FillModel, Liquidity, MinimumCheck
from trading_bot.execution.order import (
    Order,
    OrderType,
    PortfolioState,
    Side,
    make_client_order_id,
)
from trading_bot.execution.translate import TargetTranslation, orders_for_target

__all__ = [
    "FillModel",
    "Liquidity",
    "MinimumCheck",
    "Order",
    "OrderType",
    "PortfolioState",
    "Side",
    "TargetTranslation",
    "make_client_order_id",
    "orders_for_target",
]
