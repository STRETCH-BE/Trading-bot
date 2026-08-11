"""Durable SQLite state. Survives restarts; a crash mid-write leaves it consistent.

Every logical step is exactly ONE transaction, so a process killed at any
moment lands on a step boundary and never inside a half-applied change.
Recording a fill — insert the fill row, update the order's aggregates, update
the position — is a single atomic unit precisely because those three must
never disagree.

The ``decisions`` table records EVERY strategy evaluation including "do
nothing", with its reasoning. It is the answer to "why did it do that in
March", and a cycle that decides nothing still writes a row.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading_bot.broker.base import OrderState, OrderStatus, Position
from trading_bot.execution.order import Order

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY,
    pair            TEXT NOT NULL,
    side            TEXT NOT NULL,
    units           REAL NOT NULL,
    order_type      TEXT NOT NULL,
    limit_price     REAL,
    status          TEXT NOT NULL,
    submitted_at    TEXT NOT NULL,
    filled_units    REAL NOT NULL DEFAULT 0.0,
    avg_fill_price  REAL NOT NULL DEFAULT 0.0,
    fees            REAL NOT NULL DEFAULT 0.0,
    reason          TEXT NOT NULL DEFAULT '',
    rejection_reason TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS fills (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id  TEXT NOT NULL REFERENCES orders(client_order_id),
    timestamp TEXT NOT NULL,
    units     REAL NOT NULL,
    price     REAL NOT NULL,
    fee       REAL NOT NULL,
    liquidity TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id);

CREATE TABLE IF NOT EXISTS positions (
    pair      TEXT PRIMARY KEY,
    units     REAL NOT NULL,
    avg_cost  REAL NOT NULL,
    opened_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS equity (
    timestamp      TEXT NOT NULL,
    cash           REAL NOT NULL,
    position_value REAL NOT NULL,
    total_equity   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity(timestamp);

CREATE TABLE IF NOT EXISTS decisions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    strategy         TEXT NOT NULL,
    pair             TEXT NOT NULL,
    current_position REAL NOT NULL,
    target_position  REAL NOT NULL,
    action_taken     TEXT NOT NULL,
    reasoning        TEXT NOT NULL,
    cycle_id         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_cycle ON decisions(cycle_id);

CREATE TABLE IF NOT EXISTS account (
    key   TEXT PRIMARY KEY,
    value REAL NOT NULL
);
"""


@dataclass(frozen=True)
class Decision:
    timestamp: pd.Timestamp
    strategy: str
    pair: str
    current_position: float
    target_position: float
    action_taken: str
    reasoning: str
    cycle_id: str


class StateStore:
    """All persistence. Nothing else may write to the database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One atomic unit. A crash inside leaves the database untouched."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    # --- account scalars -----------------------------------------------------

    def set_cash(self, cash: float) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO account(key, value) VALUES('cash', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (cash,),
            )

    def get_cash(self, default: float = 0.0) -> float:
        row = self._conn.execute("SELECT value FROM account WHERE key='cash'").fetchone()
        return float(row["value"]) if row else default

    # --- orders --------------------------------------------------------------

    def insert_order(self, order: Order, status: OrderStatus, submitted_at: pd.Timestamp,
                     rejection_reason: str = "") -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO orders(client_order_id, pair, side, units, order_type, "
                "limit_price, status, submitted_at, reason, rejection_reason) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    order.client_order_id, order.pair, order.side, order.units,
                    order.order_type, order.limit_price, str(status),
                    order.timestamp.isoformat() if hasattr(order.timestamp, "isoformat")
                    else str(order.timestamp),
                    order.reason, rejection_reason,
                ),
            )

    def get_order(self, client_order_id: str) -> OrderState | None:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE client_order_id=?", (client_order_id,)
        ).fetchone()
        return self._row_to_state(row) if row else None

    def open_orders(self) -> list[OrderState]:
        rows = self._conn.execute(
            "SELECT * FROM orders WHERE status IN (?,?,?) ORDER BY submitted_at",
            (str(OrderStatus.PENDING), str(OrderStatus.OPEN),
             str(OrderStatus.PARTIALLY_FILLED)),
        ).fetchall()
        return [self._row_to_state(r) for r in rows]

    def all_orders(self) -> list[OrderState]:
        rows = self._conn.execute("SELECT * FROM orders ORDER BY submitted_at").fetchall()
        return [self._row_to_state(r) for r in rows]

    def set_order_status(self, client_order_id: str, status: OrderStatus) -> None:
        with self.transaction() as c:
            c.execute(
                "UPDATE orders SET status=? WHERE client_order_id=?",
                (str(status), client_order_id),
            )

    @staticmethod
    def _row_to_state(row: sqlite3.Row) -> OrderState:
        order = Order(
            client_order_id=row["client_order_id"],
            pair=row["pair"],
            side=row["side"],
            units=row["units"],
            order_type=row["order_type"],
            limit_price=row["limit_price"],
            reason=row["reason"],
            timestamp=pd.Timestamp(row["submitted_at"]),
        )
        return OrderState(
            order=order,
            status=OrderStatus(row["status"]),
            filled_units=row["filled_units"],
            avg_fill_price=row["avg_fill_price"],
            fees=row["fees"],
            submitted_at=pd.Timestamp(row["submitted_at"]),
            rejection_reason=row["rejection_reason"],
        )

    # --- the atomic fill -----------------------------------------------------

    def record_fill(
        self,
        client_order_id: str,
        timestamp: pd.Timestamp,
        units: float,
        price: float,
        fee: float,
        liquidity: str,
        *,
        cash_delta: float,
        pair: str,
        side: str,
    ) -> None:
        """Fill row + order aggregates + position + cash, in ONE transaction.

        These four must never disagree, so they commit together or not at all.
        """
        with self.transaction() as c:
            c.execute(
                "INSERT INTO fills(order_id, timestamp, units, price, fee, liquidity) "
                "VALUES(?,?,?,?,?,?)",
                (client_order_id, timestamp.isoformat(), units, price, fee, liquidity),
            )
            row = c.execute(
                "SELECT units, filled_units, avg_fill_price, fees FROM orders "
                "WHERE client_order_id=?",
                (client_order_id,),
            ).fetchone()
            prev_filled = row["filled_units"]
            new_filled = prev_filled + units
            new_avg = (
                (row["avg_fill_price"] * prev_filled + price * units) / new_filled
                if new_filled > 0 else 0.0
            )
            status = (
                OrderStatus.FILLED
                if new_filled >= row["units"] - 1e-12
                else OrderStatus.PARTIALLY_FILLED
            )
            c.execute(
                "UPDATE orders SET filled_units=?, avg_fill_price=?, fees=?, status=? "
                "WHERE client_order_id=?",
                (new_filled, new_avg, row["fees"] + fee, str(status), client_order_id),
            )

            signed = units if side == "buy" else -units
            pos = c.execute(
                "SELECT units, avg_cost FROM positions WHERE pair=?", (pair,)
            ).fetchone()
            if pos is None:
                c.execute(
                    "INSERT INTO positions(pair, units, avg_cost, opened_at) "
                    "VALUES(?,?,?,?)",
                    (pair, signed, price, timestamp.isoformat()),
                )
            else:
                new_units = pos["units"] + signed
                if side == "buy":
                    total = pos["units"] + units
                    new_cost = (
                        (pos["avg_cost"] * pos["units"] + price * units) / total
                        if total > 0 else 0.0
                    )
                else:
                    new_cost = pos["avg_cost"]  # selling does not change basis
                if abs(new_units) < 1e-12:
                    c.execute("DELETE FROM positions WHERE pair=?", (pair,))
                else:
                    c.execute(
                        "UPDATE positions SET units=?, avg_cost=? WHERE pair=?",
                        (new_units, new_cost, pair),
                    )

            cash_row = c.execute("SELECT value FROM account WHERE key='cash'").fetchone()
            cash = (cash_row["value"] if cash_row else 0.0) + cash_delta
            c.execute(
                "INSERT INTO account(key, value) VALUES('cash', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (cash,),
            )

    def fills_for(self, client_order_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM fills WHERE order_id=? ORDER BY id", (client_order_id,)
        ).fetchall()

    def all_fills(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM fills ORDER BY id").fetchall()

    # --- positions / equity / decisions --------------------------------------

    def positions(self) -> list[Position]:
        rows = self._conn.execute("SELECT * FROM positions ORDER BY pair").fetchall()
        return [
            Position(
                pair=r["pair"], units=r["units"], avg_cost=r["avg_cost"],
                opened_at=pd.Timestamp(r["opened_at"]),
            )
            for r in rows
        ]

    def record_equity(
        self, timestamp: pd.Timestamp, cash: float, position_value: float
    ) -> None:
        with self.transaction() as c:
            c.execute(
                "INSERT INTO equity(timestamp, cash, position_value, total_equity) "
                "VALUES(?,?,?,?)",
                (timestamp.isoformat(), cash, position_value, cash + position_value),
            )

    def equity_history(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM equity ORDER BY timestamp"
        ).fetchall()

    def record_decision(self, d: Decision) -> None:
        """Every evaluation, including 'do nothing'."""
        with self.transaction() as c:
            c.execute(
                "INSERT INTO decisions(timestamp, strategy, pair, current_position, "
                "target_position, action_taken, reasoning, cycle_id) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    d.timestamp.isoformat(), d.strategy, d.pair, d.current_position,
                    d.target_position, d.action_taken, d.reasoning, d.cycle_id,
                ),
            )

    def recent_decisions(self, limit: int = 5) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    # --- integrity -----------------------------------------------------------

    def check_integrity(self) -> list[str]:
        """Invariants that must hold after ANY crash. Empty list == consistent."""
        problems: list[str] = []

        orphans = self._conn.execute(
            "SELECT COUNT(*) n FROM fills f LEFT JOIN orders o "
            "ON f.order_id = o.client_order_id WHERE o.client_order_id IS NULL"
        ).fetchone()["n"]
        if orphans:
            problems.append(f"{orphans} fill(s) reference a missing order")

        for row in self._conn.execute(
            "SELECT o.client_order_id, o.units, o.filled_units, o.status, "
            "COALESCE(SUM(f.units), 0) AS fill_sum, COALESCE(SUM(f.fee), 0) AS fee_sum, "
            "o.fees FROM orders o LEFT JOIN fills f ON f.order_id = o.client_order_id "
            "GROUP BY o.client_order_id"
        ):
            if abs(row["filled_units"] - row["fill_sum"]) > 1e-9:
                problems.append(
                    f"order {row['client_order_id']}: filled_units "
                    f"{row['filled_units']} != sum(fills) {row['fill_sum']}"
                )
            if abs(row["fees"] - row["fee_sum"]) > 1e-9:
                problems.append(
                    f"order {row['client_order_id']}: fees {row['fees']} != "
                    f"sum(fill fees) {row['fee_sum']}"
                )
            if row["filled_units"] > row["units"] + 1e-9:
                problems.append(
                    f"order {row['client_order_id']}: overfilled "
                    f"{row['filled_units']} > {row['units']}"
                )
            if row["status"] == str(OrderStatus.FILLED) and (
                abs(row["filled_units"] - row["units"]) > 1e-9
            ):
                problems.append(
                    f"order {row['client_order_id']}: status FILLED but "
                    f"{row['filled_units']} of {row['units']} filled"
                )

        for row in self._conn.execute("SELECT pair FROM positions"):
            pair = row["pair"]
            net = self._conn.execute(
                "SELECT COALESCE(SUM(CASE WHEN o.side='buy' THEN f.units ELSE -f.units END), 0) "
                "AS net FROM fills f JOIN orders o ON f.order_id=o.client_order_id "
                "WHERE o.pair=?",
                (pair,),
            ).fetchone()["net"]
            held = self._conn.execute(
                "SELECT units FROM positions WHERE pair=?", (pair,)
            ).fetchone()["units"]
            if abs(net - held) > 1e-9:
                problems.append(
                    f"position {pair}: held {held} != net of fills {net}"
                )

        return problems
