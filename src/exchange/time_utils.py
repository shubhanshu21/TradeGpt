"""NSE market hours are always IST, but the host machine's system clock
often isn't (most cloud VMs default to UTC). Every engine that makes
"is the market open" / "what date is it" decisions must go through
now_ist() instead of datetime.now() - using the wrong timezone here means
the market-hours check fires at the wrong real-world time, silently.

This also matters for a second reason: broker candle timestamps (at least
Upstox's) come back timezone-AWARE in IST (e.g. "2026-07-10 14:50:00+05:30").
Comparing that against a naive datetime.now() raises TypeError at runtime
the first time it happens - now_ist() returns a tz-aware datetime so it
compares cleanly against candle data from any broker.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    return datetime.now(IST)
