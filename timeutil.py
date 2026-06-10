"""Time parsing/formatting shared by all GVD tools.

Tools accept timestamps as float seconds, "SS", "MM:SS", or "HH:MM:SS"
(matching DVD's convention) and always display HH:MM:SS back to the model.
"""

from __future__ import annotations

from typing import Union

TimeLike = Union[int, float, str]


def to_seconds(t: TimeLike) -> float:
    """Parse seconds / 'MM:SS' / 'HH:MM:SS' into float seconds."""
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t).strip()
    if not s:
        raise ValueError("empty timestamp")
    if ":" not in s:
        return float(s)
    parts = s.split(".")[0].split(":")
    if len(parts) == 2:
        parts = ["0"] + parts
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {t!r}. Use seconds or HH:MM:SS.")
    h, m, sec = (int(p) for p in parts)
    return float(h * 3600 + m * 60 + sec)


def fmt(seconds: float) -> str:
    """Format float seconds as HH:MM:SS."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}"


def fmt_span(t_start: float, t_end: float) -> str:
    return f"{fmt(t_start)}–{fmt(t_end)}"
