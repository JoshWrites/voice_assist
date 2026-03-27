"""Time and date tools."""

from datetime import datetime


def get_time(_text=None):
    now = datetime.now()
    return f"The time is {now.strftime('%I:%M %p')}"


def get_date(_text=None):
    now = datetime.now()
    return f"Today is {now.strftime('%A, %B %d, %Y')}"


def register(registry):
    registry.register(
        name="time",
        description="Get the current time",
        keywords=["time", "clock", "what time"],
        handler=get_time,
    )
    registry.register(
        name="date",
        description="Get the current date",
        keywords=["date", "today", "what day"],
        handler=get_date,
    )
