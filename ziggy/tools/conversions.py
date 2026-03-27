"""Unit conversion tools."""

import re


def handle_conversion(text):
    text_lower = text.lower()

    # Temperature conversion
    if "celsius" in text_lower and "fahrenheit" in text_lower:
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            if "celsius" in text_lower.split("fahrenheit")[0]:
                c = float(numbers[0])
                f = (c * 9 / 5) + 32
                return f"{c} degrees Celsius is {f:.1f} degrees Fahrenheit"
            else:
                f = float(numbers[0])
                c = (f - 32) * 5 / 9
                return f"{f} degrees Fahrenheit is {c:.1f} degrees Celsius"

    # Distance conversion
    if "feet" in text_lower and "meters" in text_lower:
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            if "feet" in text_lower.split("meters")[0]:
                ft = float(numbers[0])
                m = ft * 0.3048
                return f"{ft} feet is {m:.2f} meters"
            else:
                m = float(numbers[0])
                ft = m / 0.3048
                return f"{m} meters is {ft:.2f} feet"

    return None


CONVERSION_KEYWORDS = [
    "convert", "celsius", "fahrenheit", "meters", "feet",
    "pounds", "kilograms",
]


def register(registry):
    registry.register(
        name="conversion",
        description="Handle unit conversions",
        keywords=CONVERSION_KEYWORDS,
        handler=handle_conversion,
    )
