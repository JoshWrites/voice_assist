"""Weather tool — current conditions via wttr.in (no API key needed)."""

import re

import requests


def handle_weather(text):
    """Extract location from text and fetch current weather."""
    location = _extract_location(text)
    if not location:
        return "I didn't catch the location. Try saying something like: what's the weather in Los Angeles."

    try:
        resp = requests.get(
            f"https://wttr.in/{location}?format=j1",
            timeout=10,
            headers={"User-Agent": "ziggy-voice-assistant"},
        )
        if resp.status_code != 200:
            return f"I couldn't get the weather for {location}."

        data = resp.json()
        current = data["current_condition"][0]
        area = data["nearest_area"][0]
        city = area["areaName"][0]["value"]
        region = area.get("region", [{}])[0].get("value", "")

        temp_f = current["temp_F"]
        temp_c = current["temp_C"]
        feels_f = current["FeelsLikeF"]
        condition = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind_mph = current["windspeedMiles"]
        wind_dir = current["winddir16Point"]

        place = f"{city}, {region}" if region else city
        return (
            f"In {place} right now it's {temp_f} degrees Fahrenheit, {temp_c} Celsius. "
            f"{condition}. Feels like {feels_f}. "
            f"Humidity {humidity} percent, wind {wind_mph} miles per hour from the {wind_dir}."
        )
    except Exception as e:
        print(f"Weather error: {e}")
        return f"Sorry, I had trouble getting the weather for {location}."


def _extract_location(text):
    """Pull a location name from natural speech."""
    text_lower = text.lower()
    # "weather in <location>" / "weather for <location>"
    for prep in ["in", "for", "at", "around"]:
        pattern = rf"weather\s+{prep}\s+(.+?)(?:\s+right now|\s+today|\s+currently)?$"
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).strip()

    # "what's it like in <location>"
    match = re.search(r"like\s+in\s+(.+?)(?:\s+right now|\s+today)?$", text_lower)
    if match:
        return match.group(1).strip()

    # Fallback: remove common words and use what's left
    for word in ["weather", "what's", "the", "whats", "tell", "me", "give",
                  "about", "how", "is", "it", "right", "now", "today", "currently",
                  "go", "online", "and", "check", "get"]:
        text_lower = text_lower.replace(word, "")
    cleaned = text_lower.strip()
    return cleaned if len(cleaned) > 2 else None


WEATHER_KEYWORDS = ["weather", "temperature", "forecast"]


def register(registry):
    registry.register(
        name="weather",
        description="Get current weather for a location",
        keywords=WEATHER_KEYWORDS,
        handler=handle_weather,
    )
