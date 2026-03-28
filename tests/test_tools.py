"""Tests for ziggy.tools — registry, conversions, time/date, weather."""

from datetime import datetime
from unittest.mock import patch, MagicMock

from ziggy.tools import ToolRegistry
from ziggy.tools.conversions import handle_conversion, register as register_conversions
from ziggy.tools.time_date import get_time, get_date, register as register_time_date


class TestToolRegistry:
    def test_empty_registry(self):
        reg = ToolRegistry()
        assert reg.find_tool("hello") is None
        assert reg.list_tools() == []

    def test_register_and_find(self):
        reg = ToolRegistry()
        reg.register("greet", "Say hello", ["hello", "hi"], lambda t: "Hello!")
        tool = reg.find_tool("say hello to me")
        assert tool is not None
        assert tool.name == "greet"
        assert tool.handler("") == "Hello!"

    def test_find_returns_first_match(self):
        reg = ToolRegistry()
        reg.register("a", "Tool A", ["time"], lambda t: "A")
        reg.register("b", "Tool B", ["time"], lambda t: "B")
        tool = reg.find_tool("what time is it")
        assert tool.name == "a"

    def test_no_match(self):
        reg = ToolRegistry()
        reg.register("greet", "Say hello", ["hello"], lambda t: "Hello!")
        assert reg.find_tool("what is the weather") is None

    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register("a", "Desc A", ["x"], lambda t: None)
        reg.register("b", "Desc B", ["y"], lambda t: None)
        tools = reg.list_tools()
        assert ("a", "Desc A") in tools
        assert ("b", "Desc B") in tools

    def test_register_time_date_tools(self):
        reg = ToolRegistry()
        register_time_date(reg)
        assert reg.find_tool("what time is it") is not None
        assert reg.find_tool("what date is today") is not None

    def test_register_conversion_tools(self):
        reg = ToolRegistry()
        register_conversions(reg)
        assert reg.find_tool("convert celsius to fahrenheit") is not None


class TestConversions:
    def test_celsius_to_fahrenheit(self):
        result = handle_conversion("Convert 100 celsius to fahrenheit")
        assert result is not None
        assert "212.0" in result

    def test_fahrenheit_to_celsius(self):
        result = handle_conversion("Convert 32 fahrenheit to celsius")
        assert result is not None
        assert "0.0" in result

    def test_feet_to_meters(self):
        result = handle_conversion("Convert 10 feet to meters")
        assert result is not None
        assert "3.05" in result

    def test_meters_to_feet(self):
        result = handle_conversion("Convert 1 meters to feet")
        assert result is not None
        assert "3.28" in result

    def test_negative_temperature(self):
        result = handle_conversion("Convert -40 celsius to fahrenheit")
        assert result is not None
        assert "-40" in result

    def test_no_match_returns_none(self):
        assert handle_conversion("hello world") is None

    def test_missing_number_returns_none(self):
        assert handle_conversion("celsius to fahrenheit") is None


class TestTimeDate:
    @patch("ziggy.tools.time_date.datetime")
    def test_get_time(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 27, 14, 30)
        result = get_time()
        assert "02:30 PM" in result

    @patch("ziggy.tools.time_date.datetime")
    def test_get_date(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 3, 27)
        result = get_date()
        assert "Friday" in result
        assert "March" in result
        assert "27" in result


class TestWeather:
    @patch("ziggy.tools.weather.requests.get")
    def test_weather_success(self, mock_get):
        from ziggy.tools.weather import handle_weather
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "current_condition": [{
                    "temp_F": "72", "temp_C": "22", "FeelsLikeF": "70",
                    "weatherDesc": [{"value": "Sunny"}],
                    "humidity": "45", "windspeedMiles": "5", "winddir16Point": "SW",
                }],
                "nearest_area": [{
                    "areaName": [{"value": "Los Angeles"}],
                    "region": [{"value": "California"}],
                }],
            }
        )
        result = handle_weather("weather in los angeles")
        assert "Los Angeles" in result
        assert "72" in result
        assert "Sunny" in result

    @patch("ziggy.tools.weather.requests.get")
    def test_weather_api_failure(self, mock_get):
        from ziggy.tools.weather import handle_weather
        mock_get.return_value = MagicMock(status_code=500)
        result = handle_weather("weather in london")
        assert "couldn't" in result.lower()

    def test_weather_no_location(self):
        from ziggy.tools.weather import handle_weather
        result = handle_weather("weather")
        assert "location" in result.lower()

    def test_extract_location(self):
        from ziggy.tools.weather import _extract_location
        assert _extract_location("weather in los angeles") == "los angeles"
        assert _extract_location("weather for tokyo") == "tokyo"
        assert _extract_location("weather around paris") == "paris"

    def test_register_weather(self):
        from ziggy.tools.weather import register as register_weather
        reg = ToolRegistry()
        register_weather(reg)
        assert reg.find_tool("what's the weather") is not None
        assert reg.find_tool("temperature in paris") is not None
