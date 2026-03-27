import re
import math
from datetime import datetime
from typing import Dict, Any
from .interfaces import CommandHandlerInterface, CommandResult


class TimeHandler(CommandHandlerInterface):
    """Handler for time-related queries"""
    
    def can_handle(self, text: str) -> bool:
        """Check if this is a time query"""
        text_lower = text.lower()
        time_keywords = ['time', 'clock', 'what time']
        return any(keyword in text_lower for keyword in time_keywords)
    
    def handle(self, text: str) -> CommandResult:
        """Get current time"""
        now = datetime.now()
        response = f"The time is {now.strftime('%I:%M %p')}"
        return CommandResult(
            response=response,
            command_type="time",
            metadata={"timestamp": now.isoformat()}
        )
    
    def get_command_type(self) -> str:
        return "time"


class DateHandler(CommandHandlerInterface):
    """Handler for date-related queries"""
    
    def can_handle(self, text: str) -> bool:
        """Check if this is a date query"""
        text_lower = text.lower()
        date_keywords = ['date', 'today', 'what day']
        return any(keyword in text_lower for keyword in date_keywords)
    
    def handle(self, text: str) -> CommandResult:
        """Get current date"""
        now = datetime.now()
        response = f"Today is {now.strftime('%A, %B %d, %Y')}"
        return CommandResult(
            response=response,
            command_type="date",
            metadata={"date": now.date().isoformat()}
        )
    
    def get_command_type(self) -> str:
        return "date"


class MathHandler(CommandHandlerInterface):
    """Handler for mathematical calculations"""
    
    def can_handle(self, text: str) -> bool:
        """Check if this is a math query"""
        text_lower = text.lower()
        math_keywords = ['plus', 'minus', 'times', 'divided', 'calculate', 'math']
        has_math_keyword = any(keyword in text_lower for keyword in math_keywords)
        
        # Also check for numbers and operators
        has_numbers = bool(re.search(r'\d+', text))
        math_operators = ['+', '-', '*', '/', 'x', '×', '÷']
        has_operator = any(op in text for op in math_operators)
        
        return has_math_keyword and (has_numbers or has_operator)
    
    def handle(self, text: str) -> CommandResult:
        """Perform mathematical calculation"""
        try:
            result = self._parse_and_calculate(text)
            return CommandResult(
                response=result,
                command_type="math",
                metadata={"original_query": text}
            )
        except Exception as e:
            return CommandResult(
                response="I couldn't understand that calculation. Please try rephrasing.",
                command_type="math",
                success=False,
                metadata={"error": str(e)}
            )
    
    def _parse_and_calculate(self, text: str) -> str:
        """Parse text and perform calculation"""
        text_lower = text.lower()
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if len(numbers) < 2:
            raise ValueError("Need at least two numbers")
        
        num1 = float(numbers[0])
        num2 = float(numbers[1])
        
        # Determine operation
        if any(word in text_lower for word in ['plus', 'add', '+']):
            result = num1 + num2
            operation = "plus"
        elif any(word in text_lower for word in ['minus', 'subtract', '-']):
            result = num1 - num2
            operation = "minus"
        elif any(word in text_lower for word in ['times', 'multiply', '*', 'x', '×']):
            result = num1 * num2
            operation = "times"
        elif any(word in text_lower for word in ['divided', 'divide', '/', '÷']):
            if num2 == 0:
                raise ValueError("Cannot divide by zero")
            result = num1 / num2
            operation = "divided by"
        else:
            raise ValueError("Unknown operation")
        
        # Format result
        if result == int(result):
            result = int(result)
        
        return f"{num1} {operation} {num2} equals {result}"
    
    def get_command_type(self) -> str:
        return "math"


class ConversionHandler(CommandHandlerInterface):
    """Handler for unit conversions"""
    
    CONVERSIONS = {
        # Temperature
        'fahrenheit_to_celsius': lambda f: (f - 32) * 5/9,
        'celsius_to_fahrenheit': lambda c: c * 9/5 + 32,
        
        # Length
        'feet_to_meters': lambda ft: ft * 0.3048,
        'meters_to_feet': lambda m: m / 0.3048,
        'miles_to_kilometers': lambda mi: mi * 1.60934,
        'kilometers_to_miles': lambda km: km / 1.60934,
        
        # Weight
        'pounds_to_kilograms': lambda lb: lb * 0.453592,
        'kilograms_to_pounds': lambda kg: kg / 0.453592,
    }
    
    def can_handle(self, text: str) -> bool:
        """Check if this is a conversion query"""
        text_lower = text.lower()
        unit_keywords = ['fahrenheit', 'celsius', 'feet', 'meters', 'miles', 'kilometers', 'pounds', 'kilograms']
        has_unit = any(keyword in text_lower for keyword in unit_keywords)
        has_number = bool(re.search(r'\d+', text))
        return has_unit and has_number
    
    def handle(self, text: str) -> CommandResult:
        """Perform unit conversion"""
        try:
            result = self._parse_and_convert(text)
            return CommandResult(
                response=result,
                command_type="conversion",
                metadata={"original_query": text}
            )
        except Exception as e:
            return CommandResult(
                response="I couldn't understand that conversion. Please try rephrasing.",
                command_type="conversion",
                success=False,
                metadata={"error": str(e)}
            )
    
    def _parse_and_convert(self, text: str) -> str:
        """Parse and perform conversion"""
        text_lower = text.lower()
        
        # Extract number
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if not numbers:
            raise ValueError("No number found")
        
        value = float(numbers[0])
        
        # Determine conversion type
        if 'fahrenheit' in text_lower and 'celsius' in text_lower:
            if text_lower.find('fahrenheit') < text_lower.find('celsius'):
                result = self.CONVERSIONS['fahrenheit_to_celsius'](value)
                return f"{value} degrees Fahrenheit is {result:.1f} degrees Celsius"
            else:
                result = self.CONVERSIONS['celsius_to_fahrenheit'](value)
                return f"{value} degrees Celsius is {result:.1f} degrees Fahrenheit"
        
        elif 'feet' in text_lower and 'meter' in text_lower:
            if text_lower.find('feet') < text_lower.find('meter'):
                result = self.CONVERSIONS['feet_to_meters'](value)
                return f"{value} feet is {result:.2f} meters"
            else:
                result = self.CONVERSIONS['meters_to_feet'](value)
                return f"{value} meters is {result:.2f} feet"
        
        elif 'mile' in text_lower and 'kilometer' in text_lower:
            if text_lower.find('mile') < text_lower.find('kilometer'):
                result = self.CONVERSIONS['miles_to_kilometers'](value)
                return f"{value} miles is {result:.2f} kilometers"
            else:
                result = self.CONVERSIONS['kilometers_to_miles'](value)
                return f"{value} kilometers is {result:.2f} miles"
        
        elif 'pound' in text_lower and 'kilogram' in text_lower:
            if text_lower.find('pound') < text_lower.find('kilogram'):
                result = self.CONVERSIONS['pounds_to_kilograms'](value)
                return f"{value} pounds is {result:.2f} kilograms"
            else:
                result = self.CONVERSIONS['kilograms_to_pounds'](value)
                return f"{value} kilograms is {result:.2f} pounds"
        
        else:
            raise ValueError("Unknown conversion type")
    
    def get_command_type(self) -> str:
        return "conversion"