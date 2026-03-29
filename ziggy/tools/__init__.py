"""Tool registry for Ziggy.

Tools are simple callables that handle specific query types. The registry
provides a central place to register and discover tools, making it easy
to add new capabilities.

Future: Each tool can expose a schema for LLM tool-calling integration.
"""


class Tool:
    """A registered tool that can handle voice queries."""

    def __init__(self, name, description, keywords, handler):
        self.name = name
        self.description = description
        self.keywords = keywords  # words/phrases that trigger this tool
        self.handler = handler

    def matches(self, text_lower):
        """Check if this tool should handle the given text."""
        return any(kw in text_lower for kw in self.keywords)


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self):
        self._tools = []

    def register(self, name, description, keywords, handler):
        self._tools.append(Tool(name, description, keywords, handler))

    def find_tool(self, text_lower):
        """Return the first tool that matches the text, or None."""
        for tool in self._tools:
            if tool.matches(text_lower):
                return tool
        return None

    def list_tools(self):
        return [(t.name, t.description) for t in self._tools]
