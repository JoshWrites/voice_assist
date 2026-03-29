#!/usr/bin/env python3
"""Ziggy Voice Assistant — entry point."""

from ziggy.app import VoiceAssistant


def main():
    print("Ziggy Voice Assistant v2")
    print("Local-first AI assistant with privacy protection")
    print("=" * 50)

    try:
        assistant = VoiceAssistant()
        assistant.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
