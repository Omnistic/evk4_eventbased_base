"""
core/config.py

User preference persistence.

Saves and loads user settings (e.g. dark mode preference) to a JSON file
in the user's home directory so they persist across app restarts.
"""

import json
from pathlib import Path

CONFIG_FILE = Path.home() / ".evk4_dashboard_config.json"
DEFAULT_CONFIG = {"dark_mode": True}


def load_config() -> dict:
    """Load persisted user config, returning defaults if not found or invalid."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_CONFIG.copy()


def save_config(config: dict) -> None:
    """Save user config to disk."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except OSError as e:
        print(f"Warning: Could not save config: {e}")