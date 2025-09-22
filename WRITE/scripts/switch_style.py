#!/usr/bin/env python3
"""
Helper script to easily switch citation styles in config.json
"""

import sys
import json
from pathlib import Path


def list_available_styles():
    """List all available citation styles."""
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("Available citation styles:")
        for style, description in config.get("available_styles", {}).items():
            current = " (CURRENT)" if style == config.get("citation_style") else ""
            print(f"  {style}{current}: {description}")
            
    except FileNotFoundError:
        print("Error: config.json not found")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error parsing config.json: {e}")
        return 1
    
    return 0


def switch_style(style_name):
    """Switch to a different citation style."""
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check if style exists
        available_styles = config.get("available_styles", {})
        if style_name not in available_styles:
            print(f"Error: Style '{style_name}' not found in available styles")
            print("Available styles:", ", ".join(available_styles.keys()))
            return 1
        
        # Update the citation style
        old_style = config.get("citation_style", "unknown")
        config["citation_style"] = style_name
        
        # Write back to file
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"Citation style changed from '{old_style}' to '{style_name}'")
        print(f"Description: {available_styles[style_name]}")
        
    except FileNotFoundError:
        print("Error: config.json not found")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error parsing config.json: {e}")
        return 1
    
    return 0


def main():
    if len(sys.argv) == 1:
        # No arguments - list available styles
        return list_available_styles()
    elif len(sys.argv) == 2:
        # One argument - switch to that style
        style_name = sys.argv[1]
        return switch_style(style_name)
    else:
        print("Usage:")
        print(f"  {sys.argv[0]}                    # List available styles")
        print(f"  {sys.argv[0]} <style_name>       # Switch to style")
        print()
        print("Examples:")
        print(f"  {sys.argv[0]} eurointervention.csl")
        print(f"  {sys.argv[0]} cci.csl")
        return 1


if __name__ == "__main__":
    sys.exit(main())
