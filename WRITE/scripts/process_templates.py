#!/usr/bin/env python3
"""
Template processor for replacing placeholders in markdown files with config values.
Processes title-page.md, supplements.md, and cover-letter.md files.
"""

import json
import re
from pathlib import Path
from datetime import datetime
import sys


def load_config(script_dir: Path) -> dict:
    """Load configuration from config.json file."""
    config_path = script_dir.parent / "config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: config.json not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing config.json: {e}", file=sys.stderr)
        sys.exit(1)


def get_current_date_dd_mm_yyyy() -> str:
    """Get current date in DD.MM.YYYY format."""
    now = datetime.now()
    return now.strftime("%d.%m.%Y")


def process_template_file(file_path: Path, config: dict) -> None:
    """Process a single template file, replacing placeholders with config values."""
    if not file_path.exists():
        print(f"Warning: {file_path} not found, skipping...", file=sys.stderr)
        return
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Get manuscript metadata from config
        manuscript = config.get("manuscript", {})
        title = manuscript.get("title", "Title")
        subtitle = manuscript.get("subtitle", "Subtitle")
        journal = manuscript.get("journal", "Journal Name")
        editor = manuscript.get("editor", "Editor Name")
        
        # Get current date
        current_date = get_current_date_dd_mm_yyyy()
        
        # Replace placeholders
        replacements = {
            "{title}": title,
            "{subtitle}": subtitle,
            "{journal}": journal,
            "{editor}": editor,
            "{date}": current_date
        }
        
        # Apply replacements
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
        
        # Write back to file
        file_path.write_text(content, encoding='utf-8')
        print(f"Processed: {file_path.name}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)


def main():
    """Main function to process all template files."""
    script_dir = Path(__file__).parent
    paper_dir = script_dir.parent / "paper"
    
    # Load configuration
    config = load_config(script_dir)
    
    # List of files to process
    template_files = [
        "0_title-page.md",
        "10_supplements.md", 
        "11_cover-letter.md"
    ]
    
    print("Processing template files...")
    
    for filename in template_files:
        file_path = paper_dir / filename
        process_template_file(file_path, config)
    
    print("Template processing completed!")


if __name__ == "__main__":
    main()
