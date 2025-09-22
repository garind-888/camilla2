#!/usr/bin/env python3
"""
Build script for generating cover letter Word document with template replacement.
"""

import sys
from pathlib import Path
import subprocess
import tempfile
import json


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


def process_file_with_templates(script_dir: Path, file_path: Path) -> str:
    """Process a markdown file with template replacement."""
    if not file_path.exists():
        print(f"Error: {file_path} not found", file=sys.stderr)
        return ""
    
    content = file_path.read_text(encoding='utf-8')
    
    # Load configuration for template replacement
    config = load_config(script_dir)
    manuscript = config.get("manuscript", {})
    title = manuscript.get("title", "Title")
    subtitle = manuscript.get("subtitle", "Subtitle")
    journal = manuscript.get("journal", "Journal Name")
    editor = manuscript.get("editor", "Editor Name")
    
    # Get current date in DD.MM.YYYY format
    from datetime import datetime
    current_date = datetime.now().strftime("%d.%m.%Y")
    
    # Replace template placeholders in content
    content = content.replace("{title}", title)
    content = content.replace("{subtitle}", subtitle)
    content = content.replace("{journal}", journal)
    content = content.replace("{editor}", editor)
    content = content.replace("{date}", current_date)
    
    return content


def run_pandoc(input_content: str, output_path: Path, script_dir: Path) -> int:
    """Run pandoc to convert markdown content directly to Word."""
    try:
        # Create temporary file for the processed content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(input_content)
            temp_input_path = Path(temp_file.name)
        
        # Define paths relative to script location
        write_dir = script_dir.parent
        reference_style_dir = write_dir / "reference-style"
        templates_dir = write_dir / "templates"
        paper_dir = write_dir / "paper"
        output_dir = write_dir / "output"
        
        # Load configuration
        config = load_config(script_dir)
        citation_style = config.get("citation_style", "eurointervention.csl")
        
        # Build pandoc command
        pandoc_cmd = [
            "pandoc",
            "-f", "markdown+raw_tex+fenced_divs",
            "--lua-filter=" + str(script_dir / "pagebreak.lua"),
            "-s", str(temp_input_path),
            "--citeproc",
            "--bibliography=" + str(paper_dir / "8_references.bib"),
            "--csl=" + str(reference_style_dir / citation_style),
            "--reference-doc=" + str(templates_dir / "template-cover.docx"),
            "-o", str(output_path)
        ]
        
        # Run from the WRITE directory
        result = subprocess.run(pandoc_cmd, cwd=write_dir, capture_output=True, text=True)
        
        # Clean up temporary file
        temp_input_path.unlink()
        
        if result.returncode != 0:
            print(f"Pandoc error: {result.stderr}", file=sys.stderr)
            return result.returncode
        
        return 0
        
    except Exception as e:
        print(f"Error running pandoc: {e}", file=sys.stderr)
        return 1


def main():
    """Main function to build cover letter."""
    # Define paths
    script_dir = Path(__file__).parent
    write_dir = script_dir.parent
    paper_dir = write_dir / "paper"
    output_dir = write_dir / "output"
    output_path = output_dir / "cover-letter.docx"
    
    # Process cover letter file with templates
    cover_letter_path = paper_dir / "11_cover-letter.md"
    processed_content = process_file_with_templates(script_dir, cover_letter_path)
    
    if not processed_content.strip():
        print("Error: No content to process", file=sys.stderr)
        return 1
    
    # Convert to Word using pandoc
    result = run_pandoc(processed_content, output_path, script_dir)
    if result != 0:
        return result
    
    # Load configuration to show which style was used
    config = load_config(script_dir)
    citation_style = config.get("citation_style", "eurointervention.csl")
    print(f"Cover letter generated: {output_path}")
    print(f"Citation style used: {citation_style}")
    
    # Open the document automatically
    print("Opening cover letter in Microsoft Word...")
    try:
        subprocess.run(["open", "-a", "Microsoft Word", str(output_path)], check=True)
        print("✅ Cover letter opened in Microsoft Word!")
    except subprocess.CalledProcessError:
        print("⚠️  Could not open in Microsoft Word, trying default application...")
        try:
            subprocess.run(["open", str(output_path)], check=True)
            print("✅ Cover letter opened in default application!")
        except subprocess.CalledProcessError:
            print("⚠️  Could not open cover letter automatically. Please open manually.")
    except FileNotFoundError:
        print("⚠️  Microsoft Word not found, trying default application...")
        try:
            subprocess.run(["open", str(output_path)], check=True)
            print("✅ Cover letter opened in default application!")
        except subprocess.CalledProcessError:
            print("⚠️  Could not open cover letter automatically. Please open manually.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
