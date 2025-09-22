#!/usr/bin/env python3
"""
Build script for generating Word document from 10 separate markdown files.
Combines: title-page.md, abstract.md, introduction.md, methods.md, results.md, 
discussion.md, figures.md, table.md, end.md and converts directly to Word format using pandoc.
Ensures reference numbering is preserved in the final document.
"""

import sys
from pathlib import Path
import subprocess
import tempfile
import json
from word_counter import count_paper_content, update_title_page_with_counts
from process_templates import main as process_templates


def read_file_content(file_path: Path) -> str:
    """Read file content if it exists, otherwise return empty string."""
    if file_path.exists():
        return file_path.read_text(encoding='utf-8')
    return ""


def load_config(script_dir: Path) -> dict:
    """Load configuration from config.json file."""
    config_path = script_dir.parent / "config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: config.json not found, using default settings", file=sys.stderr)
        return {
            "citation_style": "eurointervention.csl",
            "build_settings": {
                "number_sections": True,
                "toc_depth": 3,
                "template": "template.docx"
            }
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing config.json: {e}", file=sys.stderr)
        sys.exit(1)


def combine_all_sections(script_dir: Path) -> str:
    """Combine all 10 sections in the correct order with template replacement."""
    # Define paths relative to script location
    paper_dir = script_dir.parent / "paper"
    
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
    
    sections = [
        "0_title-page.md",
        "1_abstract.md",
        "2_introduction.md",
        "3_methods.md",
        "4_results.md", 
        "5_discussion.md",
        "6_figures.md",
        "7_table.md",
        "9_end.md"
    ]
    
    combined_content = []
    
    for section_file in sections:
        section_path = paper_dir / section_file
        content = read_file_content(section_path)
        if content.strip():
            # Replace template placeholders in content
            content = content.replace("{title}", title)
            content = content.replace("{subtitle}", subtitle)
            content = content.replace("{journal}", journal)
            content = content.replace("{editor}", editor)
            content = content.replace("{date}", current_date)
            
            combined_content.append(content.strip())
            # Add page break after each section (except the last one)
            combined_content.append("\\newpage")
    
    # Remove the last page break to avoid extra blank page at the end
    if combined_content and combined_content[-1] == "\\newpage":
        combined_content.pop()
    
    return "\n\n".join(combined_content)


def process_single_file_with_templates(script_dir: Path, file_path: Path) -> str:
    """Process a single markdown file with template replacement."""
    if not file_path.exists():
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
        # Create temporary file for the combined content
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
        build_settings = config.get("build_settings", {})
        
        # Build pandoc command with configurable settings
        pandoc_cmd = [
            "pandoc",
            "-f", "markdown+raw_tex+fenced_divs",
            "--lua-filter=" + str(script_dir / "pagebreak.lua"),
            "-s", str(temp_input_path),
            "--citeproc",
            "--bibliography=" + str(paper_dir / "8_references.bib"),
            "--csl=" + str(reference_style_dir / citation_style),
            "--reference-doc=" + str(templates_dir / build_settings.get("template", "template.docx"))
        ]
        
        # Add optional build settings (number-sections disabled to avoid double numbering)
        # Note: Section numbering is handled by the markdown content itself
        if build_settings.get("toc_depth"):
            pandoc_cmd.extend(["--toc-depth=" + str(build_settings["toc_depth"])])
        
        pandoc_cmd.extend(["-o", str(output_path)])
        
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
    # Define paths
    script_dir = Path(__file__).parent
    write_dir = script_dir.parent
    paper_dir = write_dir / "paper"
    output_dir = write_dir / "output"
    output_path = output_dir / "paper.docx"
    
    # Check that required files exist
    required_files = ["2_introduction.md", "3_methods.md", "4_results.md", "5_discussion.md"]
    missing_files = []
    for required_file in required_files:
        if not (paper_dir / required_file).exists():
            missing_files.append(required_file)
    
    if missing_files:
        print(f"Error: Required files not found: {', '.join(missing_files)}", file=sys.stderr)
        return 1
    
    # Update title page with word and content counts
    print("Updating word counts and content statistics...")
    counts = count_paper_content(paper_dir)
    update_title_page_with_counts(paper_dir, counts)
    print(f"  Abstract: {counts['abstract_words']} words")
    print(f"  Main content: {counts['main_words']} words")
    print(f"  Total: {counts['total_words']} words")
    print(f"  Figures: {counts['figures']}, Tables: {counts['tables']}")
    
    # Combine all sections
    combined_content = combine_all_sections(script_dir)
    
    if not combined_content.strip():
        print("Error: No content to process", file=sys.stderr)
        return 1
    
    # Convert directly to Word using pandoc
    result = run_pandoc(combined_content, output_path, script_dir)
    if result != 0:
        return result
    
    # Load configuration to show which style was used
    config = load_config(script_dir)
    citation_style = config.get("citation_style", "eurointervention.csl")
    print(f"Word document generated: {output_path}")
    print(f"Citation style used: {citation_style}")
    
    # Open the document automatically
    print("Opening document in Microsoft Word...")
    try:
        subprocess.run(["open", "-a", "Microsoft Word", str(output_path)], check=True)
        print("✅ Document opened in Microsoft Word!")
    except subprocess.CalledProcessError:
        print("⚠️  Could not open in Microsoft Word, trying default application...")
        try:
            subprocess.run(["open", str(output_path)], check=True)
            print("✅ Document opened in default application!")
        except subprocess.CalledProcessError:
            print("⚠️  Could not open document automatically. Please open manually.")
    except FileNotFoundError:
        print("⚠️  Microsoft Word not found, trying default application...")
        try:
            subprocess.run(["open", str(output_path)], check=True)
            print("✅ Document opened in default application!")
        except subprocess.CalledProcessError:
            print("⚠️  Could not open document automatically. Please open manually.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
