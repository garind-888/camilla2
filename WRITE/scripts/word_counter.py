#!/usr/bin/env python3
"""
Word counter utility for medical papers.
Counts words in abstract and main content sections, plus figures and tables.
"""

import re
from pathlib import Path
from typing import Dict, Tuple


def clean_text_for_counting(text: str) -> str:
    """Clean text for accurate word counting by removing markdown syntax and LaTeX."""
    # Remove markdown headers
    text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
    
    # Remove markdown bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove LaTeX commands and braces
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    
    # Remove LaTeX page breaks
    text = re.sub(r'\\newpage', '', text)
    text = re.sub(r'\\pagebreak', '', text)
    
    # Remove markdown custom styles
    text = re.sub(r':::.*?:::', '', text, flags=re.DOTALL)
    
    # Remove citations (keep only the text)
    text = re.sub(r'\[@[^\]]+\]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def count_words_in_text(text: str) -> int:
    """Count words in cleaned text."""
    cleaned = clean_text_for_counting(text)
    if not cleaned:
        return 0
    
    # Split by whitespace and count non-empty strings
    words = [word for word in cleaned.split() if word.strip()]
    return len(words)


def count_figures_and_tables(figures_content: str, tables_content: str) -> Tuple[int, int]:
    """Count figures and tables based on content."""
    # Count figures - look for figure headers or image references
    figure_count = 0
    
    # Count figure headers like "## Figure 1" or "**Figure 1**"
    figure_headers = re.findall(r'(?:##\s+Figure\s+\d+|#\s+Figure\s+\d+|\*\*Figure\s+\d+\*\*)', figures_content, re.IGNORECASE)
    figure_count += len(figure_headers)
    
    # Count image references like ![alt](path)
    image_refs = re.findall(r'!\[[^\]]*\]\([^)]+\)', figures_content)
    figure_count += len(image_refs)
    
    # If no explicit figures found but content exists, assume at least 1
    if figures_content.strip() and figure_count == 0:
        figure_count = 1
    
    # Count tables - look for table headers
    table_count = 0
    
    # Count table headers like "## Table 1" or "**Table 1**"
    table_headers = re.findall(r'(?:##\s+Table\s+\d+|#\s+Table\s+\d+|\*\*Table\s+\d+\*\*)', tables_content, re.IGNORECASE)
    table_count += len(table_headers)
    
    # If no explicit tables found but content exists, assume at least 1
    if tables_content.strip() and table_count == 0:
        table_count = 1
    
    return figure_count, table_count


def count_paper_content(paper_dir: Path) -> Dict[str, int]:
    """Count words and content for the entire paper."""
    counts = {
        'abstract_words': 0,
        'main_words': 0,
        'total_words': 0,
        'figures': 0,
        'tables': 0
    }
    
    try:
        # Count abstract words
        abstract_file = paper_dir / "1_abstract.md"
        if abstract_file.exists():
            abstract_content = abstract_file.read_text(encoding='utf-8')
            counts['abstract_words'] = count_words_in_text(abstract_content)
        
        # Count main content words (intro + methods + results + discussion)
        main_sections = ["2_introduction.md", "3_methods.md", "4_results.md", "5_discussion.md"]
        main_words = 0
        
        for section_file in main_sections:
            section_path = paper_dir / section_file
            if section_path.exists():
                section_content = section_path.read_text(encoding='utf-8')
                main_words += count_words_in_text(section_content)
        
        counts['main_words'] = main_words
        counts['total_words'] = counts['abstract_words'] + main_words
        
        # Count figures and tables
        figures_file = paper_dir / "6_figures.md"
        tables_file = paper_dir / "7_table.md"
        
        figures_content = ""
        tables_content = ""
        
        if figures_file.exists():
            figures_content = figures_file.read_text(encoding='utf-8')
        if tables_file.exists():
            tables_content = tables_file.read_text(encoding='utf-8')
        
        counts['figures'], counts['tables'] = count_figures_and_tables(figures_content, tables_content)
        
    except Exception as e:
        print(f"Error counting content: {e}", file=sys.stderr)
    
    return counts


def update_title_page_with_counts(paper_dir: Path, counts: Dict[str, int]) -> None:
    """Update the title page with word and content counts."""
    title_file = paper_dir / "0_title-page.md"
    
    if not title_file.exists():
        return
    
    try:
        content = title_file.read_text(encoding='utf-8')
        
        # Remove all existing count fields first
        content = re.sub(r'\*\*Abstract word count:\*\*\s*[^\n]*\n?', '', content)
        content = re.sub(r'\*\*Word count:\*\*\s*[^\n]*\n?', '', content)
        content = re.sub(r'\*\*Figures:\*\*\s*[^\n]*\n?', '', content)
        content = re.sub(r'\*\*Tables:\*\*\s*[^\n]*\n?', '', content)
        
        # Add all fields in the correct order after Keywords
        content = re.sub(
            r'(\*\*Keywords:\*\*\s*[^\n]*)',
            f'\\1\n\n**Abstract word count:** {counts["abstract_words"]}\n\n**Word count:** {counts["total_words"]}\n\n**Figures:** {counts["figures"]}\n\n**Tables:** {counts["tables"]}',
            content
        )
        
        # Write back to file
        title_file.write_text(content, encoding='utf-8')
        
    except Exception as e:
        print(f"Error updating title page: {e}", file=sys.stderr)


def main():
    """Main function to count and update title page."""
    import sys
    
    script_dir = Path(__file__).parent
    paper_dir = script_dir.parent / "paper"
    
    # Count content
    counts = count_paper_content(paper_dir)
    
    # Update title page
    update_title_page_with_counts(paper_dir, counts)
    
    # Print summary
    print(f"Content counts updated:")
    print(f"  Abstract: {counts['abstract_words']} words")
    print(f"  Main content: {counts['main_words']} words") 
    print(f"  Total: {counts['total_words']} words")
    print(f"  Figures: {counts['figures']}")
    print(f"  Tables: {counts['tables']}")


if __name__ == "__main__":
    main()
