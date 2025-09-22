#!/usr/bin/env python3
"""
Deduplicate BibTeX entries in a .bib file by title.

Behaviour:
- Detect duplicates by normalized title (casefolded, diacritics removed, LaTeX markup reduced, whitespace collapsed).
- Keep the "bigger" entry (longer raw entry string). On ties, keep the earliest.
- Make a timestamped backup next to the original before writing changes.

Usage:
  python3 WRITE/deduplicate_references_bib.py [path/to/references.bib]

Defaults to the repository's WRITE/references.bib when no path is provided.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
from pathlib import Path
import re
import sys
import unicodedata


DEFAULT_BIB_PATH = \
    "/Users/doriangarin/Documents/Médecine/recherche/PROJECT/WRITE/references.bib"


def read_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def write_with_backup(file_path: Path, new_text: str) -> Path:
    timestamp = _dt.datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = file_path.with_name(f"{file_path.name}.bak-{timestamp}")
    # Write backup
    backup_path.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")
    # Write new content
    file_path.write_text(new_text, encoding="utf-8")
    return backup_path


def split_bib_entries(text: str) -> list[str]:
    """Split a .bib file into a list of entry strings. Uses brace balancing
    starting at each '@' until the matching closing '}'.
    Also returns non-entry text (e.g., leading/trailing whitespace) as-is if any.
    """
    entries: list[str] = []
    i = 0
    n = len(text)
    # Skip leading whitespace/comments not part of entries
    while i < n:
        at = text.find('@', i)
        if at == -1:
            remainder = text[i:].strip()
            if remainder:
                entries.append(text[i:])
            break
        # Capture any interstitial text before the next entry
        if at > i and text[i:at].strip():
            entries.append(text[i:at])
        # Find the opening '{' of this entry
        brace_open = text.find('{', at)
        if brace_open == -1:
            # Malformed remainder; include and stop
            entries.append(text[at:])
            break
        depth = 1
        j = brace_open + 1
        while j < n and depth > 0:
            ch = text[j]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            j += 1
        # j is one past the closing '}' or end-of-text
        entries.append(text[at:j])
        i = j
    # Normalize entries by stripping purely whitespace-only chunks
    entries = [e for e in entries if e.strip()]
    return entries


def _find_field_start(entry_lower: str, field_name_lower: str) -> int:
    # Match at start of line with optional leading spaces: ^\s*fieldname\s*=\s*
    pattern = re.compile(r"(?m)^\s*" + re.escape(field_name_lower) + r"\s*=\s*")
    m = pattern.search(entry_lower)
    return -1 if not m else m.end()


def _extract_balanced_braces_value(s: str, start_index: int) -> tuple[str, int]:
    # s[start_index] should be '{'
    assert s[start_index] == '{'
    depth = 1
    i = start_index + 1
    n = len(s)
    while i < n and depth > 0:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        i += 1
    # return content inside braces and index after closing brace
    return s[start_index + 1:i - 1], i


def _extract_quoted_value(s: str, start_index: int) -> tuple[str, int]:
    # s[start_index] should be '"'
    assert s[start_index] == '"'
    i = start_index + 1
    n = len(s)
    out_chars: list[str] = []
    while i < n:
        ch = s[i]
        if ch == '"':
            # Check for escaped quote
            if i > 0 and s[i - 1] == '\\':
                out_chars[-1] = '"'  # replace the backslash marker with quote
                i += 1
                continue
            i += 1
            break
        out_chars.append(ch)
        i += 1
    return ''.join(out_chars), i


def extract_bib_field(entry: str, field_name: str) -> str | None:
    entry_lower = entry.lower()
    idx = _find_field_start(entry_lower, field_name.lower())
    if idx == -1:
        return None
    # Find value starting point in original entry (same index)
    # Skip whitespace
    while idx < len(entry) and entry[idx].isspace():
        idx += 1
    if idx >= len(entry):
        return None
    ch = entry[idx]
    if ch == '{':
        val, _end = _extract_balanced_braces_value(entry, idx)
        return val.strip()
    if ch == '"':
        val, _end = _extract_quoted_value(entry, idx)
        return val.strip()
    # Bare word (macro): read until comma or newline
    m = re.match(r"([^,\n\r]+)", entry[idx:])
    if m:
        return m.group(1).strip()
    return None


def _strip_latex_commands_keep_args(s: str) -> str:
    # Iteratively replace commands like \cmd{arg} -> arg
    pattern_cmd_arg = re.compile(r"\\[a-zA-Z]+\s*\{([^{}]*)\}")
    prev = None
    while prev != s:
        prev = s
        s = pattern_cmd_arg.sub(r"\1", s)
    # Remove remaining commands like \alpha or \'e, \"o, etc.
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"\\[`'\^\"~=.]{1}\s*\{?\s*([A-Za-z])\s*\}?", r"\1", s)
    return s


def normalize_title(raw_title: str) -> str:
    s = raw_title
    # Remove LaTeX commands while preserving arguments
    s = _strip_latex_commands_keep_args(s)
    # Remove curly braces used for capitalization protection
    s = s.replace('{', ' ').replace('}', ' ')
    # Unicode normalize and strip diacritics
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # Lowercase and keep alphanumerics/spaces
    s = s.casefold()
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def deduplicate_entries(entries: list[str]) -> tuple[list[str], dict[str, int]]:
    """Return deduplicated entries and statistics.

    Stats keys:
      - original_count
      - kept_count
      - removed_count
      - groups_deduplicated
    """
    # Map normalized title -> (best_length, best_index, best_entry)
    best_by_title: dict[str, tuple[int, int, str]] = {}
    title_by_index: dict[int, str] = {}
    for idx, entry in enumerate(entries):
        title = extract_bib_field(entry, 'title')
        if not title:
            continue
        norm = normalize_title(title)
        title_by_index[idx] = norm
        length = len(entry)
        if norm not in best_by_title:
            best_by_title[norm] = (length, idx, entry)
        else:
            best_len, best_idx, best_entry = best_by_title[norm]
            if length > best_len or (length == best_len and idx < best_idx):
                best_by_title[norm] = (length, idx, entry)

    # Build output preserving original order
    emitted_titles: set[str] = set()
    output: list[str] = []
    for idx, entry in enumerate(entries):
        norm = title_by_index.get(idx)
        if norm is None:
            output.append(entry)
            continue
        best_len, best_idx, best_entry = best_by_title[norm]
        if idx == best_idx and norm not in emitted_titles:
            output.append(entry)
            emitted_titles.add(norm)
        # else: skip duplicate

    original_count = len(entries)
    kept_count = len(output)
    groups_deduplicated = sum(1 for _, (l, _i, _e) in best_by_title.items())
    removed_count = original_count - kept_count
    stats = {
        'original_count': original_count,
        'kept_count': kept_count,
        'removed_count': removed_count,
        'groups_deduplicated': groups_deduplicated,
    }
    return output, stats


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Deduplicate BibTeX entries by title, keeping the longest entry.")
    parser.add_argument("bibfile", nargs='?', default=DEFAULT_BIB_PATH, help="Path to the .bib file (defaults to WRITE/references.bib)")
    args = parser.parse_args(argv)

    bib_path = Path(args.bibfile)
    if not bib_path.exists():
        print(f"Error: file not found: {bib_path}", file=sys.stderr)
        return 2
    try:
        text = read_text(bib_path)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 2

    entries = split_bib_entries(text)
    if not entries:
        print("No entries found. Nothing to do.")
        return 0

    deduped_entries, stats = deduplicate_entries(entries)
    if stats['removed_count'] <= 0:
        print("No duplicates by title detected. File unchanged.")
        return 0

    new_text = "\n\n".join(deduped_entries)
    if not new_text.endswith("\n"):
        new_text += "\n"

    try:
        backup_path = write_with_backup(bib_path, new_text)
    except Exception as e:
        print(f"Error writing updated file: {e}", file=sys.stderr)
        return 2

    print(f"Backup saved to: {backup_path}")
    print(f"Entries: {stats['original_count']} -> {stats['kept_count']} (removed {stats['removed_count']})")
    print(f"Duplicate groups by title: {stats['groups_deduplicated']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


