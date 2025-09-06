#!/usr/bin/env python3
"""
Branding cleanup script for QCML -> QGML migration.

Finds and optionally replaces branding strings in old repositories
before cherry-picking commits into QGML.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Branding strings to find and replace
BRANDING_PATTERNS = {
    'qcml': 'qgml',
    'QCML': 'QGML',
    'Quantum Cognition Machine Learning': 'Quantum Geometric Machine Learning',
    'Quantum Computing Machine Learning': 'Quantum Geometric Machine Learning',
    'QognitiveAI': 'QGML',
    'qognitiveai': 'qgml',
    'qognitive': 'qgml',
    'Qognitive': 'QGML',
}

# File extensions to process
TEXT_EXTENSIONS = {'.py', '.md', '.rst', '.txt', '.toml', '.ini', '.yml', '.yaml', '.json'}

# Directories to exclude
EXCLUDE_DIRS = {'.git', '__pycache__', '.pytest_cache', '_build', 'node_modules', 
                '.venv', 'venv', 'env', 'dist', 'build', '.egg-info'}


def should_process_file(filepath: Path) -> bool:
    """Check if file should be processed."""
    if filepath.suffix not in TEXT_EXTENSIONS:
        return False
    
    # Check if in excluded directory
    parts = filepath.parts
    for exclude in EXCLUDE_DIRS:
        if exclude in parts:
            return False
    
    return True


def find_branding_occurrences(repo_path: Path, patterns: Dict[str, str]) -> Dict[str, List[Tuple[int, str, str]]]:
    """
    Find all occurrences of branding patterns in repository.
    
    Returns: Dict mapping filepath -> List of (line_number, original, pattern_name)
    """
    occurrences = {}
    
    for root, dirs, files in os.walk(repo_path):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            filepath = Path(root) / file
            if not should_process_file(filepath):
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                file_occurrences = []
                for line_num, line in enumerate(lines, 1):
                    for pattern, replacement in patterns.items():
                        # Case-sensitive search
                        if pattern in line:
                            file_occurrences.append((line_num, line.rstrip(), pattern))
                
                if file_occurrences:
                    rel_path = filepath.relative_to(repo_path)
                    occurrences[str(rel_path)] = file_occurrences
                    
            except Exception as e:
                print(f"Warning: Could not process {filepath}: {e}", file=sys.stderr)
                continue
    
    return occurrences


def replace_branding_in_file(filepath: Path, patterns: Dict[str, str], dry_run: bool = True) -> int:
    """
    Replace branding patterns in a file.
    
    Returns: Number of replacements made
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        replacements = 0
        
        for pattern, replacement in patterns.items():
            count = content.count(pattern)
            if count > 0:
                content = content.replace(pattern, replacement)
                replacements += count
        
        if not dry_run and content != original_content:
            with open(filepath, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(content)
        
        return replacements
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return 0


def print_occurrences(occurrences: Dict[str, List[Tuple[int, str, str]]], limit: int = 50):
    """Print found occurrences in readable format."""
    total_files = len(occurrences)
    total_occurrences = sum(len(occs) for occs in occurrences.values())
    
    print(f"\nFound {total_occurrences} occurrences in {total_files} files\n")
    print("=" * 80)
    
    count = 0
    for filepath, file_occurrences in sorted(occurrences.items()):
        if count >= limit:
            print(f"\n... and {total_files - count} more files")
            break
        
        print(f"\n{filepath}:")
        for line_num, line, pattern in file_occurrences[:10]:  # Show first 10 per file
            print(f"  Line {line_num}: {line[:70]}...")
            print(f"    Pattern: '{pattern}' -> '{BRANDING_PATTERNS[pattern]}'")
        
        if len(file_occurrences) > 10:
            print(f"  ... and {len(file_occurrences) - 10} more occurrences")
        
        count += 1


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Find and replace branding strings in repositories')
    parser.add_argument('repo_path', type=str, help='Path to repository to process')
    parser.add_argument('--find-only', action='store_true', help='Only find occurrences, do not replace')
    parser.add_argument('--replace', action='store_true', help='Actually perform replacements (default is dry-run)')
    parser.add_argument('--limit', type=int, default=50, help='Limit number of files shown in output')
    parser.add_argument('--pattern', type=str, help='Only process specific pattern (for testing)')
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}", file=sys.stderr)
        sys.exit(1)
    
    patterns = BRANDING_PATTERNS
    if args.pattern:
        patterns = {args.pattern: BRANDING_PATTERNS[args.pattern]}
    
    print(f"Scanning repository: {repo_path}")
    print(f"Looking for patterns: {list(patterns.keys())}\n")
    
    occurrences = find_branding_occurrences(repo_path, patterns)
    
    if not occurrences:
        print("No occurrences found.")
        return
    
    print_occurrences(occurrences, limit=args.limit)
    
    if args.find_only:
        print("\n[Find-only mode: No replacements made]")
        return
    
    # Ask for confirmation if not in replace mode
    if not args.replace:
        print("\n" + "=" * 80)
        print("DRY RUN MODE - No files will be modified")
        print("Use --replace to actually perform replacements")
        print("=" * 80)
    
    # Perform replacements
    total_replacements = 0
    files_modified = 0
    
    for filepath_str in occurrences.keys():
        filepath = repo_path / filepath_str
        replacements = replace_branding_in_file(filepath, patterns, dry_run=not args.replace)
        if replacements > 0:
            total_replacements += replacements
            files_modified += 1
            if args.replace:
                print(f"Modified {filepath_str}: {replacements} replacements")
    
    print(f"\n{'Would modify' if not args.replace else 'Modified'}: {files_modified} files, {total_replacements} total replacements")


if __name__ == '__main__':
    main()

