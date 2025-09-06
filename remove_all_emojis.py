#!/usr/bin/env python3
"""
Script to identify and optionally remove ALL emojis from the QGML codebase.
This is more reliable than sed commands that break with emojis.
"""

import os
import re
import glob
from collections import defaultdict

def find_emojis_in_text(text):
    """Find all emojis in text using Unicode ranges."""
    # Comprehensive emoji detection regex
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U0001F018-\U0001F0FF"  # playing cards
        "\U0001F200-\U0001F2FF"  # enclosed CJK letters and months
        "\U0001F300-\U0001F5FF"  # miscellaneous symbols and pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport and map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # geometric shapes extended
        "\U0001F800-\U0001F8FF"  # supplemental arrows-C
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U0001FB00-\U0001FBFF"  # symbols for legacy computing
        "\U0001FC00-\U0001FCFF"  # symbols for legacy computing
        "\U0001FD00-\U0001FDFF"  # symbols for legacy computing
        "\U0001FE00-\U0001FEFF"  # variation selectors
        "\U0001FF00-\U0001FFFF"  # symbols for legacy computing
        "]+", 
        flags=re.UNICODE
    )
    
    # Find all emojis
    emojis = emoji_pattern.findall(text)
    return emojis

def remove_emojis_from_text(text):
    """Remove all emojis from text using Unicode ranges."""
    # Comprehensive emoji removal regex
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U0001F018-\U0001F0FF"  # playing cards
        "\U0001F200-\U0001F2FF"  # enclosed CJK letters and months
        "\U0001F300-\U0001F5FF"  # miscellaneous symbols and pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport and map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # geometric shapes extended
        "\U0001F800-\U0001F8FF"  # supplemental arrows-C
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U0001FB00-\U0001FBFF"  # symbols for legacy computing
        "\U0001FC00-\U0001FCFF"  # symbols for legacy computing
        "\U0001FD00-\U0001FDFF"  # symbols for legacy computing
        "\U0001FE00-\U0001FEFF"  # variation selectors
        "\U0001FF00-\U0001FFFF"  # symbols for legacy computing
        "]+", 
        flags=re.UNICODE
    )
    
    # Remove emojis but preserve ALL formatting including indentation
    cleaned = emoji_pattern.sub('', text)
    # Only clean up multiple spaces that are NOT indentation
    lines = cleaned.split('\n')
    cleaned_lines = []
    for line in lines:
        # Find leading whitespace (indentation)
        leading_whitespace = re.match(r'^(\s*)', line).group(1)
        # Get content after indentation
        content = line[len(leading_whitespace):]
        # Clean up multiple spaces in content only, preserve indentation
        cleaned_content = re.sub(r'[ \t]+', ' ', content)
        # Reconstruct line with original indentation
        cleaned_line = leading_whitespace + cleaned_content
        cleaned_lines.append(cleaned_line)
    cleaned = '\n'.join(cleaned_lines)
    
    return cleaned

def inventory_file(file_path):
    """Inventory emojis in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        emojis = find_emojis_in_text(content)
        
        if emojis:
            # Count unique emojis
            unique_emojis = set(emojis)
            emoji_counts = defaultdict(int)
            for emoji in emojis:
                emoji_counts[emoji] += 1
            
            return {
                'file': file_path,
                'total_emojis': len(emojis),
                'unique_emojis': len(unique_emojis),
                'emoji_list': list(unique_emojis),
                'emoji_counts': dict(emoji_counts)
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_file(file_path, dry_run=True):
    """Process a single file to remove emojis."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        cleaned_content = remove_emojis_from_text(content)
        
        if original_content != cleaned_content:
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                print(f"Cleaned: {file_path}")
            else:
                print(f"Would clean: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to inventory and optionally clean all files in the QGML directory."""
    import sys
    
    # Check command line arguments
    dry_run = True
    test_file = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--remove':
            dry_run = False
            print("WARNING: Running in REMOVE mode - emojis will be permanently deleted!")
            response = input("Are you sure? Type 'yes' to continue: ")
            if response.lower() != 'yes':
                print("Aborted.")
                return
        elif sys.argv[1] == '--test':
            if len(sys.argv) > 2:
                test_file = sys.argv[2]
            else:
                print("Usage: python remove_all_emojis.py --test <filename>")
                return
        elif sys.argv[1].startswith('--'):
            print("Usage:")
            print("  python remove_all_emojis.py                    # Inventory only")
            print("  python remove_all_emojis.py --remove           # Remove all emojis")
            print("  python remove_all_emojis.py --test <filename>  # Test on single file")
            return
        else:
            test_file = sys.argv[1]  # Treat as filename for testing
    
    # Handle single file test mode
    if test_file:
        if not os.path.exists(test_file):
            print(f"Error: File '{test_file}' not found!")
            return
        
        print("=" * 80)
        print(f"TESTING EMOJI REMOVAL ON: {test_file}")
        print("=" * 80)
        
        # Show before
        inventory = inventory_file(test_file)
        if inventory:
            print(f"BEFORE:")
            print(f"  Total emojis: {inventory['total_emojis']}")
            print(f"  Unique emojis: {inventory['unique_emojis']}")
            print(f"  Emojis: {' '.join(inventory['emoji_list'])}")
            print()
            
            # Show what would be removed
            with open(test_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            cleaned_content = remove_emojis_from_text(original_content)
            
            print("PREVIEW OF CHANGES:")
            print("-" * 40)
            # Show first few lines with context
            lines = original_content.split('\n')
            cleaned_lines = cleaned_content.split('\n')
            
            for i, (orig, clean) in enumerate(zip(lines[:10], cleaned_lines[:10])):
                if orig != clean:
                    print(f"Line {i+1}:")
                    print(f"  BEFORE: {repr(orig)}")
                    print(f"  AFTER:  {repr(clean)}")
                    print()
            
            if len(lines) > 10:
                print(f"... and {len(lines) - 10} more lines")
            
            print("-" * 40)
            print()
            
            # Ask for confirmation
            response = input("Apply changes to this file? (y/N): ")
            if response.lower() in ['y', 'yes']:
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                print(f"Successfully cleaned {test_file}")
            else:
                print("Aborted - no changes made")
        else:
            print("No emojis found in this file.")
        
        return
    
    base_dir = "."
    
    # File extensions to process
    extensions = ['*.py', '*.md', '*.rst', '*.txt', '*.yml', '*.yaml']
    
    print("=" * 80)
    print("EMOJI INVENTORY REPORT")
    print("=" * 80)
    
    total_files = 0
    files_with_emojis = 0
    total_emojis = 0
    all_emojis = defaultdict(int)
    file_details = []
    
    for ext in extensions:
        pattern = os.path.join(base_dir, '**', ext)
        files = glob.glob(pattern, recursive=True)
        
        for file_path in files:
            # Skip __pycache__, .git, virtual environments, and build directories
            skip_dirs = ['__pycache__', '.git', 'qgml_env', '.venv', 'venv', 'env', '_build', '.pytest_cache', 'node_modules', 'qgml.egg-info']
            if any(skip_dir in file_path for skip_dir in skip_dirs):
                continue
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
                
            total_files += 1
            inventory = inventory_file(file_path)
            
            if inventory:
                files_with_emojis += 1
                total_emojis += inventory['total_emojis']
                file_details.append(inventory)
                
                # Count all emojis globally
                for emoji, count in inventory['emoji_counts'].items():
                    all_emojis[emoji] += count
            
            # Process file for removal if not dry run
            if not dry_run and inventory:
                process_file(file_path, dry_run=False)
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"Total files scanned: {total_files}")
    print(f"Files with emojis: {files_with_emojis}")
    print(f"Total emoji instances: {total_emojis}")
    print(f"Unique emoji types: {len(all_emojis)}")
    
    # Print emoji frequency
    if all_emojis:
        print(f"\nEMOJI FREQUENCY:")
        sorted_emojis = sorted(all_emojis.items(), key=lambda x: x[1], reverse=True)
        for emoji, count in sorted_emojis[:20]:  # Top 20
            print(f"  {emoji} : {count} times")
        if len(sorted_emojis) > 20:
            print(f"  ... and {len(sorted_emojis) - 20} more emoji types")
    
    # Print file details
    if file_details:
        print(f"\nFILES WITH EMOJIS:")
        for detail in sorted(file_details, key=lambda x: x['total_emojis'], reverse=True):
            print(f"  {detail['file']}")
            print(f"    Total emojis: {detail['total_emojis']}")
            print(f"    Unique emojis: {detail['unique_emojis']}")
            print(f"    Emojis: {' '.join(detail['emoji_list'])}")
            print()
    
    # Show what would be cleaned
    if dry_run and files_with_emojis > 0:
        print("=" * 80)
        print("DRY RUN COMPLETE")
        print("=" * 80)
        print(f"To actually remove all emojis, run:")
        print(f"  python {sys.argv[0]} --remove")
        print()
        print("This will permanently delete all emojis from the codebase!")
    elif not dry_run:
        print("=" * 80)
        print("EMOJI REMOVAL COMPLETE")
        print("=" * 80)
        print(f"Removed emojis from {files_with_emojis} files")

if __name__ == "__main__":
    main()
