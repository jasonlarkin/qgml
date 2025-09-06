#!/usr/bin/env python3
"""
Targeted emoji removal for the remaining files we know have emojis.
"""

import os
import re

def remove_emojis_from_text(text):
    """Remove all emojis from text using Unicode ranges."""
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
    
    # Remove emojis but preserve line structure
    cleaned = emoji_pattern.sub('', text)
    # Only clean up multiple spaces within lines, preserve newlines
    lines = cleaned.split('\n')
    cleaned_lines = []
    for line in lines:
        # Clean up multiple spaces within the line
        cleaned_line = re.sub(r'[ \t]+', ' ', line.strip())
        cleaned_lines.append(cleaned_line)
    cleaned = '\n'.join(cleaned_lines)
    
    return cleaned

def clean_file(file_path):
    """Clean a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        cleaned = remove_emojis_from_text(content)
        
        if content != cleaned:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            print(f"Cleaned: {file_path}")
            return True
        else:
            print(f"No emojis in: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Clean the remaining files with emojis."""
    
    # Files we know have emojis from the inventory
    files_to_clean = [
        "migration_test_report.md",
        "docs/source/experimental_results/integration_analysis.rst",
        "docs/source/experimental_results/gpu_performance.rst",
        "experiments/backend_comparison/gpu_performance.py",
        "experiments/backend_comparison/scaling_analysis.py",
        "experiments/backend_comparison/performance_comparison.py",
        "experiments/backend_comparison/real_data_comparison.py",
        "experiments/backend_comparison/gpu_test_suite.py",
        "experiments/backend_comparison/quick_validation.py",
        "experiments/applications/genomics/chromosomal_instability.py",
        "experiments/backend_comparison/function_comparison.py",
        "docs/source/examples/genomics_application.rst",
        "experiments/integration_validation/comprehensive_validation.py",
        "tests/test_integration/test_trainer_integration.py",
        "docs/source/experimental_results/backend_performance.rst",
        "experiments/backend_comparison/gpu_benchmarks.py",
        "experiments/integration_validation/quick_validation.py",
        "experiments/quantum_hardware/qiskit_implementations.py",
        "docs/README.md",
        "experiments/backend_comparison/results/real_comparison/real_test_comparison_report.md",
        "docs/experimental_results/index.rst",
        "docs/index.rst",
        "docs/experimental_results/performance_visualizations.rst"
    ]
    
    print("Cleaning remaining files with emojis...")
    print("=" * 50)
    
    cleaned_count = 0
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            if clean_file(file_path):
                cleaned_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print("=" * 50)
    print(f"Cleaned {cleaned_count} files")

if __name__ == "__main__":
    main()
