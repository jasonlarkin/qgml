#!/usr/bin/env python3
"""Simple script to check if known emoji files are clean."""

import re

# Files we cleaned
files_to_check = [
    "./migration_test_report.md",
    "./docs/source/experimental_results/integration_analysis.rst",
    "./docs/source/experimental_results/gpu_performance.rst",
    "./tests/test_integration/test_trainer_integration.py",
    "./docs/source/examples/genomics_application.rst",
    "./docs/source/experimental_results/backend_performance.rst",
    "./docs/README.md",
    "./experiments/backend_comparison/results/real_comparison/real_test_comparison_report.md",
    "./docs/experimental_results/index.rst",
    "./docs/index.rst",
    "./docs/experimental_results/performance_visualizations.rst"
]

emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]+')

print("Checking cleaned files for remaining emojis...")
print("=" * 50)

all_clean = True
for file_path in files_to_check:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        emojis = emoji_pattern.findall(text)
        if emojis:
            print(f"FAILED {file_path}: {len(emojis)} emojis found")
            all_clean = False
        else:
            print(f"PASS  {file_path}: Clean")
    except FileNotFoundError:
        print(f"SKIP  {file_path}: File not found")
    except Exception as e:
        print(f"ERROR {file_path}: Error - {e}")

print("=" * 50)
if all_clean:
    print("SUCCESS: All known emoji files are clean!")
else:
    print("WARNING: Some files still have emojis")
