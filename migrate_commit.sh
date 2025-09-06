#!/bin/bash
# Script to migrate a commit from old repo to QGML with branding cleanup

if [ $# -lt 3 ]; then
    echo "Usage: $0 <commit_hash> <repo_path> <target_date>"
    echo "Example: $0 be53748 old_repos/qcml_orig 2025-04-13"
    exit 1
fi

COMMIT=$1
REPO=$2
TARGET_DATE=$3
WORK_DIR="work_$(basename $REPO)_${COMMIT:0:7}"

echo "=== Migrating commit $COMMIT from $REPO ==="
echo "Target date: $TARGET_DATE"
echo ""

# Create work directory
mkdir -p "$WORK_DIR"
cd "$REPO" || exit 1

# Export the commit to work directory
echo "Exporting commit files..."
git archive "$COMMIT" | tar -x -C "../$WORK_DIR"

cd "../$WORK_DIR" || exit 1

echo ""
echo "=== Files exported to $WORK_DIR ==="
echo "Run branding cleanup:"
echo "  python3 ../clean_branding.py . --find-only"
echo "  python3 ../clean_branding.py . --replace"
echo ""
echo "After cleanup, review changes and then we can create a new commit"

