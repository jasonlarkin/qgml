#!/bin/bash
# Complete workflow: export commit, clean branding, rename directories, create new commit

if [ $# -lt 3 ]; then
    echo "Usage: $0 <commit_hash> <repo_path> <target_date> [new_commit_message]"
    echo "Example: $0 be53748 old_repos/qcml_orig 2025-04-13"
    exit 1
fi

COMMIT=$1
REPO=$2
TARGET_DATE=$3
NEW_MSG="${4:-Initial commit: Quantum Geometric Machine Learning library}"
WORK_DIR="work_${COMMIT:0:7}"
CLEAN_DIR="${WORK_DIR}_cleaned"

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$SCRIPT_DIR/work_${COMMIT:0:7}"
CLEAN_DIR="$SCRIPT_DIR/work_${COMMIT:0:7}_cleaned"
REPO_PATH="$SCRIPT_DIR/$REPO"

echo "=== Step 1: Exporting commit $COMMIT ==="
mkdir -p "$WORK_DIR"
cd "$REPO_PATH" || exit 1
git archive "$COMMIT" | tar -x -C "$WORK_DIR"
cd "$WORK_DIR" || exit 1

echo "=== Step 2: Applying branding cleanup ==="
python3 ../clean_branding.py . --replace

echo ""
echo "=== Step 3: Renaming directories and files ==="
# Rename qcml/ to qgml/
if [ -d "qcml" ]; then
    mv qcml qgml
    echo "Renamed qcml/ -> qgml/"
fi

# Rename files containing qcml in their names
find . -type f -name "*qcml*" | while read file; do
    newfile=$(echo "$file" | sed 's/qcml/qgml/g')
    if [ "$file" != "$newfile" ]; then
        mv "$file" "$newfile"
        echo "Renamed file: $(basename $file) -> $(basename $newfile)"
    fi
done

# Update any remaining references in file paths
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.toml" \) | while read file; do
    # Update import paths
    sed -i 's/from qcml\./from qgml./g' "$file"
    sed -i 's/import qcml\./import qgml./g' "$file"
    sed -i "s/'qcml\./'qgml./g" "$file"
    sed -i 's/"qcml\./"qgml./g' "$file"
    # Update file references in strings
    sed -i 's/qcml\.py/qgml.py/g' "$file"
    sed -i 's/test_qcml/test_qgml/g' "$file"
done

echo "=== Step 4: Creating cleaned directory ==="
cd "$SCRIPT_DIR" || exit 1
cp -r "$WORK_DIR" "$CLEAN_DIR"
cd "$CLEAN_DIR" || exit 1

# Initialize git repo for the cleaned commit
git init
git add .
git commit -m "$NEW_MSG" --date="$TARGET_DATE"

echo ""
echo "=== Cleaned commit created in $CLEAN_DIR ==="
echo "Commit hash: $(git rev-parse HEAD)"
echo ""
echo "Next steps:"
echo "1. Review changes in $CLEAN_DIR"
echo "2. Cherry-pick into qgml repo using the commit hash above"

