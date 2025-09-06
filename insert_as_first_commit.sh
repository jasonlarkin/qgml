#!/bin/bash
# Insert cleaned commit as the first commit in qgml repo

if [ $# -lt 2 ]; then
    echo "Usage: $0 <cleaned_commit_dir> <target_date>"
    echo "Example: $0 work_be53748_cleaned 2025-04-13"
    exit 1
fi

CLEANED_DIR=$1
TARGET_DATE=$2
BACKUP_BRANCH="backup-main-$(date +%s)"

set -e

echo "=== Step 1: Creating backup branch ==="
cd /mnt/c/Users/jason/Documents/qcml_new/qgml
git branch "$BACKUP_BRANCH"
echo "Backup created: $BACKUP_BRANCH"

echo ""
echo "=== Step 2: Adding cleaned commit as remote ==="
cd "$CLEANED_DIR"
CLEANED_COMMIT=$(git rev-parse HEAD)
cd ..

# Add cleaned repo as a temporary remote
git remote add temp_cleaned "$(pwd)/$CLEANED_DIR" 2>/dev/null || git remote set-url temp_cleaned "$(pwd)/$CLEANED_DIR"
git fetch temp_cleaned

echo ""
echo "=== Step 3: Creating new orphan branch ==="
git checkout --orphan new-main
git rm -rf . 2>/dev/null || true

echo ""
echo "=== Step 4: Cherry-picking cleaned commit ==="
git cherry-pick "$CLEANED_COMMIT" --allow-empty

# Update commit date
git commit --amend --date="$TARGET_DATE" --no-edit

echo ""
echo "=== Step 5: Rebasing existing commits on top ==="
# Get list of commits from old main (excluding the emoji one)
OLD_MAIN=$(git rev-parse "$BACKUP_BRANCH")
FIRST_OLD_COMMIT=$(git rev-list --reverse "$BACKUP_BRANCH" | head -1)

# Cherry-pick all commits from old main onto new base
git cherry-pick "$FIRST_OLD_COMMIT".."$OLD_MAIN" || {
    echo "Cherry-pick had conflicts. Resolve and continue with:"
    echo "  git cherry-pick --continue"
    echo "Or abort with:"
    echo "  git cherry-pick --abort"
    echo "  git checkout main"
    echo "  git branch -D new-main"
    exit 1
}

echo ""
echo "=== Step 6: Replacing main branch ==="
echo "Review the new history:"
git log --oneline --date=short --format="%h %ad %s" | head -10
echo ""
read -p "Replace main with this new history? (y/n) " -r
if [ "$REPLY" = "y" ]; then
    git branch -D main
    git branch -m main
    git remote remove temp_cleaned
    echo "Main branch replaced. Old main saved as $BACKUP_BRANCH"
    echo "To restore: git checkout $BACKUP_BRANCH && git branch -D main && git branch -m main"
else
    echo "Aborted. New history in branch 'new-main'"
    echo "Old main still active. Cleanup:"
    echo "  git checkout main"
    echo "  git branch -D new-main"
    echo "  git remote remove temp_cleaned"
fi

