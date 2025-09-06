#!/bin/bash
# Script to inspect a commit from old repos before migration

if [ $# -lt 2 ]; then
    echo "Usage: $0 <commit_hash> <repo_path>"
    echo "Example: $0 be53748 old_repos/qcml_orig"
    exit 1
fi

COMMIT=$1
REPO=$2

if [ ! -d "$REPO" ]; then
    echo "Error: Repository path does not exist: $REPO"
    exit 1
fi

cd "$REPO" || exit 1

echo "=== Commit Information ==="
git log -1 --format="%h %ad %an%n%s%n%b" --date=short "$COMMIT"
echo ""
echo "=== Files Changed ==="
git show --stat "$COMMIT"
echo ""
echo "=== Checkout this commit? (y/n) ==="
read -r response
if [ "$response" = "y" ]; then
    git checkout "$COMMIT"
    echo "Checked out commit $COMMIT"
    echo "You can now inspect files and run branding cleanup"
    echo "To return: git checkout main (or master)"
fi

