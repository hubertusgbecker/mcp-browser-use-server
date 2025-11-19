#!/usr/bin/env bash
# Install local git hooks from .githooks to .git/hooks
set -euo pipefail

HOOK_DIR=".githooks"
GIT_HOOKS_DIR=".git/hooks"

if [ ! -d ".git" ]; then
	echo "Not a git repository. Run this from the repo root." >&2
	exit 1
fi

if [ ! -d "$HOOK_DIR" ]; then
	echo "No $HOOK_DIR directory found; nothing to install." >&2
	exit 0
fi

mkdir -p "$GIT_HOOKS_DIR"

for hook in "$HOOK_DIR"/*; do
	[ -e "$hook" ] || continue
	hook_name=$(basename "$hook")
	target="$GIT_HOOKS_DIR/$hook_name"
	if [ -e "$target" ]; then
		backup="$target.bak.$(date +%s)"
		echo "Backing up existing hook $target -> $backup"
		mv "$target" "$backup"
	fi
	echo "Installing $hook_name -> $target"
	cp "$hook" "$target"
	chmod +x "$target"
done

echo "Git hooks installed from $HOOK_DIR into $GIT_HOOKS_DIR."
