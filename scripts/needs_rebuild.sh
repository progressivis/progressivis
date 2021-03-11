#!/usr/bin/env bash
var=$(git diff-tree --no-commit-id --name-only -r HEAD)
if (echo "$var"  | fgrep -q "binder/environment.yml"); then
    echo "REBUILD"
    exit
fi
if (echo "$var"  | fgrep -q "setup.py"); then
    echo "REBUILD"
    exit
fi
if (echo "$var"  | fgrep -q "requirements.txt"); then
    echo "REBUILD"
    exit
fi
