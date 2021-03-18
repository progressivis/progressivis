#!/usr/bin/env bash

envs=$(conda info --envs)
if ! (echo "$envs"  | grep -q "progressivis"); then
    echo "REBUILD"
    exit
fi
var=$(git diff-tree --no-commit-id --name-only -r HEAD)

if (echo "$var"  | fgrep -q "environment-ci.yml"); then
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
if (echo "$var"  | fgrep -q "needs_rebuild.sh"); then
    echo "REBUILD"
    exit
fi
