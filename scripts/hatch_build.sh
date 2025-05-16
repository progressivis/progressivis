#!/bin/bash
set -e


GIT_URL="git@github.com:progressivis/progressivis.git"
REPO_DIR="progressivis"
PKG_DIR="progressivis"

tag=$1
if [ -z "$tag" ]; then
    echo "Error: tag is blank";
    exit 1
fi
dir=$(mktemp -d)
if [ -z "$dir" ]; then
    echo "Error: dir is blank";
    exit 1
fi
if test -d $dir; then
   echo "OK: directory $dir exists."
else
    echo "Error: $dir directory does not exist"
    exit 1
fi
if ! [ -z "$( ls -A $dir )" ]; then
    echo "Error: $dir is not empty"
    exit 1
fi
echo "Can work!"
cd $dir
git clone $GIT_URL
cd $REPO_DIR
if test -f "$PKG_DIR/_version.py"; then
    echo "$PKG_DIR/_version.py already exists, ABORT"
    exit 1
else
    echo "$PKG_DIR/_version.py does not exist yet, OK"
fi
git checkout "$tag"
git status
hatch build
if test -f "$PKG_DIR/_version.py"; then
    echo "$PKG_DIR/_version.py exists now:"
    cat "$PKG_DIR/_version.py"
else
    echo "$PKG_DIR/_version.py does not exist after build, ABORT"
    exit 1
fi
python -c "exec(open('./$PKG_DIR/_version.py').read()); assert 'v' + __version__ == '$tag'; print(f'Version in _version.py is: {version}')"
#hatch version
echo "Get your files here $(realpath './dist')"
