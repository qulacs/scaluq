#!/bin/sh

set -eux

nanobind-stubgen scaluq --out './typings'
cp -R typings/scaluq/* python/scaluq/
find python/ -name __init__.pyi | sed -e 's/__init__.pyi/py.typed/' | xargs touch

black python/
isort python/
