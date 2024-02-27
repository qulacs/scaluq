#!/bin/sh

set -eux

nanobind-stubgen qulacs2023 --out './typings'
cp -R typings/qulacs2023/* python/qulacs2023/
find python/ -name __init__.pyi | sed -e 's/__init__.pyi/py.typed/' | xargs touch

black python/
isort python/
