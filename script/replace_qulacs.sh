#!/bin/sh

set -eu

files=$(git ls-files)
for file in $files; do
    if [ $file = $0 ]; then
        continue
    fi
    sed -i -e 's/scaluq/scaluq/g' -e 's/scaluq/scaluq/g' -e 's/SCALUQ/SCALUQ/g' -e 's/SCALUQ/SCALUQ/g' -e 's/qulacs.osaka@gmail.com/qulacs.osaka@gmail.com/g' $file
    if [ $file = *'scaluq'* ]; then
        echo 'Warning: filename including scaluq: ' $file
    fi
done

files=$(git ls-files)

