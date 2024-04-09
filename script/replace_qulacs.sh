#!/bin/sh

set -eu

files=$(git ls-files)
for file in $files; do
    if [ $file = $0 ]; then
        continue
    fi
    sed -i -e 's/qulacs2023/scaluq/g' -e 's/qulacs/scaluq/g' -e 's/QULACS/SCALUQ/g' -e 's/scaluq.osaka@gmail.com/qulacs.osaka@gmail.com/g' $file
    if [ $file = *'qulacs'* ]; then
        echo 'Warning: filename including qulacs: ' $file
    fi
done

files=$(git ls-files)

