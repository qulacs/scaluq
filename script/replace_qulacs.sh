#!/bin/sh

set -eu

script_name=$(basename "$0")
files=$(git ls-files)
for file in $files; do
    if [ $(basename "$file") = "$script_name" ]; then
        continue
    fi
    sed -i -e 's/qulacs2023/scaluq/g' -e 's/qulacs/scaluq/g' -e 's/QULACS2023/SCALUQ/g' -e 's/QULACS/SCALUQ/g' -e 's/scaluq.osaka@gmail.com/qulacs.osaka@gmail.com/g' $file
    if [ $file = *'qulacs'* ]; then
        echo 'Warning: filename including qulacs: ' $file
    fi
done

files=$(git ls-files)

