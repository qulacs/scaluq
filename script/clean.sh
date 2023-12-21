#!/bin/sh

find build/ -mindepth 1 -maxdepth 1 | xargs rm -rf
rm -rf ./bin
rm -rf ./lib
rm -rf ./.mypy_cache
