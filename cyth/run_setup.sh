#!/bin/sh
rm __init__.py
rm __init__.pyc
rm -rf build/
rm *.c
rm *.so
python setup.py build_ext --inplace
touch __init__.py
