#!/bin/bash
pkg=data_modules
python3 -m build && 
    pip uninstall -y ${pkg} &&
    pip install ${pkg} -f dist
