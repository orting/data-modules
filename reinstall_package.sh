#!/bin/bash
pkg=data_module
python3 -m build && 
    pip uninstall -y ${pkg} &&
    pip install ${pkg} -f dist
