#!/bin/bash

maturin develop --release
python3 test.py