# fast-mc

[![Build Status](https://travis-ci.org/tbs1980/fast-mc.svg?branch=develop)](https://travis-ci.org/tbs1980/bclest)
[![codecov](https://codecov.io/gh/tbs1980/fast-mc/branch/master/graph/badge.svg)](https://codecov.io/gh/tbs1980/fast-mc)

## Installation

```bash
conda update conda
conda config --add channels intel
conda create -n fast-mc-env intelpython3_core python=3
source activate fast-mc-env
conda install --file requirements.txt
conda install --file requirements_dev.txt
```