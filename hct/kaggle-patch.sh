#!/bin/sh

set -eux

# kaggle c leaderboard ...
# failed with KeyError
# This is fixed in the latest development version
# https://github.com/bedwards/kaggle-api/blob/main/src/kaggle/api/kaggle_api_extended.py#L2044-L2049
# but pip install -U kaggle
# did not fix it

# for me the argument passed to this script is
# p="$HOME/miniconda3/envs/jupyterlab/lib/python3.12/site-packages/kaggle/api/kaggle_api_extended.py"
# diff -u kaggle_api_extended.py $p > kaggle-patch.patch

patch $1 < kaggle-patch.patch
