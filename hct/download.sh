#!/bin/sh

set -eux

mkdir -p ./input/hct-conduit
mkdir -p ./working/csv

c="equity-post-HCT-survival-predictions"
csv="./input/equity-post-HCT-survival-predictions"
mkdir -p $csv
kaggle c download $c -p $csv

lb="./input/hct-leaderboard"
mkdir -p $lb
kaggle c leaderboard $c -d -p $lb
