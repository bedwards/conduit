#!/bin/sh

set -eux

mkdir -p ./input/hct-conduit
mkdir -p ./working/csv

c="equity-post-HCT-survival-predictions"
csv="./input/$c"
mkdir -p $csv
kaggle c download $c -p $csv
unzip -o ${csv}/*.zip -d $csv
rm -fv ${csv}/*.zip

lb="./input/hct-leaderboard"
mkdir -p $lb
rm -fv ${lb}/*.csv
kaggle c leaderboard $c -d -p $lb
unzip -o "$lb"/*.zip -d $lb
rm -fv ${lb}/*.zip
mv -fv ${lb}/*.csv ${lb}/lb-`date +%Y-%m-%d`.csv
