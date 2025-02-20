#!/bin/sh

set -eux

mkdir -p ./input/titanic-models
mkdir -p ./working/csv

c="titanic"
csv="./input/$c"
mkdir -p $csv
kaggle c download $c -p $csv
unzip -o ${csv}/*.zip -d $csv
rm -fv ${csv}/*.zip

lb="./input/titanic-leaderboard"
mkdir -p $lb
rm -fv ${lb}/*.csv
kaggle c leaderboard $c -d -p $lb
unzip -o "$lb"/*.zip -d $lb
rm -fv ${lb}/*.zip
mv -fv ${lb}/*.csv ${lb}/lb-`date +%Y-%m-%d`.csv
