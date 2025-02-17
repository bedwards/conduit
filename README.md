### Kaggle competition pipeline

### Competitions

One top-level directory per competition

- [hct](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions)

### Structure

Each competition directory has the following structure

```
# results from running ./download.sh

input/equity-post-HCT-survival-predictions/test.csv
input/equity-post-HCT-survival-predictions/train.csv
input/equity-post-HCT-survival-predictions/data_dictionary.csv
input/equity-post-HCT-survival-predictions/sample_submission.csv

input/hct-leaderboard/lb-2025-02-17.csv


# results from `cd working` and running ./conduit-00.py

input/hct-conduit/conduit-00.json
input/hct-conduit/conduit-00.joblib


# more advanced conduit scripts will save their
# preprocessed X and y csv files here

working/csv
```
