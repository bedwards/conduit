# Conduit - A Kaggle Competition Framework

Conduit is a lightweight framework for Kaggle competitions that emphasizes:

- Local development with seamless transition to Kaggle notebooks
- Reproducible experiments through self-contained scripts 
- Easy model iteration and ensemble creation
- Efficient use of multiprocessing for local training

## Key Features

- **Single-File Experiments**: Each experiment is a standalone Python script that can run both locally and in Kaggle notebooks
- **GPU/CPU Flexibility**: Automatic detection and configuration for running on Kaggle's GPU or local CPU
- **Efficient Training**: Local multiprocessing support for faster model training
- **Version Control**: Simple versioning through numbered conduit scripts (conduit-00.py, conduit-01.py, etc.)
- **Competition Structure**: Organized directory structure for managing competition data, models, and submissions

## Directory Structure

```
competition_name/
├── input/
│   ├── competition-data/          # Competition datasets
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   ├── competition-leaderboard/   # Historical leaderboard data
│   └── competition-models/        # Saved models for Kaggle upload
│
└── working/
├── csv/                       # Preprocessed data
├── download.sh               # Data download script
└── conduit-XX.py            # Experiment scripts
```

## Usage

1. Create new competition directory:

```
mkdir competition_name
cd competition_name
```

2. Download competition data:

```
./download.sh
```

3. Create experiment:

```
cd working
python conduit-XX.py
```

4. Submit to Kaggle:
- Upload saved models as dataset
- Copy experiment script to notebook
- Run with `INCLUDE_FIT_ON_KAGGLE=False`

## Current Competitions

- [HCT Survival Prediction](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions) - Predicting survival outcomes after hematopoietic cell transplantation

## Design Philosophy

- Minimize dependencies between files
- Make experiments self-contained and reproducible
- Enable quick iteration while maintaining organization
- Support both local development and Kaggle environment
