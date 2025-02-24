#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from lifelines import KaplanMeierFitter, NelsonAalenFitter

# Read the data
data_dir = "equity-post-HCT-survival-predictions"
train = pd.read_csv(f"../input/{data_dir}/train.csv")
test = pd.read_csv(f"../input/{data_dir}/test.csv")

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Basic statistics about the dataset
print("\n" + "="*80)
print("COLUMN INFORMATION")
print("="*80)
print("\nColumn types:")
print(train.dtypes)

# Identify column types
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")

# Missing values
print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
missing_train = train.isnull().sum()
missing_train_pct = (train.isnull().sum() / len(train)) * 100
missing_test = test.isnull().sum()
missing_test_pct = (test.isnull().sum() / len(test)) * 100

missing_df = pd.DataFrame({
    'Missing Train': missing_train,
    'Missing Train %': missing_train_pct,
    'Missing Test': missing_test,
    'Missing Test %': missing_test_pct
})

print("\nColumns with missing values:")
print(missing_df[missing_df['Missing Train'] > 0].sort_values('Missing Train', ascending=False))

# Target variables analysis
print("\n" + "="*80)
print("TARGET VARIABLE ANALYSIS")
print("="*80)
print("\nEFS (Event) Statistics:")
efs_counts = train['efs'].value_counts()
print(efs_counts)
print(f"Event rate: {efs_counts[1] / len(train):.4f}")

print("\nEFS Time Statistics:")
print(train['efs_time'].describe())

print("\nEFS Time by Event Status:")
print(train.groupby('efs')['efs_time'].describe())

# Race group analysis
print("\n" + "="*80)
print("RACE GROUP ANALYSIS")
print("="*80)
race_counts = train['race_group'].value_counts()
print("\nRace Group Distribution:")
print(race_counts)
print("\nPercentage by Race Group:")
print(race_counts / len(train) * 100)

print("\nEvent Rate by Race Group:")
event_by_race = train.groupby('race_group')['efs'].mean()
print(event_by_race)

print("\nEFS Time Statistics by Race Group:")
print(train.groupby('race_group')['efs_time'].describe())

# Categorical variables analysis
print("\n" + "="*80)
print("CATEGORICAL VARIABLES ANALYSIS")
print("="*80)

for col in categorical_cols:
    if col != 'ID' and col != 'race_group':
        value_counts = train[col].value_counts()
        if len(value_counts) <= 20:  # Only print if not too many categories
            print(f"\n{col} - {len(value_counts)} unique values:")
            print(value_counts)
        else:
            print(f"\n{col} - {len(value_counts)} unique values (top 10):")
            print(value_counts.head(10))

# Numerical variables analysis
print("\n" + "="*80)
print("NUMERICAL VARIABLES ANALYSIS")
print("="*80)

for col in numerical_cols:
    if col not in ['ID', 'efs', 'efs_time']:
        print(f"\n{col} statistics:")
        print(train[col].describe())
        
        # Check if the column has integer-like values even though stored as float
        if train[col].dtype == 'float64':
            integer_like = (train[col] % 1 == 0).all()
            if integer_like:
                print(f"Note: {col} has only integer values despite being float type")
                
                # Print value counts if not too many unique values
                unique_count = train[col].nunique()
                if unique_count <= 20:
                    print(f"Value counts for {col}:")
                    print(train[col].value_counts().sort_index())

# KM and NA transformations exploration
print("\n" + "="*80)
print("SURVIVAL ANALYSIS TRANSFORMATIONS")
print("="*80)

def transform_km(df):
    """Transform using Kaplan-Meier survival function"""
    kmf = KaplanMeierFitter()
    kmf.fit(df['efs_time'], df['efs'])
    return kmf.survival_function_at_times(df['efs_time']).values

def transform_na(df):
    """Transform using Nelson-Aalen cumulative hazard function"""
    naf = NelsonAalenFitter()
    naf.fit(df['efs_time'], df['efs'])
    return -naf.cumulative_hazard_at_times(df['efs_time']).values

# Calculate the transforms
km_values = transform_km(train)
na_values = transform_na(train)

# Print statistics
print("\nKaplan-Meier transformation statistics:")
print(f"Mean: {np.mean(km_values):.6f}")
print(f"Std Dev: {np.std(km_values):.6f}")
print(f"Min: {np.min(km_values):.6f}")
print(f"Max: {np.max(km_values):.6f}")

print("\nNelson-Aalen transformation statistics:")
print(f"Mean: {np.mean(na_values):.6f}")
print(f"Std Dev: {np.std(na_values):.6f}")
print(f"Min: {np.min(na_values):.6f}")
print(f"Max: {np.max(na_values):.6f}")

# Check difference in transforms by event status
km_by_event = {
    'event=1': transform_km(train[train['efs'] == 1]),
    'event=0': transform_km(train[train['efs'] == 0])
}

na_by_event = {
    'event=1': transform_na(train[train['efs'] == 1]),
    'event=0': transform_na(train[train['efs'] == 0])
}

print("\nKaplan-Meier by event status:")
for status, values in km_by_event.items():
    print(f"{status} - Mean: {np.mean(values):.6f}, Std: {np.std(values):.6f}, Min: {np.min(values):.6f}, Max: {np.max(values):.6f}")

print("\nNelson-Aalen by event status:")
for status, values in na_by_event.items():
    print(f"{status} - Mean: {np.mean(values):.6f}, Std: {np.std(values):.6f}, Min: {np.min(values):.6f}, Max: {np.max(values):.6f}")

# Model names analysis from the code
print("\n" + "="*80)
print("MODEL TYPES ANALYSIS")
print("="*80)
models = {
    'xgb_kmrace': 'XGBoost with Kaplan-Meier target by race',
    'lgb_kmrace': 'LightGBM with Kaplan-Meier target by race',
    'xgb_na': 'XGBoost with Nelson-Aalen target',
    'cb_kmrace': 'CatBoost with Kaplan-Meier target by race',
    'cb_na': 'CatBoost with Nelson-Aalen target',
    'lgb_na': 'LightGBM with Nelson-Aalen target',
    'xgb_cox': 'XGBoost with Cox proportional hazards target',
    'cb_cox': 'CatBoost with Cox proportional hazards target'
}

print("\nModel types used in the competition:")
for model_key, description in models.items():
    print(f"- {model_key}: {description}")

# Feature relationships analysis
print("\n" + "="*80)
print("FEATURE RELATIONSHIPS")
print("="*80)

# Check correlation between age_at_hct and donor_age
corr = train['age_at_hct'].corr(train['donor_age'])
print(f"\nCorrelation between age_at_hct and donor_age: {corr:.4f}")

# Check common categorical feature relationships
# print("\nRelationship between donor_gender and recipient_gender:")
# gender_crosstab = pd.crosstab(train['donor_gender'], train['recipient_gender'], normalize='index')
# print(gender_crosstab)

print("\nRelationship between hla, tce_match and tce_div_match:")
for hla_col in [col for col in train.columns if 'hla' in col]:
    unique_count = train[hla_col].nunique()
    print(f"\n{hla_col}: {unique_count} unique values")
    if unique_count <= 10:
        print(train[hla_col].value_counts())

print("\nTCE match types distribution:")
for tce_col in ['tce_match', 'tce_div_match']:
    print(f"\n{tce_col}:")
    print(train[tce_col].value_counts())

# Check relationship between dri_score, disease_group, and disease_status
print("\nRelationship between disease features:")
print("\ndri_score value counts:")
print(train['dri_score'].value_counts())

# print("\ndisease_group value counts:")
# print(train['disease_group'].value_counts())

# Analysis of age-related features
print("\n" + "="*80)
print("AGE-RELATED FEATURES ANALYSIS")
print("="*80)

print("\nBins of age_at_hct:")
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80']
train['age_group'] = pd.cut(train['age_at_hct'], bins=age_bins, labels=age_labels)
age_counts = train['age_group'].value_counts().sort_index()
print(age_counts)

print("\nEvent rate by age group:")
event_by_age = train.groupby('age_group')['efs'].mean().sort_index()
print(event_by_age)

print("\nMedian survival time by age group:")
median_time_by_age = train.groupby('age_group')['efs_time'].median().sort_index()
print(median_time_by_age)

# Summary of key findings
print("\n" + "="*80)
print("SUMMARY OF KEY FINDINGS")
print("="*80)

print("\n1. Dataset Structure:")
print(f"   - Train set has {train.shape[0]} rows and {train.shape[1]} columns")
print(f"   - {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features")
print(f"   - Event rate: {efs_counts[1] / len(train):.4f}")

print("\n2. Missing Values:")
missing_cols = missing_df[missing_df['Missing Train'] > 0]
print(f"   - {len(missing_cols)} columns have missing values")
if len(missing_cols) > 0:
    print(f"   - Top column with missing values: {missing_cols.index[0]} ({missing_cols['Missing Train %'].iloc[0]:.2f}%)")

print("\n3. Race Distribution:")
print(f"   - Number of race groups: {len(race_counts)}")
print(f"   - Most common race: {race_counts.index[0]} ({race_counts.iloc[0]} patients, {race_counts.iloc[0]/len(train)*100:.2f}%)")
print(f"   - Least common race: {race_counts.index[-1]} ({race_counts.iloc[-1]} patients, {race_counts.iloc[-1]/len(train)*100:.2f}%)")

print("\n4. Target Transformations:")
print("   - The competition uses several target transformations:")
print("     * Kaplan-Meier survival probability")
print("     * Nelson-Aalen cumulative hazard")
print("     * Cox proportional hazards")
print("     * Custom KM by race group")

print("\n5. Model Types:")
print("   - Tree-based models (XGBoost, LightGBM, CatBoost)")
print("   - Multiple target transformations for each model type")
print("   - Ensemble of different model types and targets")

# End of analysis
print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)
