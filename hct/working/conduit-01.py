#!/usr/bin/env python

import warnings; warnings.simplefilter('ignore')
import os
import sys
import csv
import shutil
import contextlib
from glob import glob
from functools import partial
from collections import defaultdict
from itertools import combinations, starmap
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.utils import concordance_index
import joblib

NAME = 'conduit-01'

# if True: submission.csv created using all models from PIPE
# if False: uses ensemble combination with best CV score
SUBMIT_FULL_ENSEMBLE = True

# if True: model fit is run on kaggle
# if False: saved models from local run are used from a kaggle dataset
INCLUDE_FIT_ON_KAGGLE = False

PROVE_MY_SCORE_IS_SAME = False

C_PATH = '../input/equity-post-HCT-survival-predictions'
LB_PATH = '../input/hct-leaderboard'
CSV_PATH = f'./csv/{NAME}'
os.makedirs(CSV_PATH, exist_ok=True)

MODEL_PATH = '../input/hct-conduit'

try:
    get_ipython
except NameError:
    RUNNING_ON_KAGGLE = False
else:
    RUNNING_ON_KAGGLE = True

if RUNNING_ON_KAGGLE:
    if INCLUDE_FIT_ON_KAGGLE:
        MODEL_PATH = f'./models/{NAME}'
        os.makedirs(MODEL_PATH, exist_ok=True)
else:
    os.makedirs(MODEL_PATH, exist_ok=True)

# https://www.kaggle.com/code/cdeotte/gpu-lightgbm-baseline-cv-681-lb-685
kwargs_xgb = dict(max_depth=3,
                  colsample_bytree=0.5,
                  subsample=0.8,
                  n_estimators=2000,
                  learning_rate=0.02,
                  min_child_weight=80,
                  enable_categorical=True,
                  verbosity=0)

kwargs_cb = dict(learning_rate=0.1,
                 grow_policy='Lossguide',
                 silent=True)

kwargs_lgb = dict(max_depth=3,
                  colsample_bytree=0.4,
                  n_estimators=2500,
                  learning_rate=0.02,
                  num_leaves=8,
                  verbose=-1,
                  verbosity=-1)

PIPES = [
    {
        'm': 'xgb',  # unique. lgb, xgb, cb, or lgb_extrainfo, etc.
        'X': 'label',  # raw, label, onehot
        'y': 'kmf',  # kmf, naf, cox
        'kwargs': kwargs_xgb,
    }, {
        'm': 'lgb',
        'X': 'label',
        'y': 'kmf',
        'kwargs': kwargs_lgb,
    }, {
        'm': 'cb',
        'X': 'label',
        'y': 'kmf',
        'kwargs': kwargs_cb,
    }, {
        'm': 'xgb_cox',
        'X': 'label',
        'y': 'cox',
        'kwargs': dict(objective='survival:cox', eval_metric='cox-nloglik', **kwargs_xgb),
    }, {
        'm': 'cb_cox',
        'X': 'label',
        'y': 'cox',
        'kwargs': dict(loss_function="Cox", iterations=400, use_best_model=False, **kwargs_cb),
    }
]

if RUNNING_ON_KAGGLE:
    for pipe in PIPES:
        if pipe['m'] in ['xgb', 'xgb_cox']:
            pipe['kwargs']= dict(device='cuda', **pipe['kwargs'])
            continue

        if pipe['m'] == 'cb':
            pipe['kwargs'] = dict(task_type='GPU', **pipe['kwargs'])
            continue

        if pipe['m'] == 'lgb':
            pipe['kwargs'] = dict(device='gpu', **pipe['kwargs'])

sns.set_palette('deep')

def read_csv(path):
    df = pd.read_csv(path).set_index('ID')
    df.index = df.index.astype('int32')
    fn = path.split('/')[-1].split('.')[0]
    if fn.startswith('X_') and fn.endswith('_raw'):
        df = pd.concat([df.select_dtypes('float').astype('float32'),
                        df.select_dtypes('object').astype('category')],
                        axis=1)
    print(f'read {path}')
    return df

def plot_y_transformation(Y_name, Y):
    if RUNNING_ON_KAGGLE:
        fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharey=True)  # Increased height from 5 to 8
        plt.subplots_adjust(hspace=0.3)
        fig.suptitle(f'Y transformation: {Y_name}', fontsize='medium')

        sns.histplot(Y,
                     y='y',
                     hue='efs',
                     ax=axes[0, 0])

        sns.scatterplot(Y,
                        x='efs_time',
                        y='y',
                        hue='efs',
                        ax=axes[0, 1])

        sns.scatterplot(Y[Y['efs'] == 0],
                        x='efs_time',
                        y='y',
                        label='efs=0',
                        ax=axes[1, 0])

        axes[1, 0].legend()

        sns.scatterplot(Y[Y['efs'] == 1],
                        x='efs_time',
                        y='y',
                        label='efs=1',
                        color=sns.color_palette()[1],
                        ax=axes[1, 1])

        axes[1, 1].legend()
        plt.tight_layout()
        plt.show()

def preprocess():
    print('preprocess')
    train = read_csv(f'{C_PATH}/train.csv')
    test = read_csv(f'{C_PATH}/test.csv')
    X = train.drop(columns=['efs', 'efs_time'])
    X = pd.concat([X, test])
    Xf = X.select_dtypes('float').astype('float32')
    Xc_raw = X.select_dtypes('object').astype('category')

    for col in Xc_raw:
        if Xc_raw[col].isna().any():
            Xc_raw[col] = Xc_raw[col].cat.add_categories('Missing').fillna('Missing')

    X_raw = pd.concat([Xf, Xc_raw], axis=1)
    Xc = X.select_dtypes('object').astype('category')

    for col in Xc:
        Xc[col], _ = Xc[col].factorize(use_na_sentinel=False)
        Xc[col] = Xc[col].astype('int32').astype('category')

    X_label = pd.concat([Xf, Xc], axis=1)
    Xc = pd.get_dummies(Xc, drop_first=True, dtype='int32')
    Xc.columns = Xc.columns.str.replace(r'[\[\]<]', '_', regex=True)
    X_onehot = pd.concat([Xf, Xc], axis=1)

    naf = NelsonAalenFitter(label='y')
    naf.fit(train['efs_time'], event_observed=train['efs'])
    Y_naf = train[['efs', 'efs_time']].join(-naf.cumulative_hazard_, on='efs_time')
    plot_y_transformation('NelsonAalenFitter', Y_naf)
    y_naf = Y_naf['y']

    kmf = KaplanMeierFitter(label='y')
    kmf.fit(train['efs_time'], event_observed=train['efs'])
    Y_kmf = train[['efs', 'efs_time']].join(kmf.survival_function_, on='efs_time')
    plot_y_transformation('KaplanMeierFitter', Y_kmf)
    y_kmf = Y_kmf['y']

    Y_cox = train[['efs', 'efs_time']]
    Y_cox['y'] = train['efs_time']
    Y_cox.loc[Y_cox['efs'] == 0, 'y'] *= -1
    plot_y_transformation('XGB survival:cox, LGBM Cox', Y_cox)
    y_cox = Y_cox['y']

    for n, data in [('X_train_raw', X_raw.iloc[:len(train)]),
                    ('X_train_label', X_label.iloc[:len(train)]),
                    ('X_train_onehot', X_onehot.iloc[:len(train)]),
                    ('X_test_raw', X_raw.iloc[len(train):]),
                    ('X_test_label', X_label.iloc[len(train):]),
                    ('X_test_onehot', X_onehot.iloc[len(train):]),
                    ('y_naf', y_naf),
                    ('y_kmf', y_kmf),
                    ('y_cox', y_cox)]:
        filename = f'{CSV_PATH}/{n}.csv'
        data.to_csv(filename)
        print(f'wrote {filename}')


def get_Xs(train_or_test):
    return {'raw': read_csv(f'{CSV_PATH}/X_{train_or_test}_raw.csv'),
            'label': read_csv(f'{CSV_PATH}/X_{train_or_test}_label.csv'),
            'onehot': read_csv(f'{CSV_PATH}/X_{train_or_test}_onehot.csv')}


def get_ys():
    return {'naf': read_csv(f'{CSV_PATH}/y_naf.csv'),
            'kmf': read_csv(f'{CSV_PATH}/y_kmf.csv'),
            'cox': read_csv(f'{CSV_PATH}/y_cox.csv')}


def create_kfold():
    return KFold(n_splits=10, shuffle=True, random_state=42)


def fit_fold_model(X, y, fold_n, i_fold, i_oof, m_name, m):
    fit_kwargs = dict(eval_set=[(X.iloc[i_oof], y.iloc[i_oof])])

    # fit_kwargs = dict(eval_set=[
    #         (X.iloc[i_fold], y.iloc[i_fold]),
    #         (X.iloc[i_oof], y.iloc[i_oof]),
    # ])

    if m_name.startswith('xgb'):
        fit_kwargs['verbose'] = False

    m.fit(X.iloc[i_fold], y.iloc[i_fold], **fit_kwargs)

    filename = f'{MODEL_PATH}/{NAME}-{m_name}-{fold_n}.joblib'
    joblib.dump(m, filename)
    print(f'fold {fold_n} wrote {filename}')


def fit():
    print('\nfit')
    Xs = get_Xs('train')
    ys = get_ys()
    kfold = create_kfold()

    m_constructors = {
        'xgb': xgb.XGBRegressor,
        'lgb': lgb.LGBMRegressor,
        'cb': cb.CatBoostRegressor,
    }

    models = []

    for pipe in PIPES:
        Model = m_constructors[pipe['m'].split('_')[0]]
        if pipe['m'].startswith('cb') and pipe['X'] in ['raw', 'label']:
            pipe['kwargs']['cat_features'] = Xs[pipe['X']].select_dtypes('category').columns.to_list()

        print(f'{Model.__name__}')

        for k, v in pipe['kwargs'].items():
            print(f'  {k}={v}')

        models.append((
            pipe['m'],
            Model(**pipe['kwargs']),
            Xs[pipe['X']],
            ys[pipe['y']],
        ))

    args = []

    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(Xs['raw'])):

        for m_name, m, X, y in models:
            args.append((X, y, fold_n, i_fold, i_oof, m_name, m))

    if RUNNING_ON_KAGGLE:
        list(starmap(fit_fold_model, args))
    else:
        from multiprocessing import Pool
        with Pool(min(os.cpu_count(), kfold.get_n_splits() * len(models))) as pool:
            pool.starmap(fit_fold_model, args)


def calc_score(train, y_pred_oof):
    merged_df = train[['race_group', 'efs_time', 'efs']].assign(prediction=y_pred_oof)
    merged_df = merged_df.reset_index()
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []

    for race in merged_df_race_dict.keys():
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]

        c_index_race = concordance_index(merged_df_race['efs_time'],
                                         -merged_df_race['prediction'],
                                         merged_df_race['efs'])

        metric_list.append(c_index_race)

    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))


def cv_score():
    print('\nCV score')
    train = read_csv(f'{C_PATH}/train.csv')
    Xs = get_Xs('train')
    kfold = create_kfold()
    y_pred_oofs = defaultdict(lambda: np.zeros(len(train)))

    for fold_n, (i_fold, i_oof) in enumerate(kfold.split(train)):
        for pipe in PIPES:
            m_name = pipe['m']
            for fn in glob(f'{MODEL_PATH}/{NAME}-{m_name}-{fold_n}.joblib'):
                m = joblib.load(fn)
                print(f'read {fn}')
                y_pred_oofs[pipe['m']][i_oof] = m.predict(Xs[pipe['X']].iloc[i_oof])

    for m_name, raw_risk_score in y_pred_oofs.items():
        y_pred_oofs[m_name] = rankdata(raw_risk_score)

    fn = sorted(glob(f'{LB_PATH}/*.csv'))[-1]
    lb = pd.read_csv(fn)
    print(f'read {fn}')
    lb = np.sort(lb['Score'])

    def leaderboard_percentile(score):
        return np.searchsorted(lb, score) / len(lb)

    scores = []

    for r in range(1, len(y_pred_oofs)+1):
        for m_combo in combinations(y_pred_oofs.items(), r):
            m_names, ranked_risk_scores = zip(*m_combo)
            m_names = '-'.join(m_names)
            score = calc_score(train, sum(ranked_risk_scores))
            lb_pct = leaderboard_percentile(score)
            scores.append((score, lb_pct, m_names))

    scores = sorted(scores)
    fn = f'{CSV_PATH}/scores.csv'

    with open(fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['score','leaderboard_percentile','models'])
        w.writerows(scores)
        print(f'wrote {fn}')

    for score, lb_pct, m_names in scores:
        print(f'{score:.4f} {lb_pct:.4f} {m_names}')

    if PROVE_MY_SCORE_IS_SAME:
        print('-'*79)
        import metric
        solution = train[['race_group', 'efs', 'efs_time']].reset_index()
        row_id_column_name = 'ID'
        scores = []

        for r in range(1, len(y_pred_oofs)+1):
            for m_combo in combinations(y_pred_oofs.items(), r):
                m_names, ranked_risk_scores = zip(*m_combo)
                m_names = '-'.join(m_names)
                submission = pd.DataFrame(sum(ranked_risk_scores), index=train.index, columns=['prediction']).reset_index()
                score = metric.score(solution.copy(), submission.copy(), row_id_column_name)
                lb_pct = leaderboard_percentile(score)
                scores.append((score, lb_pct, m_names))

        scores = sorted(scores)

        for score, lb_pct, m_names in scores:
            print(f'{score:.4f} {lb_pct:.4f} {m_names}')

def predict():
    print('\npredict')

    fn = f'{CSV_PATH}/scores.csv'
    row_models_len_max = 0

    with open(fn, newline='') as f:
        for row in csv.DictReader(f):
            values = lambda: (float(row['score']),
                              float(row['leaderboard_percentile']),
                              row['models'].split('-'))

            if not SUBMIT_FULL_ENSEMBLE:
                cv_score, lb_pct, m_names = values()
                continue

            if row_models_len_max < len(row['models']):
                cv_score, lb_pct, m_names = values()
                m_names = row['models'].split('-')

    msg = 'full ensemble' if SUBMIT_FULL_ENSEMBLE else 'ensemble with best CV score'
    print(f'using {msg}: {cv_score:.4f} {lb_pct:.4f} {m_names}')

    Xs = get_Xs('test')
    m_name_Xs = []

    for pipe in PIPES:
        if pipe['m'] in m_names:
            m_name_Xs.append((pipe['m'], Xs[pipe['X']]))

    y_preds = defaultdict(partial(np.zeros, len(Xs['raw'])))

    for m_name, X in m_name_Xs:
        for fn in glob(f'{MODEL_PATH}/{NAME}-{m_name}-*.joblib'):
            m = joblib.load(fn)
            print(f'read {fn}')
            y_preds[m_name] += m.predict(X)

    kfold = create_kfold()

    for m_name in y_preds.keys():
        y_preds[m_name] /= kfold.get_n_splits()
        y_preds[m_name] = rankdata(y_preds[m_name])

    y_pred = sum(y_preds.values())
    s = pd.DataFrame(y_pred, index=Xs['raw'].index, columns=['prediction'])
    s.to_csv('submission.csv')
    print('wrote submission.csv')


if __name__ == '__main__':
    if not RUNNING_ON_KAGGLE:
        if sys.argv[1:] and sys.argv[1] == 'preprocess':
            preprocess()
            sys.exit()

        if sys.argv[1:] and sys.argv[1] == 'cv_score':
            cv_score()
            sys.exit()

        if sys.argv[1:] and sys.argv[1] == 'predict':
            predict()
            sys.exit()

        for fn in glob(f'{CSV_PATH}/*.csv'):
            os.remove(fn)

        for fn in glob(f'{MODEL_PATH}/*.joblib'):
            os.remove(fn)

        preprocess()
        fit()
        cv_score()
        predict()
        sys.exit()

    # running on kaggle
    preprocess()
    if INCLUDE_FIT_ON_KAGGLE:
        fit()
    cv_score()
    predict()

    shutil.rmtree(CSV_PATH)

    if INCLUDE_FIT_ON_KAGGLE:
        shutil.rmtree('catboost_info')
        shutil.rmtree(MODEL_PATH)
