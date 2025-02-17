#!/usr/bin/env python

import json
import joblib
import xgboost as xgb

d_path = '../input/hct-conduit'

m = xgb.XGBRegressor()
m.fit([[0]], [1])
config = {0: 1}
filename = 'conduit-00'
with open(f'{d_path}/{filename}.json', 'w') as f:
    json.dump(config, f)
joblib.dump(m, f'{d_path}/{filename}.joblib')

del config, m

with open(f'{d_path}/{filename}.json') as f:
    config = json.load(f)
m = joblib.load(f'{d_path}/{filename}.joblib')
print(config)
y_pred = m.predict([0])
print(y_pred)
