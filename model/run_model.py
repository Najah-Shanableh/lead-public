#!/usr/bin/python

import yaml
import pandas as pd
from sklearn.externals import joblib
import os
import argparse
from copy import deepcopy

import model
import util

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(description='Use this script to run a single model.')
parser.add_argument('--file', action='store_true')
parser.add_argument('input', type=str, help='filename or yaml string')
args = parser.parse_args()

if args.file:
    with open(args.input) as f:
        params_orig = yaml.load(f)
else:
    params_orig =  yaml.load(args.input.replace('\\n', '\n'))

params = deepcopy(params_orig)
data_name = params['data'].pop('name')
model_name = params['model'].pop('name')

print 'Loading ' + data_name
print '    with parameters ' + str(params['data'])

lead_data = util.get_class(data_name)(**params['data'])

lead_data.read()

print 'Tranforming with parameter ' + str(params['transform'])
lead_data.transform(**params['transform'])

train,test = lead_data.cv

print 'Training ' + model_name
print '    with parameters ' + str(params['model'])
print '    on ' + str(train.sum()) + ' examples'
print '    with ' + str(len(lead_data.X.columns)) + ' features'

estimator = util.get_class(model_name)(**params['model'])

estimator.fit(lead_data.X[train],lead_data.y[train])

print 'Validating model'
print '    on ' + str(test.sum()) + ' examples'

y_score = pd.Series(model.y_score(estimator, lead_data.X[test]), index=lead_data.X[test].index)

counts = [.01,.02,.05,.1]
precisions = model.precision(lead_data.y[test], y_score, [.01,.02,.05,.1,.2])

print '    baseline: ' + str(model.baseline(lead_data.y[test]))
print '    precision: ' + str(', '.join('%s=%.2f' % t for t in zip(counts, precisions)))
print '    auc: ' + str(model.auc(lead_data.y[test], y_score))

if 'output' in params:
    print 'Writing results in ' + params['output']
    if not os.path.exists(params['output']):
        os.makedirs(params['output'])
        
    with open( os.path.join(params['output'], 'params.yaml'), 'w') as outfile:
        yaml.dump(params_orig, outfile)
    joblib.dump(estimator, os.path.join(params['output'], 'estimator.pkl'))
    y = pd.DataFrame({'score':y_score, 'true': lead_data.y[test]}, index=y_score.index)
    y.to_csv(os.path.join(params['output'], 'y.csv'), index=True)
    pd.DataFrame(columns=lead_data.X.columns).to_csv(os.path.join(params['output'], 'columns.csv'),index=False)
