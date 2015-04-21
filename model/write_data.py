#!/usr/bin/python

import sys
import yaml
import util
import os

directory = sys.argv[1]

if not os.path.exists(directory):
    os.makedirs(directory)

with open(sys.argv[2]) as f:
    params = yaml.load(f)
    
params['data']['directory'] = directory
engine = util.create_engine()

data_name = params['data'].pop('name')
data = util.get_class(data_name)(**params['data'])

data.read_sql()
data.write()
