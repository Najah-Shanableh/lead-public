#!/usr/bin/python

import yaml
import sys

with open(sys.argv[1]) as f:
    param_dicts = yaml.load(f)
    
outputs = param_dicts['outputs']
n = 0
for output in outputs:
    m = 1
    m *= reduce(lambda x,y: x*y, map(lambda x: len(x), output['transform'].values()))
    m *= reduce(lambda x,y: x*y, map(lambda x: len(x), output['model'].values()))
    n += m

print 'Running {} models...'.format(n)
