import sqlalchemy
import os
import numpy as np
import datetime

def create_engine():
    return sqlalchemy.create_engine('postgresql://{user}:{pwd}@{host}:5432/{db}'.format(
            host=os.environ['PGHOST'], db=os.environ['PGDATABASE'], user=os.environ['PGUSER'], pwd=os.environ['PGPASSWORD']))

def get_class(name):
    i = name.rfind('.')
    cls = name[i+1:]
    module = name[:i]
    
    mod = __import__(module, fromlist=[cls])
    return getattr(mod,cls)

def init_object(name, **kwargs):
    return get_class(name)(**kwargs)

def randtimedelta(low, high, size):
    d = np.empty(shape=size, dtype=datetime.timedelta)
    r = np.random.randint(low, high, size=size)
    for i in range(size):
        d[i] = datetime.timedelta(r[i])
    return d

def randdates(start,end, size):
    d = np.empty(shape=size, dtype=datetime.datetime)
    r = randtimedelta(0, (end-start).days, size)
    for i in range(size):
        d[i] = start + r[i]
    return d
