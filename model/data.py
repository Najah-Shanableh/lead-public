import pandas as pd
from sklearn import preprocessing
import re
import numpy as np
import collections
import random
import datetime
import model
from lead.output.aggregate import aggregate
import util
import warnings

CATEGORY_CLASSES = {
    'address_building_condition': ['SOUND', 'NEEDS MINOR REPAIR',
        'NEEDS MAJOR REPAIR', 'UNINHABITABLE'],
    'kid_sex' : ['M', 'F'],
    'test_type': ['V', 'C'],
    'surname_ethnicity': ['black', 'white', 'api', 'aian', 'p2race'],
    'kid_ethnicity': ['black', 'white', 'hispanic', 'asian'],
    'tract_ethnicity': ['asian', 'black', 'white', 'latino'],
    'ward_ethnicity': ['black', 'white', 'latino'], # there are no majority asian wards
    'ward_id': range(1,51),
    'community_area_id': range(1,78)
}


class ModelData(object):
    def read(self, **args):
        raise NotImplementedError
    
    def write(self, **args):
        raise NotImplementedError
        
    def transform(self):
        raise NotImplementedError

# train and test years control the date of the blood test 
class AddressData(ModelData):
    
    EXCLUDE = {'source', 'date', 'address', 'address_id', 'census_tract_id'}
    
    def __init__(self, source, date, period_days, train_years, test_years, directory=None, max_age=3*365, min_age=90, undersample=1, exclude = {}):
        self.source = source
        self.directory = directory
        
        self.date = date
        self.period = [datetime.timedelta(period_days[0]), datetime.timedelta(period_days[1])]
        
        self.max_age = max_age
        self.min_age = min_age
        self.undersample = undersample
    
        self.exclude = self.EXCLUDE.union(exclude)
    
        self.date_from = self.date - datetime.timedelta(train_years*365)
        self.date_to = self.date + datetime.timedelta(test_years*365)
        
        print self.date_from
        print self.date_to
    
    # TODO move this to ModelData    
    def read(self, **args):
        if self.source == 'csv':
            self.read_csv()
        elif self.source == 'sql':
            self.read_sql()
        else:
            raise ValueError('Unexpected data source: ' + str(self.source))
        
    def read_sql(self):
        engine = util.create_engine()
        print 'reading addresses'
        self.addresses = pd.read_sql('select * from output.addresses', engine)
        
        print 'reading kids'
        self.kids = pd.read_sql('select * from output.kid_addresses where minmax AND ('
                        '((date_of_birth between \'{date_from}\' and \'{date_to}\') and ( (min_test_date - date_of_birth) between {min_age} and {max_age})) OR ' 
                        '((date_of_birth between \'{date_from}\' and \'{date_to}\') and ( (max_test_date - date_of_birth) between {min_age} and {max_age})))'
                        .format(date_from=self.date_from, date_to=self.date_to,
                        min_age=self.min_age,max_age=self.max_age), engine)
        
    def transform(self):    
        print 'postitives'
        # random date for positive training examples
        train_mask = self.kids.min_test_date < self.date
        
        dob_noise = self.kids.date_of_birth.values + util.randtimedelta(self.period[0].days, self.period[1].days, len(self.kids))
        
        positive = self.kids.reindex(columns=['address_id'])
        positive['date'] = pd.Series(dob_noise, index=self.kids.index).where(train_mask, self.date)
        positive['kid_present'] = True
        
        # TODO allow for oversampling positives?
        
        print 'negatives'
        # DOING random addresses and dates for negative training examples
        # TODO don't use uniform distribution over addresses. weighted by population? etc.
        # n_samples = train_mask.sum()
        #ids = [self.addresses.address_id.values[np.random.randint(len(self.addresses))] for i in range(n_samples)]
        #dates = randdates(self.date_from, self.date, n_samples)
        #kid_present = np.empty(shape=n_samples, dtype=np.bool)
        
        #print(n_samples)
        
        address_ids = set(self.addresses.address_id)
        positive_address_ids = set(self.kids[train_mask].address_id.values)
        
        negative_address_ids = list(address_ids.difference(positive_address_ids))
        dates = util.randdates(self.date_from, self.date, len(negative_address_ids))
        
        negative = pd.DataFrame({'address_id':negative_address_ids, 'date':dates})
        negative['kid_present'] = False
        
        train = pd.concat([positive[train_mask],negative], ignore_index=True)
        
        # METHOD 1: iterate over postiive examples and flip bits
        #training = self.kids[train_mask]
        #ids_series = pd.Series(ids)
        #for i,row in enumerate(training.values):
        #    if i%1000==0: print(i)
        #    neg_idx = ids_series[ids_series.address_id == row[0]]
        #    for i in neg_idx.index:
        #        delta = (row[1] - dates[i])
        #        if (delta < self.period[1]) and (delta > self.period[0]):
        #            kid_present[i] = True
        
        # METHOD 2: iterate over potential negatives and do lookup
        #for i in range(n_samples):
        #    if i%1000==0: print(i)
        #    d = self.kids.date_of_birth[self.kids.address_id == ids[i]]
        #    delta = (d - dates[i])
        #    mask = (delta < self.period[1]) & (delta > self.period[0])
        #    kid_present[i] = mask.sum()
            
        #self.negative = pd.DataFrame({'address_id':ids, 'date':dates, 'kid_present':kid_present})
        
        positive_address_ids = set(positive[~train_mask].address_id)
        negative_address_ids = address_ids.difference(positive_address_ids)
        
        positive = pd.DataFrame({'address_id':list(positive_address_ids)})
        positive['kid_present'] = True
        
        negative = pd.DataFrame({'address_id':list(negative_address_ids)})
        negative['kid_present'] = False
        
        test = pd.concat([positive, negative], ignore_index=True)
        test['date'] = self.date
        
        train['train'] = True
        test['train'] = False
        
        df = pd.concat([train, test], ignore_index = True)
        df = df.merge(self.addresses, on='address_id', how='left')
        
        train = df['train']
        df.drop(['train'], axis=1, inplace=True)
        self.cv = (train, ~train)
        self.df = df # TODO remove, for debugging purposes!
        
        X,y = Xy(df=df, y_column='kid_present', exclude=self.exclude)
        self.X = X
        self.y = y
        
    def write_csv(self, filename):
        self.kids.to_csv(filename + '/kids.csv')
        self.addresses.to_csv(filename + '/addresses.csv')
        
    def read_csv(self, filename):
        self.kids = pd.read_csv(filename + '/kids.csv')
        self.addresses = pd.read_csv(filename + '/addresses.csv')

class LeadData(ModelData):
    # default exclusions set
    # TODO: organize and explain these
    EXCLUDE = {'kid_id', 'kid_first_name', 'kid_last_name', 'test_id', 'test_type', 
       'test_kid_age_days', 'test_date', 'test_minmax', 'test_maxmax', 'test_min', 'address_id', 'census_tract_id',
               'year', 'join_year', 'kid_birth_days_to_test', 'kid_date_of_birth', 'address_inspection_init_days_to_test', 'address_method',
               'test_number', 'minmax_test_number', 'test_bll',
    }
    
    TRACT_YEAR_COLUMNS = {
            'test_count': {'numerator': 'test_count'},
            'kid_count' : {'numerator': 'test_kid_count'},
            'ebll_test_count' : {'numerator': 'ebll_test_count'},
            'ebll_test_ratio' : {'numerator': 'ebll_test_count', 'denominator':'test_count'},
            'avg_ebll' : {'numerator': lambda ty: ty.avg_ebll * ty.ebll_test_count, 'denominator' : 'ebll_test_count'},
            'ebll_kid_count' : {'numerator': 'ebll_kid_count'},
            'ebll_kid_ratio' : {'numerator': 'ebll_kid_count', 'denominator':'test_kid_count'},
            
            'inspection_count' : {'numerator' : 'inspection_count'},
            'hazard_int_count' : {'numerator' : 'hazard_int_count'},
            'hazard_ext_count' : {'numerator' : 'hazard_ext_count'},
            'hazard_int_ratio' : {'numerator' : 'hazard_int_count', 'denominator': 'inspection_count'},
            'hazard_ext_ratio' : {'numerator' : 'hazard_ext_count', 'denominator': 'inspection_count'},
            'compliance_count' : {'numerator' : 'compliance_count'},
            'compliance_ratio' : {'numerator' : 'compliance_count', 'denominator': 'inspection_count'},
            # the next two features are incorrect, need to be weighted...
            'avg_init_to_comply_days' : {'numerator' : 'avg_init_to_comply_days'},
            'pct_inspected' : {'numerator' : 'pct_inspected'}
    }
    
    KIDS_DATE_COLUMNS = ['kid_date_of_birth', 'test_date', 
                    'address_inspection_init_date', 'address_inspection_comply_date']
    
    def __init__(self, source, date_from, date_to, directory=None, 
                 tables=['inspections', 'tracts', 'wards', 'tract_year', 'ward_year', 'addresses']):
        self.source = source
        self.directory = directory
        
        self.date_from = date_from
        self.date_to = date_to
        
        self.tables = {t:None for t in tables}
    
    def read(self, **args):
        if self.source == 'pkl':
            self.read_pkl()
        elif self.source == 'sql':
            self.read_sql()
        elif self.source == 'csv':
            self.read_csv()
        else:
            raise ValueError('Unexpected data source: ' + str(self.source))
    
    def read_sql(self):
        engine = util.create_engine()
        
        self.kids = pd.read_sql('select * from output.kids where '
                        'test_date between \'{date_from}\' and \'{date_to}\' '
                        .format(date_from=self.date_from, date_to=self.date_to), engine, index_col='test_id')
        
        for table in self.tables:
            self.tables[table] = pd.read_sql('select * from output.' + table, engine)
            
    def write(self):
        self.kids.to_pickle(self.directory + '/kids.pkl')
        
        for table in self.tables:
            self.tables[table].to_pickle(self.directory + "/" + table + '.pkl')
        
    def read_pkl(self):
        self.kids = pd.read_pickle(self.directory + "/kids.pkl")
        
        for table in self.tables:
            self.tables[table] = pd.read_pickle(self.directory + "/" + table + '.pkl')
    
    def read_csv(self):
        self.kids = pd.read_csv(self.directory + "/kids.csv", index_col='test_id')
        
        for c in self.KIDS_DATE_COLUMNS:
            self.kids[c] = self.kids[c].apply(lambda x: 
                    x if pd.isnull(x) else datetime.datetime.strptime(x, '%Y-%m-%d').date())
        
        for table in self.tables:
            self.tables[table] = pd.read_csv(self.directory + "/" + table + '.csv')
            
    def transform(self, year, train_years,
                kid, tract, address, ward,
                tract_years, # number of years of tract history
                address_history, # boolean true or falsea
                bll_threshold=5, max_age=3*365, min_age=90, require_address=True, 

                # these parameters have defaults that were established by testing
                tract_years_scale = True, # whether or not to normalize each year of tract data
                join_year = True, # use only data up to test year (vs all data available)
                training='minmax',
                community_area = False, # don't include community area binaries
                exclude={}, 
                undersample=1,
                impute=True, normalize=True,
                census_tract_binarize=False,
                ward_id = None,
                building_year_decade=True,
                test_date_season=True,
                exclude_addresses=[203008]): # exclude Aunt Martha's health center

        exclude = self.EXCLUDE.union(exclude)
        age_mask = (self.kids.test_kid_age_days >=  min_age) & (self.kids.test_kid_age_days <= max_age)
        df = self.kids[age_mask]
        if ward_id is not None:
            df = df[df.ward_id==ward_id]
            
        if  require_address:
            df = df[df.address_id.notnull() & df.ward_id.notnull() & df.census_tract_id.notnull() & df.community_area_id.notnull()]
            exclude.update({'address_null', 'ward_null', 'census_tract_null', 'community_area_null'})

        if exclude_addresses is not None:
            df = df[~df.address_id.isin(exclude_addresses)]
        tests_subset = df # used below for test aggregation
        
        # get tests relevant to date
        today = datetime.date(year, 1, 1)
        date_from = datetime.date(year - train_years, 1, 1)
        date_to = datetime.date(year, 1, 1) + datetime.timedelta(max_age)
        
        if date_from < self.date_from:
            raise ValueError('transform dates out of range')
        
        if date_to > self.date_to:
            warnings.warn('today + max_days is outside of datasource')
        
        date_mask = (df.test_date >= date_from) & (df.test_date < date_to)
        df = df[date_mask]

        # cross validation
        train = (df.test_date < today) 
        if training == 'minmax':
            train = train & (df.test_minmax)
        elif training == 'pre_minmax':
            train = train & (df.test_number <= df.minmax_test_number)
        elif training != 'all':
            raise ValueError("Invalid training option: " + str(training))
        
        test = (df.test_date >= today) & (df.kid_date_of_birth < today) & \
            ((df.test_minmax))# | (df.test_maxmax & df.test_bll <= 5))

        train_or_test = train | test
        train = train.loc[train_or_test]
        test = test.loc[train_or_test]
        self.cv = (train,test)
        df = df[train_or_test].reset_index().copy()
        
        # if inspection date is in the future make it null and make the inspection_null true
        # note this is only using the *first* inspection date, which is usually close to, if not equal to, the last, but not always
        past_inspection = (df['address_inspection_init_date'] < today)
        df['address_inspection_init_date'] = df['address_inspection_init_date'].where(past_inspection)
        df['address_inspection_hazard_int'] = df['address_inspection_hazard_int'].where(past_inspection)
        df['address_inspection_hazard_ext'] = df['address_inspection_hazard_ext'].where(past_inspection)
        df['address_inspection_null'] = df['address_inspection_null'] | (~past_inspection)
        
        past_compliance = (df['address_inspection_comply_date'] < today)
        df['address_inspection_comply_date'] = df['address_inspection_comply_date'].where(past_compliance)
        df['address_inspection_compliance'] = df['address_inspection_comply_date'].notnull().where(~df['address_inspection_null'])

        epoch = datetime.date.fromtimestamp(0)
        mean_age = datetime.timedelta(df[df['test_date'] < today]['test_kid_age_days'].mean())
        past_test = df.test_date < today
        future_test_date = (df.kid_date_of_birth + mean_age).apply(lambda d: max(d, today))
        pseudo_test_date = df.test_date.where(past_test, future_test_date)
        
        df['kid_date_of_birth_month'] = df['kid_date_of_birth'].apply(lambda d: d.month)
        df['kid_birth_date'] = df['kid_date_of_birth']
    
        for c in ['address_inspection_init', 'address_inspection_comply', 'kid_birth']:
            df[c + '_days'] = df[c + '_date'].apply(lambda d: None if pd.isnull(d) else (d - epoch).days)
            df[c + '_days_to_test'] = pd.to_timedelta((pseudo_test_date - df[c + '_date']), 'D').astype(int)
            df.drop(c + '_date', axis=1, inplace=True)
        
        df['join_year'] = df.year.apply(lambda y: min(y-1, year-1)) if join_year else year-1
        
        if test_date_season:
            df['test_date_month'] = df['test_date'].apply(lambda d: d.month).where(df['test_date'] < today)
            CATEGORY_CLASSES['test_date_month'] = range(1,13)

        # spatial 
        if not kid:
            exclude.add('kid_.*')
        if not address:
            exclude.update(['address_building_.*', 'address_assessor_.*', 'address_lat', 'address_lng'])
        if tract:
            prefix_columns(self.tables['tracts'], 'tract_', 'census_tract_id')
            df = df.merge(self.tables['tracts'], on=['census_tract_id'], how='left', copy=False)
        if not ward:
            exclude.add('ward_id')
        if not community_area:
            exclude.add('community_area_id')
        
        # spatio-temporal
        inspections_tract = self.aggregate_inspections(range(year-train_years, year), levels=['census_tract_id'])[0]
        tests_tract = self.aggregate_tests(levels=['census_tract_id'], df=tests_subset)[0]

        prefix_columns(inspections_tract, 'tract_inspections_cumulative_')
        prefix_columns(tests_tract, 'tract_tests_1y_')

        spacetime = inspections_tract.join(tests_tract, how='outer')
        spacetime.fillna(0, inplace=True)

        if tract_years_scale:
            spacetime = spacetime.groupby(level='year').apply(lambda x: pd.DataFrame(preprocessing.scale(x), index=x.index, columns=x.columns))

        df = df.merge(spacetime, left_on=['census_tract_id','join_year'], right_index=True, how='left', copy=False )

        # additional features
        if not address_history:
            exclude.update(['address_inspection_.*', 'address_test_.*'])
        
        if census_tract_binarize:
            exclude.remove('census_tract_id')
            CATEGORY_CLASSES['census_tract_id'] = self.tables['tracts'].census_tract_id.values

        if building_year_decade:
            df['address_building_year_decade'] = (df['address_building_year'] // 10)
            CATEGORY_CLASSES['address_building_year_decade'] =  df['address_building_year_decade'].unique()


        df.set_index('test_id', inplace=True)
        X,y = Xy(df, y_column = 'minmax_bll', exclude=exclude, impute=impute, normalize=normalize, train=train)
 
        self.X = X
        self.y = y >  bll_threshold
        
        if undersample > 1:
            self.cv = (undersample_cv(df, self.cv[0], 1.0/undersample), self.cv[1])

    def aggregate_inspections(self, years, levels):
        res_columns = {'res_count': {'numerator': 'residential'}}
        inspections = self.tables['inspections']
        for index in levels:
            count = aggregate(self.tables['addresses'], res_columns, index=index)
            count.columns = [index + '_res_count']
            inspections = inspections.merge(count.reset_index(), how='left', on=index)
        
        for column in ['hazard_int','hazard_ext']:
            inspections[column].fillna(True, inplace=True)
            inspections[column] = inspections[column].astype(int)
        
        inspections['y'] = inspections['init_date'].fillna(inspections['comply_date']).apply(lambda d:d.year)
        inspections_index = inspections[['y']].reset_index()
        inspections_index.index = np.zeros(len(inspections_index))
        years = pd.Series(years, name='year')
        years.index = np.zeros(len(years))
        
        inspections_index = inspections_index.join(years)
        
        cond = lambda df: ((df['y'] <= df['year']))
        inspections2 = inspections.merge(inspections_index[cond(inspections_index)], left_index=True, right_on='index')
        inspections2.set_index([range(len(inspections2))], inplace=True)
        
        comply_not_null = inspections2[inspections2.comply_date.notnull()]
        inspections2['comply'] = (comply_not_null['comply_date'].apply(lambda d: d.year) < comply_not_null.year)
        inspections2['comply'] = inspections2['comply'].fillna(False).astype('int')
        dt = (inspections2['comply_date'] - inspections2['init_date']).where(inspections2['comply'])
        inspections2['days_to_compliance'] = dt[dt.notnull()] / np.timedelta64(1, 'D')
        
        INSPECTION_COLUMNS = {
            'count': {'numerator':1},
            'hazard_int_count': {'numerator':'hazard_int'},
            'hazard_ext_count': {'numerator':'hazard_ext'},
            'hazard_int_ratio': {'numerator':'hazard_int', 'denominator':1},
            'hazard_ext_ratio': {'numerator':'hazard_ext', 'denominator':1},
            'compliance_count': {'numerator': 'comply'},
            'compliance_ratio': {'numerator': 'comply', 'denominator': 1},
            'avg_init_to_comply_days': {'numerator': 'days_to_compliance', 'func':'mean'},
        }
        
        r = []
        for level in levels:
            INSPECTION_COLUMNS['pct_inspected'] = {'numerator': 1, 'denominator': level + '_res_count', 'denominator_func': np.max}
            r.append(aggregate(inspections2, INSPECTION_COLUMNS, index=[level,'year']))
            
        return r
    
    def aggregate_tests(self, levels, df=None):
        ebll_test_count = lambda t: (t.test_bll > 5).astype(int)
        ebll_kid_count = lambda t: ((t.test_bll > 5) & t.test_minmax).astype(int)
        TEST_COLUMNS = {
            'test_count': {'numerator': 1},
            'kid_count': {'numerator': 'test_minmax'},
            'ebll_test_count': {'numerator': ebll_test_count},
            'ebll_test_ratio': {'numerator': ebll_test_count, 'denominator': 1},
            'avg_ebll': {'numerator': lambda t: t.test_bll.where(t.test_bll > 5, np.nan), 'denominator': ebll_test_count}, # use np.nan because 0/0 is infinity but nan/0 is nan...
            'ebll_kid_count': {'numerator': ebll_kid_count},
            'ebll_kid_ratio': {'numerator': ebll_kid_count, 'denominator': 'test_minmax'}
        }
        
        if df is None: df = self.kids
        return [aggregate(df, TEST_COLUMNS, index=[level, 'year']) for level in levels]
 
def aggregate_years(df, columns, index, years, max_year, min_year, scale): 
    tys = []     
    for y in range(min_year, max_year+1):
        ty = df[(df.year <= y) & (df.year > y-years)]
        ty = aggregate(ty, columns, index=index)
        ty.fillna(0, inplace=True) # fillna(0) since na implies no records
        if scale:
            ty = pd.DataFrame(preprocessing.scale(ty), index=ty.index, columns=ty.columns)
        ty['join_year'] = y
        tys.append(ty.reset_index())
    df2 = pd.concat(tys, ignore_index=True)
    return df2

# generate year, month, day features from specified date features
def expand_dates(df, columns=[]):
    columns=df.columns.intersection(columns)
    df2 = df.reindex(columns=set(df.columns).difference(columns))
    for column in columns:
        df2[column + '_year'] = df[column].apply(lambda x: x.year)
        df2[column + '_month'] = df[column].apply(lambda x: x.month)
        df2[column + '_day'] = df[column].apply(lambda x: x.day)
    return df2

# binarize specified categoricals using specified category class dictionary
def binarize(df, category_classes):
    #for category,classes in category_classes.iteritems():
    columns = set(category_classes.keys()).intersection(df.columns)
    
    for category in columns:
        classes = category_classes[category]
        for i in range(len(classes)-1):
            df[category + '_' + str(classes[i]).replace( ' ', '_')] = (df[category] == classes[i])
        #binarized = pd.get_dummies(df[category], prefix=df[category])#.drop(classes[len(classes)-1], axis=1, inplace=True)
        #df = df.merge(binarized, left_index=True, right_index=True, copy=False)
        
    df = df.drop(columns, axis=1, copy=False)                                      
    return df

# returns endogenous and exogenous variables
# normalization requires imputation (can't normalize null values)
# training mask is used for normalization
def Xy(df, y_column, include=None, exclude=None, impute=True, normalize=True, train=None):
    X = select_features(df, include, exclude)
    
    X = binarize(X, CATEGORY_CLASSES)
    
    nulcols = null_columns(X)
    if len(nulcols) > 0:
        print 'Warning: null columns ' + str(nulcols)
    
    nonnum = non_numeric_columns(X)
    if len(nonnum) > 0:
        print 'Warning: non-numeric columns ' + str(nonnum)
    
    if impute:
        imputer = preprocessing.Imputer()
        Xfit = X[train] if train is not None else X
        imputer.fit(Xfit)
        d = imputer.transform(X)
        if normalize:
            d = preprocessing.StandardScaler().fit_transform(d)
        X = pd.DataFrame(d, index=X.index, columns = X.columns)

    y = (df[y_column]).astype(int)
    X.drop(y_column, axis=1, inplace=True)
    
    return (X,y)

def select_features(df, include=None, exclude=None, regex=True):

    if isinstance(include, collections.Iterable):
        columns = set.union(*[set(filter(re.compile('^' + feature + '$').match, df.columns)) for feature in include])
    else: 
        columns = set(df.columns)
        
    if isinstance(exclude, collections.Iterable):
        d = set.union(*[set(filter(re.compile('^'  + feature + '$').search, df.columns)) for feature in exclude])
        columns = columns.difference(d)
    
    df = df.reindex(columns = columns)
    return df

def prefix_columns(df, prefix, ignore=[]):
    df.columns =  [prefix + c if c not in ignore else c for c in df.columns]

def get_collinear(df, tol=.1, verbose=False):
    q, r = np.linalg.qr(df)
    diag = r.diagonal()
    if verbose:
        for i in range(len(diag)):
            if np.abs(diag[i]) < tol:
                pass
                print r[:,i] # TODO print equation with column names!
    return [df.columns[i] for i in range(len(diag)) if np.abs(diag[i]) < tol]

def null_columns(df):
    nulcols = df.isnull().sum() == len(df)
    return nulcols[nulcols==True].index

def non_numeric_columns(df):
    columns = []
    for c in df.columns:
        try: 
            df[c].astype(float)
        except:
            columns.append(c)
            
    return columns

def get_correlates(df, c=.99):
    corr = df.corr().values
    for i in range(len(df.columns)):
        for j in range(i):
            if corr[i][j] >= c:
                print df.columns[i] + ', ' + df.columns[j] + ': ' + str(corr[i][j])
                
def undersample_cv(d, train, p):
    a = pd.Series([random.random() < p for i in range(len(d))], index=d.index)
    return train & ((d.test_bll > 5).values | a) 
