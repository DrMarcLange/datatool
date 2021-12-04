def dirty_typecast(x):
    from pandas import to_datetime as cv_
    x = str(x)
    try:
        dt = str(cv_(x))[:19]
        return str(dt)[:7]
    except:
        try:
            if abs(int(float(x)) - float(x)) < .3:
                return int(float(x))
            return float(x)
        except:
            return str(x)
            
def one_hot(s):
    import pandas as pd
    labels = sorted(s.unique())[:-1] #leave one out
    res = pd.DataFrame(index=s.index)
    for l in labels:
        res[s.name+'_oh_'+l] = s.apply(lambda x: 1 if x==l else 0)   
    return res
            
import json
with open('glue1.json','r') as fi:
    analysis_rules = json.load(fi)

from Data_Configuration import RawDataAgg
debug = 'DEBUG' in analysis_rules.keys() 
mockup = 'MOCKUP' in analysis_rules.keys()

if 'raw_data_config' in analysis_rules.keys():
    raw_data_config = analysis_rules['raw_data_config']
    if debug:
        raw_data_config['DEBUG'] = debug
    if mockup:
        raw_data_config['MOCKUP'] = mockup
    data_provider = RawDataAgg(**raw_data_config)
    global p,t
    p,t,p_names,t_names,p_cols,t_cols = data_provider.get_tables_and_names()

Process_Constraints=f"""
select * from p order by p.{raw_data_config['p_index_col']}""" 
# where p.RAW_AGG_created_at > '2010'
#   and p.RAW_AGG_last_at < '2023'
#   and p.RAW_AGG_step_count > 1 
#   and p.RAW_AGG_avg_step_duration > 0     
#   and p.RAW_AGG_max_step_duration > 0
#   and p.RAW_AGG_sum_step_duration > 0;"""

Process_Constraints_on_t=f"""
select p.{raw_data_config['p_index_col']} ix,  
       t.ACTIVITY_DE act,
       count(*) cnt,
       sum(RAW_AGG_duration) sum_dur,
       sum(RAW_AGG_d_in) sum_dur_in,
       sum(RAW_AGG_d_out) sum_dur_out
  from p join t 
on p.{raw_data_config['p_index_col']} = t.{raw_data_config['p_index_col']}
group by ix, act
"""
#where t.ACTIVITY_DE in ('act_qqrr_','act_rrss_','act_qrrs_','act_ppqq_',
#'act_qrrr_','act_sstt_')        
#group by ix, act
#having cnt > 2 and sum_dur > 2
#"""

Target_Definition=f"""
select p.{raw_data_config['p_index_col']}, coalesce(s.target,0) target 
  from p_constrained p 
  left join (select ix, 
               sum(case when act='act_qqrr_' then cnt else 0 end +
               case when act='act_rrss_' then cnt else 0 end) target
       from (select p.{raw_data_config['p_index_col']} ix,  
       t.ACTIVITY_DE act,
       count(*) cnt,
       sum(RAW_AGG_duration) sum_dur,
       sum(RAW_AGG_d_in) sum_dur_in,
       sum(RAW_AGG_d_out) sum_dur_out
  from p_constrained p join t 
on p.{raw_data_config['p_index_col']} 
 = t.{raw_data_config['p_index_col']}
group by ix, act) s group by ix
) s
  on s.ix = p.{raw_data_config['p_index_col']}

"""


from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
global p_constrained
p_constrained = pysqldf(Process_Constraints)
assert len(p_constrained)>0, "no training data satisfying the process constraints"
constraints_by_t = pysqldf(Process_Constraints_on_t)                   
ixs = constraints_by_t.ix.unique()
p_constrained = p_constrained[p_constrained[raw_data_config['p_index_col']].apply(lambda x: x in ixs)]
p_constrained = p_constrained.set_index(raw_data_config['p_index_col'])

for col in p_constrained.columns:
    p_constrained[col] = p_constrained[col].apply(dirty_typecast)

import pandas as pd
train_data = pd.DataFrame()        
col_types = dict(p_constrained.dtypes.apply(str))
for col in p_constrained.columns:
    if col_types[col] in ('int64','float64'):
        train_data[col] = p_constrained[col].copy()
    else:
        train_data = pd.concat([train_data,one_hot(p_constrained[col])],axis=1).copy()   

target = pysqldf(Target_Definition).set_index(raw_data_config['p_index_col'])
assert len(target) == len(train_data), "length mismatch, something broken?"

from polish_dataframe import make_uncorrelated_data

def repeat_regression(train_data, target):                                                       
    import statsmodels.api as sm
    #Apply logistic regression
    def logreg(train_data, target):
        results = sm.Logit(target,sm.add_constant(train_data)).fit()                                                           
        return results
    #Apply linear regression
    def linreg(train_data, target):
        results = sm.OLS(target,sm.add_constant(train_data)).fit()                                                             
        return results                                                                          
    max_p = 1                                                                                   
    while max_p > 0.05:                                                                         
        train_data = make_uncorrelated_data(train_data)                     
        try:                                                                                    
            result = logreg(train_data, target)
            if not result.mle_retvals['converged']:                                             
                raise ValueError
        except:                                                                                 
            result = linreg(train_data, target)
        res = result.pvalues; res = res[res.index!='const']   
        res_p = res.sort_values().tail(1)
        removal_candidate, max_p = res_p.index[0], res_p.values[0]
        print(removal_candidate, max_p)
        if max_p > 0.05:                              
            print("the variable to remove is: ", removal_candidate)     
            train_data = train_data.drop(columns=removal_candidate)                               
        else:
            print(result.summary2())
            break 
    return train_data

res = repeat_regression(train_data,target)
