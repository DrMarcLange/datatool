import datetime
import numpy as np; np.random.seed(42)
from numpy import log1p; intlog1p = lambda x: int(log1p(x))
import os
import pandas as pd
#import pyodbc ###braucht ihr wieder fürs sql, errors werden euch leiten :D
import sqlite3
from pandasql import sqldf
local_state = lambda seed: np.random.RandomState(seed)

class RawDataAgg():
    def __init__(self, **kwargs) -> None:
        self.kwargs=kwargs
        self.debug = 'DEBUG' in self.kwargs
        self.mockup = 'MOCKUP' in self.kwargs.keys()

        local_temps = {"_DEFAULT_local_TMP_FOLDER_":str(datetime.datetime.now())[:10],
                           "_DEFAULT_local_TMP_DB_":str(datetime.datetime.now())[:10]+'/local_agg.sqlite',
                          "_DEFAULT_local_P_TABLE_":'properties',
                          "_DEFAULT_local_T_TABLE_":'timeseries'}

        for k,v in local_temps.items():
            if k not in kwargs.keys():
                self.kwargs[k]=v

        logtuple=(
            str(datetime.datetime.now())[:16],
            self.kwargs['p_Table'],self.kwargs['p_index_col'],
            self.kwargs['t_Table'],self.kwargs['t_datetime_col'],
            str(local_temps)
            )
        loghdr='"'+'"\t"'.join(('created_at',
               'p_table','p_index_col',
               't_table','t_datetime','local_temps'))+'"'
        local_temps = {k:self.kwargs[k] for k in local_temps.keys()}
        logstr='"'+'"\t"'.join(logtuple)+'"'

        if not ( self.kwargs['_DEFAULT_local_TMP_FOLDER_'] in os.listdir() ):
            print('|------- Setting up local temporary sqlite folder\t  -|')
            os.mkdir( self.kwargs['_DEFAULT_local_TMP_FOLDER_'] )
        with open(self.kwargs['_DEFAULT_local_TMP_FOLDER_']+'/configlog.tsv','w') as fi:
            fi.write(loghdr)
            fi.write(logstr)

    @property
    def sql_connection(self, ):
        try:
            return self.global_con
        except AttributeError:
            assert 1==0, "Marc konnte hier locally nicht weiter, bitte wieder auskommentieren :D"
            #self.global_con = pyodbc.connect(self.kwargs['SQL_URI'])
            return self.global_con

    def _save_p_t_to_local__(self,p,t):
        sqlite = sqlite3.connect(self.kwargs['_DEFAULT_local_TMP_DB_'])
        p[['RAW_AGG_created_at','RAW_AGG_last_at']]=\
               p[['RAW_AGG_created_at','RAW_AGG_last_at']].applymap(lambda x: str(x)[:19])
        t[[self.kwargs['t_datetime_col'],'RAW_AGG_tm1','RAW_AGG_tp1']]=\
               t[[self.kwargs['t_datetime_col'],'RAW_AGG_tm1','RAW_AGG_tp1']].applymap(lambda x: str(x)[:19])
        p.to_sql(self.kwargs['_DEFAULT_local_P_TABLE_'], sqlite, if_exists='replace', index=False)
        t.to_sql(self.kwargs['_DEFAULT_local_T_TABLE_'], sqlite, if_exists='replace', index=False)

    @property
    def _read_local_p_t_(self,):
        sqlite = sqlite3.connect(self.kwargs['_DEFAULT_local_TMP_DB_'])
        p = pd.read_sql_query('select * from '+self.kwargs['_DEFAULT_local_P_TABLE_']+';', sqlite)
        t = pd.read_sql_query('select * from '+self.kwargs['_DEFAULT_local_T_TABLE_']+';', sqlite)
        p[['RAW_AGG_created_at','RAW_AGG_last_at']]=\
               p[['RAW_AGG_created_at','RAW_AGG_last_at']].applymap(lambda x: str(x)[:19])
        t[[self.kwargs['t_datetime_col'],'RAW_AGG_tm1','RAW_AGG_tp1']]=\
               t[[self.kwargs['t_datetime_col'],'RAW_AGG_tm1','RAW_AGG_tp1']].applymap(lambda x: str(x)[:19])        
        return p, t

    def _download_reference_p_t_(self,):
        p = pd.read_sql_table(self.kwargs['p_Table'], self.sql_connection)
        t = pd.read_sql_table(self.kwargs['t_Table'], self.sql_connection)
        return p, t

    def _mockup_(self,):
        print('|########-- mocking up fake data --########################|')
        global p,t
        pysqldf = lambda q: sqldf(q, globals())
        #wir wissen source.csv hat 58 spalten und brauchen nurn paar
        cols = list(range(58))
        df = pd.read_csv('source.csv',names=cols,skiprows=1)
        df['longstr'] = df[6]
        #shuffle for random times
        df = df.sample(len(df),random_state=local_state(0))
        act_type1 = lambda x: 'act_'+''.join(sorted(x.lower())[len(x)//2+25:len(x)//2+28])\
                    +'_'+''.join(sorted(x.lower())[len(x)//2+15:len(x)//2+20])
        act_type2 = lambda x: 'act_'+''.join(sorted(x.lower())[len(x)//2+5:len(x)//2+9])\
                    +'_'+''.join(sorted(x.lower())[len(x)//2+27:len(x)//2+31])
        act = lambda x: act_type1(x) if x[len(x)//2-len(x)//3]<'a' else act_type2(x)        
        # secrets = 
        # self.pysqldf = lambda q: sqldf(q, globals())
        df['ACTIVITY_DE'] = df.longstr.apply(act)
        
        l = local_state(80)
        df['_random_index'] = df[0]
        df['t_0'] = df[1].apply(lambda x: str(x).split('.')[0])
        df['s5_02'] = df[9]
        df['gaid'] = df[4]
        df['s5_01']= df[8]
        df['s5_09']= df[16]
        df = df.rename(columns={'gaid':self.kwargs['p_index_col'],
                                's5_09':'some_int',
                                's5_01':'some_small_int'})

        df[self.kwargs['t_datetime_col']] = \
            sorted(list(
                (df.t_0.apply(int)
                +df.t_0.apply(lambda x: 10000*l.randint(5))
                -3*10**7
                -(5*10**3)*df.s5_02).\
                           apply(lambda x:
                                 pd.to_datetime(
                                     str(pd.to_datetime(int((10**8.999)*x)))[:13]
                                    +':'+str(l.randint(60))
                                    +':'+str(l.randint(60))))))
                      
        t = df[[self.kwargs['p_index_col'],self.kwargs['t_datetime_col'],'ACTIVITY_DE',
                'some_int','some_small_int']].copy()        
                 
        l = local_state(44)
        
        t['agent_id'] = 0
        t['agent_id'] = t['agent_id'].apply(\
            lambda x: l.randint(1000)+l.randint(15)\
            +(l.randint(2)+l.randint(2))*1000 )
        
        t = t.sort_values(self.kwargs['t_datetime_col'],ascending=True\
            ).reset_index().drop(columns=['index'])
            
        p = pysqldf(\
       f"""select max({self.kwargs['p_index_col']}) {self.kwargs['p_index_col']},
                  round(avg(some_int%97)) SOME_ORD1,
                  sum(some_small_int%100+.95) CURRENT_TOTAL,
                  agent_id
           from t group by agent_id""")
           
        t = pysqldf( f"""select p.{self.kwargs['p_index_col']},                               
                                t.{self.kwargs['t_datetime_col']},
                                t.ACTIVITY_DE, t.some_int, t.some_small_int
                                from t join p on t.agent_id = p.agent_id
                                order by t.{self.kwargs['t_datetime_col']} desc""")
                                
        p = p.drop(columns=['agent_id',])
        p,t = p.copy(),t.copy()        
        return p, t

    @property
    def raw_path_tables(self,):
        try:
            self.p, self.t = self._read_local_p_t_
            if self.mockup:
                return self.p, self.t
            if self.debug:
                print('|== RAW:    loaded preaggregated process-data     \t ==|')
            return self.p, self.t
        except:            
            def one_time_aggregate_p_t(raw_p:pd.DataFrame,raw_t:pd.DataFrame):                                                
                            
                print('|== RAW:    preaggregating process-data           \t ==|')                                     
                from pandas import to_datetime as _cv
                global p,t,tmp
                p = raw_p.copy(); t = raw_t.copy();
                t[self.kwargs['t_datetime_col']] = \
                    t[self.kwargs['t_datetime_col']].apply(_cv)
                pysqldf = lambda q: sqldf(q, globals())
                
                tmp = pysqldf(\
                f"""SELECT {self.kwargs['p_index_col']},
                           {self.kwargs['t_datetime_col']},                            
                           ROW_NUMBER() OVER (PARTITION BY {self.kwargs['p_index_col']}
                              ORDER BY {self.kwargs['t_datetime_col']}) RAW_AGG_step,
                           t.*     
                    FROM t;""")                
                t = tmp.copy()
                
                tmp = pysqldf(f"""select tn0.{self.kwargs['p_index_col']}, tn0.RAW_AGG_step,                                        
                                         coalesce(tm1.{self.kwargs['t_datetime_col']},
                                                  tn0.{self.kwargs['t_datetime_col']}) RAW_AGG_tm1,
                                                  tn0.{self.kwargs['t_datetime_col']}  RAW_AGG_tn0,
                                         coalesce(tp1.{self.kwargs['t_datetime_col']},
                                                  tn0.{self.kwargs['t_datetime_col']}) RAW_AGG_tp1
                                   from t tn0
                              left join t tp1 on tn0.RAW_AGG_step=tp1.RAW_AGG_step-1
                                             and tn0.{self.kwargs['p_index_col']}
                                               = tp1.{self.kwargs['p_index_col']} 
                              left join t tm1 on tn0.RAW_AGG_step=tm1.RAW_AGG_step+1
                                             and tn0.{self.kwargs['p_index_col']}
                                               = tm1.{self.kwargs['p_index_col']};""")            
                tmp = pysqldf(f"""select tmp.RAW_AGG_tm1, tmp.RAW_AGG_tp1, t.*
                                    from t
                                    join tmp  on t.{self.kwargs['p_index_col']}
                                               = tmp.{self.kwargs['p_index_col']}
                                             and t.RAW_AGG_step = tmp.RAW_AGG_step;""")
                tmp[self.kwargs['t_datetime_col']] = tmp[self.kwargs['t_datetime_col']].apply(_cv)
                tmp['RAW_AGG_tm1'] = tmp['RAW_AGG_tm1'].apply(_cv)
                tmp['RAW_AGG_tp1'] = tmp['RAW_AGG_tp1'].apply(_cv)
                coarse_duration = lambda x: 300*( x.total_seconds()//300 -\
                                             ( (x.total_seconds()//300) % 5 ))
                                
                tmp['RAW_AGG_d_in'] = tmp[self.kwargs['t_datetime_col']] - tmp.RAW_AGG_tm1
                tmp['RAW_AGG_d_out'] = tmp.RAW_AGG_tp1 - tmp[self.kwargs['t_datetime_col']]
                              
                tmp[['RAW_AGG_d_in','RAW_AGG_d_out']] = \
                    tmp[['RAW_AGG_d_in','RAW_AGG_d_out']].applymap(coarse_duration)                                
                tmp['RAW_AGG_duration'] = tmp.RAW_AGG_d_in + tmp.RAW_AGG_d_out                
                tmp[self.kwargs['t_datetime_col']] = \
                    tmp[self.kwargs['t_datetime_col']].apply(lambda x: str(x)[:19])       
                
                t = tmp.copy() 
                
                privileged_t_cols = list([self.kwargs['p_index_col'],
                                          self.kwargs['t_datetime_col'],
                                          'ACTIVITY_DE',
                                          'RAW_AGG_step', 'RAW_AGG_step_countdown',                                           
                                          'RAW_AGG_tm1', 'RAW_AGG_tp1', 'RAW_AGG_duration', 
                                          ])
                rest_t_cols = sorted(list(filter(lambda x: x not in privileged_t_cols, t.columns)))                
                
                #TODO!!!!! hier noch viel mehr ausschlachten!! 
                # außerdem quantile für die count geschichten
                tmp = pysqldf(f"""select {self.kwargs['p_index_col']},
                                  min({self.kwargs['t_datetime_col']}) RAW_AGG_created_at,
                                  max({self.kwargs['t_datetime_col']}) RAW_AGG_last_at,
                                  avg(RAW_AGG_duration) RAW_AGG_avg_step_duration,
                                  max(RAW_AGG_duration) RAW_AGG_max_step_duration,
                                  sum(RAW_AGG_duration) RAW_AGG_sum_step_duration,                                  
                                  count(*) RAW_AGG_step_count
                          from t
                                  group by {self.kwargs['p_index_col']};""")
                tmp['AGG_duration'] = \
                   (tmp.RAW_AGG_last_at.apply(_cv) - tmp.RAW_AGG_created_at.apply(_cv)).apply(coarse_duration)
                
                p = pysqldf(f"""select tmp.RAW_AGG_created_at,
                                       tmp.RAW_AGG_last_at,
                                       tmp.RAW_AGG_step_count,
                                       tmp.RAW_AGG_avg_step_duration,
                                       tmp.RAW_AGG_max_step_duration,
                                       tmp.RAW_AGG_sum_step_duration,
                                       p.* 
                          from p
                          join tmp
                                    on p.{self.kwargs['p_index_col']}=tmp.{self.kwargs['p_index_col']}
                                  order by tmp.RAW_AGG_created_at DESC;""")
                
                privileged_p_cols = list([self.kwargs['p_index_col'],
                                          'RAW_AGG_created_at', 
                                          'RAW_AGG_last_at',                                                                                   
                                          'RAW_AGG_step_count',
                                          'RAW_AGG_avg_step_duration',
                                          'RAW_AGG_max_step_duration',
                                          'RAW_AGG_sum_step_duration'])
                rest_p_cols = sorted(list(filter(lambda x: x not in privileged_p_cols, p.columns)))
                
                p = p[privileged_p_cols + rest_p_cols]                
                p = p.sort_values('RAW_AGG_created_at',
                                  ascending=True).reset_index().drop(columns=['index'])
                
                t = pysqldf(f"""select p.RAW_AGG_step_count - t.RAW_AGG_step RAW_AGG_step_countdown, t.*
                                from p
                                join t on p.{self.kwargs['p_index_col']} = t.{self.kwargs['p_index_col']};""")
                
                t = t[privileged_t_cols + rest_t_cols]
                t = t.sort_values(self.kwargs['t_datetime_col'],
                                  ascending=True).reset_index().drop(columns=['index'])
                                
                self._save_p_t_to_local__(p,t)
                del p, t, tmp                      
                self.p, self.t = self._read_local_p_t_
                return self.p, self.t
            
            if self.mockup:
                p, t = self._mockup_()
            else:
                p, t = self._download_reference_p_t_()                
            p,t = one_time_aggregate_p_t(p,t)  
            print(f"""|-- {'MOCKUP:' if self.mockup else 'SQL:  '} saved pre-aggregates to local sqlite  \t --|""")
            print("\\==========================================================/")                          
            return p,t
            
    def get_tables_and_names(self)-> tuple[pd.DataFrame, pd.DataFrame, tuple, tuple]:
        p, t = self.raw_path_tables
        p_names = (self.kwargs['p_Table'],self.kwargs['p_index_col'])
        t_names = (self.kwargs['t_Table'],self.kwargs['t_datetime_col'])
        p_cols, t_cols = p.columns, t.columns
        return p,t,p_names,t_names,p_cols,t_cols
        
if __name__=='__main__':
    g = RawDataAgg(**{
            "MOCKUP":True,
            "SQL_URI":
            "DRIVER={SQL Server};SERVER=svhhd-bi001\\bi1;DATABASE=master",
            "p_Table":"ProcessManagement.dbo.[xFlowCases]",
            "p_index_col":"_CASE_KEY",
            "p_class_col":"EKGRP",
            "p_partner_c":"KUNDIN",
            "t_Table":"ProcessManagement.dbo.[xFlowActivities]",
            "t_datetime_col":"EVENTTIME",
            "t_activity_col":"ACTIVITY_DE",
            "t_value_summer_col":"somewarm_eur",
            "t_value_winter_col":"somecold_eur"
        })
    p,t = g.raw_path_tables
