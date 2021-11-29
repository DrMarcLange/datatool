class ModelConfig():
    """
    auf basis der p,t - tabellen einer rawdata quelle
    mit p wie "ProcessManagement.dbo.[xFlowCases]"
    und t wie "ProcessManagement.dbo.[xFlowActivities]"
    
    modelldaten extraktion
    """
    def __init__(self,**kwargs):
        self.debug = ('DEBUG' in kwargs.keys())
        
        if self.debug:
            print()
            print('                          ______                           ')
            print('/=========================______===========================\\')
            print('|/|=========--------------\\\\__//---------\\===============|\\|')
            print('|_|=====-------------------\\\\//----------------\\=========|_|')	
            print('|===--------DEBUG key exists - model more verbose-----\\====|')	
        
        def load_raw_data(raw_data_config:dict):
            from datetime import datetime            
            read_date = lambda x: datetime.fromisoformat(str(x)[:19])
            import Data_Configuration
            raw_data_provider = Data_Configuration.RawDataAgg(**raw_data_config)
            self.ix = raw_data_config['p_index_col']
            p, t = raw_data_provider.raw_path_tables
            self.p, self.t = p.copy(), t.copy()
            coarse_duration = lambda x: (x.total_seconds()//120 - ( (x.total_seconds()//120) % 3 ))
            self.p['MDL_duration_total_minutes'] = (self.p.AGG_last_at.apply(read_date) - self.p.AGG_created_at.apply(read_date)).apply(coarse_duration)
            from numpy import log1p
            intlog1p = lambda x: int(log1p(x))
            self.p['MDL_log_duration_total_minutes'] = self.p['MDL_duration_total_minutes'].apply(intlog1p)
            if 'something_random' in self.t.columns:
                self.t['nom_target'] = self.t.something_random.apply(lambda x: x[:1])
            self.raw_data_config = raw_data_config
            self.p_cols = sorted(set(self.p.columns))
            self.t_cols = sorted(set(self.t.columns))
            self.known_cols = sorted(set(self.p_cols+self.t_cols))
            del p,t,raw_data_provider, Data_Configuration
            if self.debug:
                global pp
                pp = self.p
                from pandasql import sqldf    
                pysqldf = lambda q: sqldf(q, globals())                   
                r = pysqldf("""select max(AGG_created_at) a, min(AGG_created_at) b from pp;""")
                a,b=r.iloc[0,:]
                c,d=len(self.p),len(self.t)
                del pp,pysqldf,r,sqldf
                print(f"""----\t TOTAL Daterange:                 \t      ----\n=\t {a}\t{b}\t =""")
                print(f"""=\t row_count(p):{c}\trow_count(t):{d}\t =""")
                
        def set_process_value_constraints(constraints:dict):
            def all_unknown_process_constraints()->str:
                tmp = [str(x) for x in constraints.keys() if x not in self.KNOWN_PROCESS_CONSTRAINTS]
                return ' '.join(tmp)            
            assert all(
                map(lambda x:
                    x in self.KNOWN_PROCESS_CONSTRAINTS,
                    constraints.keys()
                   )), "some unimplemented process constraint in keyword arguments: "+all_unknown_process_constraints()
            from pandas import to_datetime as _cv
            for key in self.KNOWN_PROCESS_CONSTRAINTS:
                try:
                    constraints[key]='"'+str(int(str(constraints[key])))+'"'
                    if self.debug:
                        #print('int',key)
                        pass
                except:
                    try:
                        constraints[key]='"'+str(_cv(str(constraints[key])))+'"'
                        if self.debug:
                            #print('date',key)
                            pass
                    except:
                        if self.debug:
                            #print('except',key)
                            pass
                        constraints[key] = ""
            self.process_where_list = []
            if constraints['created_after']!="":
                self.process_where_list += [f"""(p.AGG_created_at >= {constraints['created_after']}) """]
            if constraints['created_before']!="":
                self.process_where_list += [f"""(p.AGG_created_at <= {constraints['created_before']}) """]
            if constraints['stopped_after']!="":
                self.process_where_list += [f"""(p.AGG_last_at >= {constraints['stopped_after']}) """]
            if constraints['stopped_before']!="":
                self.process_where_list += [f"""(p.AGG_last_at <= {constraints['stopped_before']}) """]
            if constraints['at_least_min']!="":
                self.process_where_list += [f"""(p.MDL_duration_total_minutes >= {constraints['at_least_min']}) """]
            if constraints['at_most__min']!="":
                self.process_where_list += [f"""(p.MDL_duration_total_minutes <= {constraints['at_most__min']}) """]
            if constraints['one_edge_at_least_min']!="":
                self.process_where_list += [f"""(p.AGG_max_d >= {constraints['one_edge_at_least_min']})"""]
            if constraints['all_edges_at_most_min']!="":
                self.process_where_list += [f"""(p.AGG_max_d <= {constraints['all_edges_at_most_min']})"""]
            if constraints['at_least_step_count']!="":
                self.process_where_list += [f"""(p.AGG_step_count >= {constraints['at_least_step_count']}) """]
            if constraints['at_most__step_count']!="":
                self.process_where_list += [f"""(p.AGG_step_count <= {constraints['at_most__step_count']}) """]
            if len(self.process_where_list)>0:
                self.process_rule_str = ' AND '.join(self.process_where_list)
                P_CONDITION=self.process_rule_str
                from pandasql import sqldf
                global p,t
                p = self.p; t = self.t
                pysqldf = lambda q: sqldf(q, globals())
                query_p = f"""select * from p where {P_CONDITION};"""
                query_t = f"""select t.* from t join (select {self.ix} from p where {P_CONDITION}) s 
                                         on t.{self.ix}=s.{self.ix};"""
                self.p = pysqldf(query_p).copy()
                self.t = pysqldf(query_t).copy()
                p = self.p
                t = self.t
                assert len(p) > 0, 'global process-conditions left empty process set p'
                assert len(t) > 0, 'global process-conditions left empty process set t'
                if self.debug:
                    global pp
                    pp,tt = self.p,self.t
                    pysqldf = lambda q: sqldf(q, globals())
                    r = pysqldf("""select max(AGG_created_at) a, min(AGG_created_at) b from pp;""")
                    a,b=r.iloc[0,:]
                    c,d=len(self.p),len(self.t)
                    del pp,pysqldf,r,sqldf
                    print('-NEW- Daterange:\n\t', r.iloc[0,1],'\t',r.iloc[0,0])
                    print('\t row_count(p):', len(p),'\trow_count(t):', len(t))
                del p, t, sqldf, pysqldf
            else:
                self.process_rule_str = ''
            if self.debug:
                #print("current process rule_str: ", self.process_rule_str)            
                pass
                
        def set_column_value_constraints(constraints:dict):
            def check_equality(colname:str,vallist:list)->str:
                if len(vallist)==0:
                    return ''
                if len(vallist)==1:
                    return colname+'="'+vallist[0]+'"'
                return colname+' in '+str(tuple(vallist)) 
            def all_unknown_columns()->str:
                tmp = [str(x) for x in constraints.keys() if ((x not in self.known_cols) and (x[1:] not in self.known_cols))]
                return ' '.join(tmp)            
            assert all(map(lambda x: ( x in self.known_cols ) or ( x[1:] in self.known_cols ), constraints.keys())), \
                    "at least one constraint column unknown: "+all_unknown_columns()
            self.p_col_where_list = list()
            self.p_col_where_not_list = list()
            self.t_col_where_list = list()
            self.t_col_where_not_list = list()

            for col in constraints.keys():
               val = tuple(constraints[col])
               if (( col in self.known_cols )
                or ( (col[0]=='+') and ( col[1:] in self.known_cols ) )):
                    if not col in self.known_cols:
                        col = col[1:]
                        assert col in self.known_cols, "marc logic error?"
                    if col in self.p_cols:                        
                        self.p_col_where_list += [" ( "+check_equality("p."+col,val)+" ) "]                        
                    if col in self.t_cols:
                        self.t_col_where_list += [" ( "+check_equality("t."+col,val)+" ) "]
               elif col[0]=='-':
                    col = col[1:]
                    if col in self.p_cols:                        
                        self.p_col_where_not_list += [" ( NOT ( "+check_equality("p."+col,val)+" )) "]
                    if col in self.t_cols:
                        self.t_col_where_not_list += [" ( NOT ( "+check_equality("t."+col,val)+" )) "]
            self.p_col_rule_str = ' AND '.join(self.p_col_where_list+self.p_col_where_not_list)                                
            self.t_col_rule_str = ' AND '.join(self.t_col_where_list+self.t_col_where_not_list)
            self.col_rule_str = self.p_col_rule_str +" AND " +self.t_col_rule_str
            if self.debug:
                pass
                #print("current column rule_str: ", self.col_rule_str)
            query_p  = f"""select * from p where {self.p_col_rule_str};"""
            query_t1 = f"""select t.* from t join (select {self.ix} from p where {self.p_col_rule_str}) s 
                                     on t.{self.ix}=s.{self.ix};"""
            query_t2 = f"""select * from t where {self.t_col_rule_str};"""                                                                 
            from pandasql import sqldf    
            global p, t
            p = self.p; t = self.t            
            pysqldf = lambda q: sqldf(q, globals())       
            self.p = pysqldf(query_p).copy()            
            t = pysqldf(query_t1)
            t = pysqldf(query_t2)
            self.t = t.copy()
            if self.debug:
                p = self.p
                r = pysqldf("""select max(AGG_created_at) a, min(AGG_created_at) b from p;""")
                print('-NEW- Column__Constraints Applied:\n\t', r.iloc[0,1],'   ',r.iloc[0,0])
                print('\t row_count(p):', len(p), '\trow_count(t):', len(t))
            assert len(p) > 0, 'column conditions left empty process set p'
            assert len(t) > 0, 'column conditions left empty process set t'                             
            del p, t, sqldf
        
        assert 'raw_data_config' in kwargs.keys(), 'no data source configured'        
        load_raw_data(kwargs['raw_data_config'])        
        if ('process_constraints' in kwargs.keys()) and (len(kwargs['process_constraints'])>0):
            self.KNOWN_PROCESS_CONSTRAINTS = \
                ['created_after','created_before',
                 'stopped_after','stopped_before',
                 'at_least_min' ,'at_most__min'  ,
                 'one_edge_at_least_min','all_edges_at_most_min',
                 'at_least_step_count','at_most__step_count']
            set_process_value_constraints(kwargs['process_constraints'])        
        else:
            print('   no process constraints applied  ')    
        if ('column_constraints' in kwargs.keys()) and (len(kwargs['column_constraints'])>0):
            set_column_value_constraints(kwargs['column_constraints'])
        else:    
            print('   no column  constraints applied  ')    
            
        if 'target_columns' in kwargs.keys():
            self.models_for_targets(kwargs['target_columns'])
    
    def _get_training_data(self,target_cfg:dict):
        def _parse_target(target_cfg:dict):
            if 'theta' in target_cfg.keys():
                try:
                    theta = int(target_cfg['theta']) if int(target_cfg['theta']) > 0 else 0                                
                    if theta == 0:
                        raise ValueError    
                except:
                    #print(f"""value of theta: {target_cfg['theta']} not a valid threshold""")
                    del target_cfg['theta']
            if 'logarithmic' in target_cfg.keys():
                try:
                    log = int(target_cfg['logarithmic'])
                    if log != 1:
                        del target_cfg['logarithmic']
                except:
                    #print(f"""value of logarithmic: {target_cfg['logarithmic']} not a valid bool""")
                    del target_cfg['logarithmic']
            if 'target' in target_cfg.keys():
                try:
                    del_keys = list()
                    for act in target_cfg['target']:
                        if not act in self.known_cols:
                            #print(f"""unknown target column, removing {act}""")
                            del_keys += del_keys + [act,]  
                    for act in set(del_keys):
                        del target_cfg['target'][act]       
                    if len(target_cfg['target'].keys()) == 0:
                        del target_cfg['target']     
                except:
                    print('Could not parse "target"-array') 
                    raise ValueError                    
            assert 'name' in target_cfg.keys(), f"""no analysis chosen: {target_cfg}"""
            assert target_cfg['name'] in ['COUNT','DURATION'] , f"""no valid analysis chosen: {target_cfg['name']}"""
            return target_cfg
        
        def get_target(analysis_name:str, t_cols:list, p_cols:list, logarithmic:bool):
            if logarithmic:
                from numpy import log1p
                intlog1p = lambda x: int(log1p(x))
            if len(t_cols + p_cols) >= 1:
                global p,t
                p,t = self.p.copy(), self.t.copy()
                from pandasql import sqldf    
                pysqldf = lambda q: sqldf(q, globals())                   
                for target_cnt, col in enumerate(t_cols):
                    cnt_col = f"""TARGET_cnt_{col}"""
                    drn_col = f"""TARGET_drn_{col}"""
                    vals = tuple(sorted(target_cfg['target'][col]))
                    if len(vals)<=1:
                        WHERE_CONDITION = f""" where t.{col} = {vals[0]} """
                    else:
                        WHERE_CONDITION = f""" where t.{col} in {vals} """
                    th = pysqldf(f"""select p.{self.ix} {self.ix}, 
                                            coalesce(count(*),0) {cnt_col}, 
                                            coalesce(sum(t.Agg_d),0) {drn_col}
                                     from t left join p on t.{self.ix}=p.{self.ix} 
                                     {WHERE_CONDITION}
                                     group by p.{self.ix}""")
                    p[cnt_col] = th[cnt_col]
                    p[drn_col] = th[drn_col]                
                    
                if analysis_name == 'COUNT':
                    p_drop_cols = list(filter(lambda x: 'TARGET_drn' == x[:10], p.columns))
                    p = p.drop(columns=p_drop_cols).rename(columns={cnt_col:'TARGET_{target_cnt}'})
                if analysis_name == 'DURATION':
                    p_drop_cols = list(filter(lambda x: 'TARGET_cnt' == x[:10], p.columns))
                    p = p.drop(columns=p_drop_cols).rename(columns={drn_col:'TARGET_{target_cnt}'})
                    
                for target_cnt, col in enumerate(p_cols):
                    cnt_col = f"""TARGET_cnt_{col}"""
                    vals = tuple(sorted(target_cfg['target'][col]))
                    if len(vals)<=1:
                        WHERE_CONDITION = f""" where p.{col} = {vals[0]} """
                    else:
                        WHERE_CONDITION = f""" where p.{col} in {vals} """
                    th = pysqldf(f"""select ps.{self.ix} {self.ix}, 
                                            coalesce(count(*),0) {cnt_col}
                                     from p ps left join p on ps.{self.ix}=p.{self.ix} 
                                     {WHERE_CONDITION}
                                     group by p.{self.ix}""")
                    p[cnt_col] = th[cnt_col]
                ps = p.copy()
                del p,t,sqldf,pysqldf    
            else: #t_cols + p_cols empty
                p['TARGET_1'] = p['MDL_duration_total_minutes'] if analysis_name =='DURATION' else p['AGG_step_count']
                ps = p.copy()
            if logarithmic:
                result_target_cols = list(filter(lambda x: 'TARGET_'==x[:7], ps.columns))
                ps[result_target_cols] = ps[result_target_cols].fillna(0).applymap(intlog1p)    
            return ps    
        
        def sum_target_values(p):
            target_cols = list(filter(lambda x: x[:7] == 'TARGET_', p.columns))
            p['TARGET']=0
            for c in target_cols:
                p['TARGET'] = p['TARGET']+p[c] #sum all 
            p = p.drop(columns=target_cols)
            return p    
            
        def apply_threshold(p,theta):
            target_cols = list(filter(lambda x: x[:7] == 'TARGET_', p.columns))
            for c in target_cols:
                p[c] = p[c].apply(lambda x: .9 if x >= theta else .1 ).apply(int)
            p['TARGET'] = 1
            for c in target_cols:
                p['TARGET'] = p['TARGET']*p[c] #all AND-connected
            p = p.drop(columns=target_cols)
            return p    
            
        self.last_target_cfg = _parse_target(target_cfg)    
        print('using target config: ', self.last_target_cfg)        
        target_cols = target_cfg.keys()
        t_cols = sorted(list(filter(lambda x: x in self.t.columns, target_cols)))
        p_cols = sorted(list(filter(lambda x: x in self.p.columns, target_cols)))
        p_targeted = get_target(target_cfg['name'],t_cols,p_cols,'logarithmic' in self.last_target_cfg.keys())
        if 'theta' in target_cfg.keys():
            p = apply_threshold(p_targeted,int(target_cfg['theta']))
        else:
            p = sum_target_values(p_targeted)    
        p['TARGET'] = p.TARGET.fillna(0).apply(lambda x: int(4*x)/4)    
        target = p['TARGET'].copy()
        p = p.drop(columns='TARGET')    
        return target,p
    
    import pandas as pd
    def _get_factors(self, target_series: pd.Series, training_p: pd.DataFrame):
        def eliminate(df,results):
            pass
        L = len(training_p.columns) + 1
        l = L - 1
        import statsmodels.api as sm
        task = 'clf' if (target_series.max() - target_series.min()) <= 1 else 'reg'
        if task == 'clf':
            def fit(df,t):            
                try:
                    results = sm.Logit(t,df).fit() 
                except:
                    results = sm.OLS(t,df).fit()    
        else:
            def fit(df,t):
                results = sm.OLS(t,df).fit()            
        return fit(training_p,target_series)
        
    def model_for_target(self,target:dict):
        target_series, training_p = self._get_training_data(target)        
        result_model = '' #self._get_factors(target_series, training_p)
        try:
            self.models += [result_model]
        except AttributeError:
            self.models = [result_model]
    
    def models_for_targets(self,targets:list):
        for target in targets:
            self.model_for_target(target)    
