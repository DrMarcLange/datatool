with open('glue.json','r') as fi:
    analysis_rules = json.load(f)

from Model_Configuration import ModelConfig
model = ModelConfig(**analysis_rules)
p,t = model.p,model.t

print(model.models)

#=================================================================================#
# kannst dir mit "python -i glue_process.py" ne model.models und model.p, model.t #
# vorladen lassen und von hand weitere targets ansehen (e.g. klassifikation ->    #
# regression oder so) mit model.model_for_target, wie zB                          #
#                                                                                 # 
# model.model_for_target({ 'name' : 'STEPS_LIKE__DURATION',                       #
#         'activities': ['act1','act2','act3'],                                   #
#         'log': True                          })                                 #
# mdl = model.models[-1]                                                          #
#                                                                                 #
# if 'summary' in dir(mdl):                                                       #
#     print(mdl.summary())                                                        #
# else:                                                                           #
#     print('BLANK')                                                              #
#=================================================================================#

#TRY sth like: cls_qtls = p[self.kwargs['p_class_col']].value_counts().\
#       quantile([0.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,0.99]).rename('quantiles')
#AND: pysqldf(f"""select {self.kwargs['p_class_col']} h, count(*) cnt from df
#       group by h order by cnt desc""").cnt.plot(); plt.show()
#lauter unabhängige seeds für reproducibility
