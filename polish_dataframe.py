import pandas as pd
import numpy as np
def remove_linear_dependence(df:pd.DataFrame)->pd.DataFrame:
	"""
	use "df = remove_linear_dependence(df)"
	to trim df down to a number of mostly uncorrelated columns
	"""
	rk = lambda x: np.linalg.matrix_rank(x, tol=1e-1)
	cols = list(df.columns)[::-1]
	rk_df = df
	for c in cols:
		if df.shape[1] == rk(df):
			print('rank-search done')
			break
		if rk(rk_df.drop(columns=c))==rk(rk_df):
			print('column ', c.encode("utf-8"), 'linearly dependent, drop it!')
			df = df.drop(columns=c) 
	print('found rank ', rk(df), ' submatrix with shape ', df.shape)
	return df 

def remove_high_corr(df:pd.DataFrame, cutoff:float)->pd.DataFrame:
	"""
	use "df=remove_high_corr(df)"
	to trim df down to only features below <cutoff> pairwise correlation
	"""	
	cr = df.corr()
	cr = pd.melt(cr.reset_index(), id_vars='index')
	cr = cr[cr.value!=1]
	cr = cr[cr['index']<cr.variable]
	cr = cr[cr.value.apply(abs) >= cutoff]
	drop_cols = cr.variable.unique()
	print('dropping ', drop_cols)
	df = df.drop(columns=drop_cols)
	print('DROPPING DONE')
	return df 

def make_uncorrelated_data(df:pd.DataFrame)->pd.DataFrame:	
	"""
	use like "df = make_uncorrelated_data(df)"	
	"""
	df = remove_high_corr(df, cutoff=.4)
	df = remove_linear_dependence(df)
	return df	
