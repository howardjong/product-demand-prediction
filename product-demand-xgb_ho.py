# product-demand-xgbr_ho
 
'''
The dataset is from Kaggle: Forcasts for Product Demand.
https://www.kaggle.com/felixzhao/productdemandforecasting

The code cleans, scales, trains, fits, and predicts using the XGBoost Regressor
with hyperparameters tuned with Bayesian Optimization
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, time, re, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import kendalltau, spearmanr

def bracket(row):
	'''
	This function converts [negative] string values in bracket form to standard integers.
	'''
	if re.search('\(', row):
		return int('-' + row[1:-1])
	else:
		return int(row)
		
def code_split(code):
	'''
	Splits the product codes and return numerical component of the product code.
	'''
	z = code.split('_')
	return int(z[1])

def plot_features(booster, figsize):    
	fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=120)
	return plot_importance(booster=booster, ax=ax)

def xgb_evaluate(learning_rate, max_depth, subsample, eta, gamma, colsample_bytree):
	params = {
	'silent': True,
	'eval_metric': 'rmse',
	'learning_rate': learning_rate,
	'max_depth': int(max_depth),
	'subsample': subsample,
	'eta': eta,
	'gamma': gamma,
	'colsample_bytree': colsample_bytree
	}

	# Used 1000 boosting rounds in the full model; computationally expensive
	cv_result = xgb.cv(params, dtrain, num_boost_round=250, nfold=5)    

	# Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
	return -1.0 * cv_result['test-rmse-mean'].iloc[-1]


if __name__ == '__main__':

	df = pd.read_csv('Historical Product Demand.csv')

	print('\nCleaning the data. Please wait...\n')
	
	df['Date'] = pd.to_datetime(df['Date'])

	# converts strings to integers
	df['Order_Demand'] = df['Order_Demand'].apply(bracket)

	# fills in empty dates with preceeding date
	df['Date'] = df['Date'].fillna(method='bfill')

	# converts product codes from strings to integers
	# scaling product codes gave much better results compared with one hot encoding
	df['Product_Code'] = df['Product_Code'].apply(code_split)

	df['Year'] = df['Date'].apply(lambda date: date.year)
	df['Month'] = df['Date'].apply(lambda date: date.month)
	df['Day'] = df['Date'].apply(lambda date: date.day)
	df['Dayofweek'] = df['Date'].apply(lambda date: date.dayofweek)

	# scaling the data
	scaler = StandardScaler()
	X = df.drop(['Warehouse','Product_Category','Date','Order_Demand'],axis=1)
	scaler.fit(X)

	scaler_file = 'scaler.pkl'
	pickle.dump(scaler, open(scaler_file,'wb'))
	
	print(f'\nThe model file {scaler_file} has been dumped into the working folder.')

	# load model to test for deployment protocol
	with open(scaler_file, 'rb') as s:
		scaler = pickle.load(s)

	print(f'\nThe model file {scaler_file} has been successfully reloaded.')

	scaled_feat = scaler.transform(X)
	df_scaled = pd.DataFrame(scaled_feat,columns=X.columns)
	XS = df_scaled
	y = df['Order_Demand']

	X_train, X_test, y_train, y_test = train_test_split(XS, y, test_size=0.20)

	dtrain = xgb.DMatrix(X_train, label=y_train)
	dtest = xgb.DMatrix(X_test)

	xgb_bo = BayesianOptimization(xgb_evaluate, {
		'learning_rate': (0.01, 0.5),
		'max_depth': (3, 7),
		'subsample': (0, 1),
		'eta': (0.1, 0.5),
		'gamma': (0, 1),
		'colsample_bytree': (0.3, 0.9)
		})

	print('\nOptimizing parameters...\n')
	start = time.time()

	# Use the expected improvement acquisition function to handle negative numbers
	# Optimally needs quite a few more initiation points and number of iterations
	xgb_bo.maximize(init_points=10, n_iter=15, acq='ei')

	params = xgb_bo.max
	params['params']['max_depth'] = int(round(params['params']['max_depth'],0))
	params = params['params']

	print(f'Optimized parameters: {params}\nLearning...')

	# Train model with the optimized parameters; full training used 1000 boost rounds
	xgbr_ho = xgb.train(params, dtrain, num_boost_round=250)

	end = (time.time() - start) / 60
	print(f'\nDone! Training duration: {round(end,2)} min')

	# pickle to alleviate need to train in future
	model_file = 'xgbr_ho_model.pkl'
	pickle.dump(xgbr_ho, open(model_file,'wb'))
	
	print(f'\nThe model file {model_file} has been dumped into the working folder.')

	# load model to test for deployment protocol
	with open(model_file, 'rb') as data:
		xgbr_ho = pickle.load(data)

	print(f'\nThe model file {model_file} has been successfully reloaded.')

	# Predict
	print('\nPredicting. Wait for it...')

	y_pred = xgbr_ho.predict(dtest)

	# Performance metrics
	errors = abs(y_pred - y_test)
	err_avg = round(np.mean(errors), 1)
	stdev = round(np.std(errors), 1)
	err_med = round(np.median(errors), 1)
	r2 = round(r2_score(y_test,y_pred),3)
	t, t_p = kendalltau(y_test, y_pred)
	r, r_p = spearmanr(y_test, y_pred)
	print('\nAverage absolute error:', err_avg, 'units')
	print('Standard deviation:    ', stdev, 'units')
	print('Average median error:  ', err_med, 'units')
	print('r2 score:              ', r2)
	print(f'\nKendall tau:  {round(t*100,1)}%, p-value: {t_p}')
	print(f'Spearman rho: {round(r*100,1)}%, p-value: {r_p}')

	xgb.plot_importance(xgbr_ho)
	plt.show()
	