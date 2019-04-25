# product-demand-rfr
 
'''
The dataset is from Kaggle: Forcasts for Product Demand.
https://www.kaggle.com/felixzhao/productdemandforecasting

The code cleans, scales, trains, fits, and predicts using the Random Forest Regressor
predict product demand.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, time, re, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
		
def datefill(df,clm):
	'''
	This function fills in empty dates with the preceeding date.
	'''
	for i in range(0,len(df)):
		if pd.isnull(df.loc[i,clm]):
			df.loc[i,clm] = df.loc[i-1,clm]
		else:
			pass

def code_split(code):
	'''
	Splits the product codes and return numerical component of the product code.
	'''
	z = code.split('_')
	return int(z[1])

if __name__ == '__main__':

	df = pd.read_csv('Historical Product Demand.csv')

	print('\nCleaning the data. Please wait...\n')
	
	df['Date'] = pd.to_datetime(df['Date'])

	# converts strings to integers
	df['Order_Demand'] = df['Order_Demand'].apply(bracket)

	# fills in empty dates with preceeding date
	datefill(df,'Date')

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
	scaled_feat = scaler.transform(X)
	df_scaled = pd.DataFrame(scaled_feat,columns=X.columns)
	XS = df_scaled
	y = df['Order_Demand']

	# spliting the training and test data
	X_train, X_test, y_train, y_test = train_test_split(XS, y, test_size=0.30)

	# no appreciable difference increasing estimators to 1000
	rfr = RandomForestRegressor(n_estimators=200,n_jobs=1)

	print('\nLearning. Hang on...')

	start = time.time()
	
	rfr.fit(X_train,y_train)
	
	end = (time.time() - start) / 60
	print(f'Done! Training duration: {round(end,2)} min')

	# pickle to alleviate need to train in future
	model_file = 'model.pkl'
	pickle.dump(rfr, open(model_file,"wb"))
	
	print(f'\nThe model file {model_file} has been dumped into the working folder.')

	# load model to test for deployment protocol
	with open(model_file, 'rb') as data:
		rfr = pickle.load(data)

	print(f'\nThe model file {model_file} has been successfully reloaded.')

	# Predict
	print('\nPredicting. Wait for it...')
	rfr_pred = rfr.predict(X_test)

	# Performance metrics
	errors = abs(rfr_pred - y_test)
	err_avg = round(np.mean(errors), 1)
	stdev = round(np.std(errors), 1)
	err_med = round(np.median(errors), 1)
	r2 = round(r2_score(y_test,rfr_pred),3)
	t, t_p = kendalltau(y_test, rfr_pred)
	r, r_p = spearmanr(y_test, rfr_pred)
	print('\nAverage absolute error:', err_avg, 'units')
	print('Standard deviation:    ', stdev, 'units')
	print('Average median error:  ', err_med, 'units')
	print('r2 score:              ', r2)
	print(f'\nKendall tau:  {round(t*100,1)}%, p-value: {t_p}')
	print(f'Spearman rho: {round(r*100,1)}%, p-value: {r_p}')