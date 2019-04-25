# product-demand-adam.py
# Uses TensorFlow to apply the Adam optimizer instead of classical stochastic gradient descent method
# for the regression analysis of the product demand dataset to predict product demand

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
import re, time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
	# this approach returned much improved values compared with one hot encoding the product codes
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
	df_scaled['Constant'] = 1
	XS = df_scaled
	y = df['Order_Demand']

	# spliting the training and test data
	X_train, X_test, y_train, y_test = train_test_split(XS, y, test_size=0.30)

	columns = X_train.columns
	X_train_val = X_train[columns].values
	y_train_val = y_train.values
	y_train_val = y_train_val.reshape(-1, 1)	# needs to convert series into np array

	# train the weights of each variable; start weights at 0 which accumulates with the backpass
	# number of variables must match the number of columns; each variable represents a feature
	w = tf.Variable([[0],[0],[0],[0],[0],[0]], trainable=True, dtype=tf.float64)
	x = tf.convert_to_tensor(X_train_val, dtype=tf.float64)
	y = tf.convert_to_tensor(y_train_val, dtype=tf.float64)

	# matmul = matrix multiplication to get predictions
	y_pred = tf.matmul(x, w)

	mse = tf.losses.mean_squared_error(y, y_pred)

	# the optimizer; typical learning_rate = 0.001 (best typically), 0.0005, 0.002 --> 0.3
	# but this example benefits from higher than normal learning rate
	adam = tf.train.AdamOptimizer(learning_rate=0.9)

	# this runs one step of gradient descent
	a = adam.minimize(mse, var_list=w)

	l, p = confusion_matrix.remove_squeezable_dimensions(y, y_pred)
	s = math_ops.square(p - l)
	mean_t = math_ops.reduce_mean(s)

	print('\nLearning. Hang on...\n')
	start = time.time()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(50000):
			sess.run(a) # run for the number of training steps
		w = sess.run(w)

		print(f'Feature weights: {w}') # this will output our current weights after training
		e_val = sess.run(mean_t) # compute our MSE
		print(f'MSE = {e_val}')

	end = (time.time() - start) / 60
	print(f'\nDone! Training duration: {round(end,2)} min')

	X_test_val = X_test[columns].values
	X_test_ten = tf.convert_to_tensor(X_test_val, dtype=tf.float64)
	y_test_val = y_test.values
	y_test_val = y_test_val.reshape(-1, 1)

	print('\nPredicting. Wait for it...')
	adam_pred = tf.Session().run(tf.matmul(X_test_ten, w))

	# Performance metrics
	errors = abs(adam_pred - y_test_val)
	err_avg = round(np.mean(errors), 1)
	stdev = round(np.std(errors), 1)
	err_med = round(np.median(errors), 1)
	r2 = round(r2_score(y_test,adam_pred),3)
	t, t_p = kendalltau(y_test, adam_pred)
	r, r_p = spearmanr(y_test, adam_pred)
	print('\nAverage absolute error:', err_avg, 'units')
	print('Standard deviation:    ', stdev, 'units')
	print('Average median error:  ', err_med, 'units')
	print('r2 score:              ', r2)
	print(f'\nKendall tau:  {round(t*100,1)}%, p-value: {t_p}')
	print(f'Spearman rho: {round(r*100,1)}%, p-value: {r_p}')
