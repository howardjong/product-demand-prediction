# product-demand-adam-auto.py

'''
Uses TensorFlow to apply the Adam optimizer instead of classical stochastic gradient descent
method for the regression analysis.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
import re, time, pickle
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
		
def code_split(code):
	'''
	Splits the product codes and return numerical component of the product code.
	'''
	z = code.split('_')
	return int(z[1])

def test(lr,epochs):
	'''
	Takes a single learning rate and a number of epochs to return a mean squared error.
	'''
	w = tf.Variable([[0],[0],[0],[0],[0],[0]], trainable=True, dtype=tf.float64)
	x = tf.convert_to_tensor(X_train_val, dtype=tf.float64)
	y = tf.convert_to_tensor(y_train_val, dtype=tf.float64)
	y_pred = tf.matmul(x, w)
	mse = tf.losses.mean_squared_error(y, y_pred)
	adam = tf.train.AdamOptimizer(learning_rate=lr)
	a = adam.minimize(mse, var_list=w)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for n in range(epochs):
			sess.run(a)
		check = sess.run(mse)
		w = sess.run(w)

	return check

def scan(lr_range, epochs):
	'''
	Applies a range of learning rates and epochs to the test function to return a list of
	mean squared errors for each corresponding learning rate.
	'''
	results = []
	for i in lr_range:
		x = test(i,epochs)
		results.append(x)
	return results

def best(keys, values):
	'''
	Creates a dataframe of the learning rates and preliminary mean squared error values then 
	returns the learning rate with the lowest MSE.
	'''
	group = dict(zip(keys, values))
	to_df = {'LR':keys, 'MSE':values}
	df = pd.DataFrame(to_df)
	lowest = df[df['MSE'] == df['MSE'].min()]['LR']
	return float(lowest)

def trunc_plot(array, length):
	'''
	Processes output values from Adam optimization for plotting.
	'''
	for i in range(1,length):
		yield array[i]

def listing(weights):
	'''
	Generates final weight values for printing.
	'''
	for i in range(0,len(weights)):
		yield weights[i][0]

def output(columns, weights):
	for i in range(0,len(weights)):
		print(f'{columns[i]:<13}: {weights[i]}')


if __name__ == '__main__':

	df = pd.read_csv('Historical Product Demand.csv')

	print('\nCleaning the data. Please wait...\n')
	
	df['Date'] = pd.to_datetime(df['Date'])

	# converts strings to integers
	df['Order_Demand'] = df['Order_Demand'].apply(bracket)

	# fills in empty dates with preceeding date
	df['Date'] = df['Date'].fillna(method='bfill')

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

	print('\nEvaluating learning rates...\n')

	# testing which learning rate is best
	lr_range = [0.0005,0.001,0.002,0.004,0.02,0.1,0.5,1.0,2.0,4.0]
	CYCLES = 50
	values = scan(lr_range,CYCLES)
	opt_lr = best(lr_range,values)

	print(f'\nFrom the range tested {lr_range} a learning rate of {opt_lr} was selected.')

	# train the weights of each variable; start weights at 0 which accumulates with the backpass
	# number of variables must match the number of columns; each variable represents a feature
	w = tf.Variable([[0],[0],[0],[0],[0],[0]], trainable=True, dtype=tf.float64)
	x = tf.convert_to_tensor(X_train_val, dtype=tf.float64)
	y = tf.convert_to_tensor(y_train_val, dtype=tf.float64)

	# matmul = matrix multiplication to get predictions
	y_pred = tf.matmul(x, w)

	mse = tf.losses.mean_squared_error(y, y_pred)

	# the optimizer
	adam = tf.train.AdamOptimizer(learning_rate=opt_lr)

	# this runs one step of gradient descent
	a = adam.minimize(mse, var_list=w)

	l, p = confusion_matrix.remove_squeezable_dimensions(y, y_pred)
	s = math_ops.square(p - l)
	mean_t = math_ops.reduce_mean(s)

	print('\nLearning. Hang on...\n')
	start = time.time()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
	    
		progress = []
		go = True
	        
		while go:
			sess.run(a) # run for the number of training steps
			step = sess.run(mse)      
			progress.append(step)
			if len(progress) > 12:
				cutoff = abs(step - np.mean(progress[-10:-1]))
				if cutoff < 1:
					go = False
				else:
					pass
			else:
				pass

		w = sess.run(w)
		w_ = list(listing(w))
		print('\nOptimized feature weights:')
		output(list(columns), w_)

		e_val = sess.run(mean_t)
		print(f'\nFinal MSE: {e_val}')

		progress = tf.stack(progress)
		epochs = len(progress.eval())
		progress_lst = progress.eval()
		progress_lst = pd.DataFrame(progress_lst, columns=['mse'])
	    
	end = (time.time() - start)/60
	print(f'\nTraining duration: {round(end,2)} min')

	model_file = 'w.pkl'
	pickle.dump(w, open(model_file,"wb"))

	print(f'\nThe model file {model_file} has been dumped into the working folder.')

	# load model to test for deployment protocol
	with open(model_file, 'rb') as data:
	    w = pickle.load(data)

	print(f'\nThe model file {model_file} has been successfully reloaded.')

	y = list(trunc_plot(progress_lst['mse'],epochs))

	fig, ax = plt.subplots(figsize=(12,8))
	ax.plot(range(1,epochs), y, marker='.', markersize=1, color='c', label='learning rate: '+str(opt_lr))
	ax.set_title('Adam Loss Minimzation')
	ax.set_xlabel('Epochs', fontsize=12)
	ax.set_ylabel('Mean Squared Error', fontsize=12)
	ax.legend(fontsize=10)
	plt.show()

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
	r2 = round(r2_score(y_test_val,adam_pred),3)
	t, t_p = kendalltau(y_test_val, adam_pred)
	r, r_p = spearmanr(y_test_val, adam_pred)
	print('\nAverage absolute error:', err_avg, ' units')
	print(f'Standard deviation:    ', stdev, 'units')
	print('Average median error:  ', err_med, ' units')
	print('r2 score:              ', r2)
	print(f'\nKendall tau:  {round(t*100,1)}%, p-value: {t_p}')
	print(f'Spearman rho: {round(r*100,1)}%, p-value: {r_p}')
