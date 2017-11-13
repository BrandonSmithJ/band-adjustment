import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from DNN import DNN 

import tensorflow as tf 
import pickle as pkl 
import numpy as np 
import tqdm

n_bands = {
	'MSI' : 4,
	'OLI' : 4,
	'VI'  : 5,
	'OLCI': 9,
	'AER' : 6,
	'ETM' : 3,
	'TM'  : 3,
	'MOD' : 10,
}

def get_data(path):
	data = np.array([l.strip().split(',') for l in 
					open(path).readlines()])
	if data.shape[0] < data.shape[1]: data = data.T
	try: data = data.astype(np.float32)
	except ValueError: 
		data[0,0] = data[0,0][3:] # Excel puts in hidden characters...
		data = data.astype(np.float32)
	if np.sum(data[0]) > 10:
		print('First row of data assumed to be wavelengths (nm)')
		data = data[1:]
	return data


class Model(object):
	''' Wrapper for a saved tensorflow model '''

	def __init__(self, model_path):
		meta = os.path.join(model_path, '.meta')

		tf.reset_default_graph()
		saver   = tf.train.import_meta_graph(meta)
		graph   = tf.get_default_graph()
		session = tf.Session(graph=graph)
		saver.restore(session, model_path + os.sep) 

		self.sess = session 
		self.X_ph = graph.get_tensor_by_name('Input:0') 
		self.Y_ph = graph.get_tensor_by_name('PredictNetwork_output:0') 
		self.drop = graph.get_tensor_by_name('Dropout:0')
		self.band = self.X_ph.get_shape().as_list()[1]

		with open(os.path.join(model_path, 'scalers.pkl'), 'rb') as f:
			self.x_scaler, self.y_scaler = pkl.load(f)


	def predict(self, data):
		assert(data.shape[1] == self.band),\
			'Input data has incorrect number of bands: %s vs %s' % (data.shape[1], self.band) 

		data = self.x_scaler.transform(data)
		pred = self.sess.run(self.Y_ph, feed_dict={
			self.X_ph: data, 
			self.drop: 0
		})
		return self.y_scaler.inverse_transform(pred)



def train_network(sensor_source, sensor_target, 
					save_path='Predictions',
					data_path='Data',
					build_path='Build',
					train_fmt ='LUT/Rrs_LUT_%s',
					test_fmt  ='In Situ/Rrs_insitu_%s',
					filename  = None,
					gridsearch=False):
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	if sensor_target == sensor_source: 
		print('Attempting to train network with same source and target')
		return

	if gridsearch: print('WARNING: Gridsearch can take a very long time.')

	test_data_path = os.path.join(data_path, test_fmt % sensor_source) if filename is None else filename
	source_data = get_data(test_data_path)

	print('Creating %s to %s' % (sensor_source, sensor_target))
	predictions = []
	iterator = tqdm.trange(n_bands[sensor_target], postfix={'TargetBand':0})
	for idx in iterator:
		iterator.set_postfix(TargetBand=idx+1)

		model_path = os.path.join(build_path, '%s_%s_%s' % (sensor_source, sensor_target, idx))
		if not os.path.exists(os.path.join(model_path, 'scalers.pkl')):
			print('No saved model for band %s of %s -> %s : now building' % (idx, sensor_source, sensor_target)) 
			
			train_source_path = data_path + train_fmt % sensor_source
			train_target_path = data_path + train_fmt % sensor_target

			X = get_data(train_source_path)[:, :n_bands[sensor_source]]
			Y = get_data(train_target_path)[:, idx:idx+1]

			scaler   = RobustScaler
			x_scaler = scaler()
			y_scaler = scaler()

			ids = np.arange(X.shape[0])
			np.random.shuffle(ids)
			
			train_data = (x_scaler.fit_transform(X[ids[:int(len(ids) * 0.7)]]), 
						  y_scaler.fit_transform(Y[ids[:int(len(ids) * 0.7)]]))

			valid_data = (x_scaler.transform(X[ids[int(len(ids)*0.7):]]),
						  y_scaler.transform(Y[ids[int(len(ids)*0.7):]]))

			all_data = (np.append(train_data[0], valid_data[0], axis=0),
						np.append(train_data[1], valid_data[1], axis=0))

			params = {}
			if gridsearch:
				model = GridSearchCV(
							DNN(), 
							{'learning_rate' : (1e-5, 5e-5, 1e-4,), 
							'hidden_layers':[[nn]*nl for nl in range(6, 11, 4) for nn in [50, 100,]],
							'dropout_rate':[0.1,0.2,],
							'l2_rate':[1e-3,1e-5]},
							scoring='neg_mean_squared_error',
							cv = 3, refit=False,
				)
				model.fit(*all_data)
				params = model.best_params_

			params['save_path'] = model_path
			params['maximum_iter'] = 100000
			model  = DNN(**params)
			model.fit(*train_data)

			with open(os.path.join(model_path, 'scalers.pkl'), 'wb+') as f:
				pkl.dump([x_scaler, y_scaler], f)

		model = Model(model_path)
		predictions.append( model.predict(source_data) )
		model.sess.close()

	predictions = np.array(predictions)
	save_file   = os.path.join(save_path, '%s_to_%s_DNN.csv' % (sensor_source, sensor_target))
	np.savetxt(save_file, predictions, delimiter=',')
	return predictions


if __name__ == '__main__':
	source_target = [('OLI','MSI'), ('MSI', 'OLI'), ('OLCI','VI'), ('OLCI','OLI'), 
					 ('OLCI', 'MSI'), ('VI', 'MSI'), ('VI', 'OLI')]
	for sensor_source,sensor_target in source_target:
		train_network(sensor_source, sensor_target, gridsearch=False)

	
				