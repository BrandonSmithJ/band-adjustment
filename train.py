'''
All that's needed is to modify the 'filename_fmt' variable below 
to point to whatever reference input file you want to convert.
The format should be csv, with each row corresponding to a 
wavelength of the reference sensor (see 'n_bands' variable for 
how many the algorithm is expecting).

The model won't perform well if there's any preprocessing to the input 
data (e.g. subtracting the mean, normalizing, etc.) - the file at 
'filename_fmt' should follow the same format and be the same type of data 
as is found at the current filename_fmt (i.e. Rrs).

To run:
	"python3 predict.py" 
'''

# change 'filename_fmt' to file format, where %s represents the sensor name
# 	abbreviation (seen below in 'n_bands'). E.g. if the input files follow the 
# 	format 'Rrs_insitu_AER_Full_V3', the filename_fmt should be 'Rrs_insitu_%s_Full_V3'
filename_fmt = 'Rrs_insitu_%s_Full_V3'

''' 
Shouldn't have to modify below this line
'''
data_path = 'Data/' 
build_path= 'Build/%s/'
save_path = 'Predictions_export/'

n_bands = {
	'MSI' : 4,
	'OLI' : 4,
	'VI'  : 5,
	'OLCI': 9,
	# 'AER' : 6,
}

from DNN import DNN 
from sklearn.preprocessing import RobustScaler
import tensorflow as tf 
import pickle as pkl 
import numpy as np 
import os 


def get_data(path):
	assert(os.path.exists(path)), 'Data file does not exist: %s'%path

	data = np.array([l.strip().split(',') for l in 
		open(path).readlines()]).astype(np.float32).T 
	if np.sum(data[0]) > 10:
		print('First row of data assumed to be wavelengths (nm)')
		data = data[1:]
	return data

class Model(object):
	''' Wrapper for a saved tensorflow model '''

	def __init__(self, model_path):
		path = build_path % model_path
		meta = path + '.meta'

		tf.reset_default_graph()
		saver   = tf.train.import_meta_graph(meta)
		graph   = tf.get_default_graph()
		session = tf.Session(graph=graph)
		saver.restore(session, path) 

		self.sess = session 
		self.X_ph = graph.get_tensor_by_name('Input:0') 
		self.Y_ph = graph.get_tensor_by_name('PredictNetwork_output:0') 
		self.drop = graph.get_tensor_by_name('Dropout:0')
		self.band = self.X_ph.get_shape().as_list()[1]

		with open(path + 'scalers.pkl', 'rb') as f:
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




if __name__ == '__main__':
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	for sensor_source in n_bands.keys():
		for sensor_target in n_bands.keys():
			if sensor_target == sensor_source: 
				continue

			test_data_path = data_path + filename_fmt
			source_data = get_data(test_data_path % sensor_source)
			predictions = []
			for idx in range(n_bands[sensor_target]):
				print('%s to %s - target band %s' % (sensor_source, sensor_target, idx))

				model_path = '%s_%s_%s'%(sensor_source, sensor_target, idx)
				if not os.path.exists(build_path % model_path + 'scalers.pkl'):
					print('No saved model for band %s of %s -> %s : now building' % (idx, sensor_source, sensor_target)) 
					
					train_source_path = data_path + 'Train/%s' % sensor_source
					train_target_path = data_path + 'Train/%s' % sensor_target

					X = get_data(train_source_path)
					Y = get_data(train_target_path)[:, idx:idx+1]

					scaler   = RobustScaler
					x_scaler = scaler()
					y_scaler = scaler()

					ids = np.arange(X.shape[0])
					np.random.shuffle(ids)
					
					train_data = (x_scaler.fit_transform(X[ids[0:]]), 
								  y_scaler.fit_transform(Y[ids[0:]]))

					model = GridSearchCV(
								DNN(), 
								{'learning_rate' : (1e-5, 3e-5, 5e-5, 7e-5, 1e-4,), 
								'hidden_layers':[[nn]*nl for nl in range(2, 11, 4) for nn in [50, 100, 200]],
								'dropout_rate':[0.1,0.2,0.3],
								'l2_rate':[1e-3,1e-4,1e-5]},
								scoring='neg_mean_squared_error',
								cv = 3, refit=False,
					)
					model.fit(*train_data)
					params = model.best_params_
					params['save_path'] = model_path
					model  = DNN(**params)
					model.fit(*train_data)

					with open(build_path % model_path + 'scalers.pkl', 'wb+') as f:
						pkl.dump([x_scaler, y_scaler], f)

				model = Model(model_path)
				predictions.append( model.predict(source_data) )
				model.sess.close()

			predictions = np.array(predictions)
			save_file   = '%s%s_to_%s_DNN.csv' % (save_path, sensor_source, sensor_target)
			np.savetxt(save_file, predictions, delimiter=',')
					