from scipy.interpolate import CubicSpline
from scipy.stats import linregress
from QAA import melin, wavelengths
from train import train_network
import pickle as pkl
import numpy as np 
import os 


names = [
	'Mélin & Sclep (2015)',
	'Spectral Matching',
	'Deep Neural Network',
	'Cubic Spline',
]

sensor_labels = {
	'OLI':'Landsat-8',
	'MSI':'Sentinel-2', 
	'OLCI':'Sentinel-3', 
	'VI': 'SNPP', 
	'AER':'AERONET-OC'
}

data_filename = 'Results/stored_results.pkl'
SM_file_fmt   = 'Data/Spectral Matching/%s_to_%s.npy'
DNN_file_fmt  = 'Predictions/%s_to_%s_DNN.csv'
insitu_file_fmt = 'Data/In Situ/Rrs_insitu_%s'

rmse = lambda y, y_hat: float(((y - y_hat)**2).mean() ** .5)
rrms = lambda y, y_hat: float((((y-y_hat)/y)**2).mean() ** .5)
bias_mean = lambda y, y_hat: float(np.mean(y_hat - y))
bias_med  = lambda y, y_hat: float(np.median(y_hat - y))
bias_std  = lambda y, y_hat: float(np.std(y_hat - y))

def load_Rrs(filename):
	data = np.loadtxt(filename, delimiter=',', dtype=np.float32)
	if data.shape[1] > data.shape[0]: data = data.T 
	if data[0].sum() > 10: data = data[1:]
	assert(data[0].sum() < 10), \
		'Sum of first row of data for "%s" was > 10. This should not happen; '+\
		'Is any preprocessing applied which shouldn\'t be?'
	return data 


def create_data():
	''' Gather all results in consistent format, then write to output file '''
	if not os.path.exists('Results'):
		os.mkdir('Results')
		
	num_points  = None
	sensor_data = {}
	write_data  = []
	for k, source_sensor in enumerate(sorted(sensor_labels)):
		for k2, target_sensor in enumerate(sorted(sensor_labels)):			
			if source_sensor == target_sensor:
				continue 

			target = load_Rrs(insitu_file_fmt % target_sensor)
			source = load_Rrs(insitu_file_fmt % source_sensor)

			if source_sensor == 'MSI': source = source[:,:4]
			if target_sensor == 'MSI': target = target[:,:4]

			DNN_filename = DNN_file_fmt % (source_sensor, target_sensor)
			SM_filename  = SM_file_fmt % (source_sensor, target_sensor)

			if not os.path.exists(DNN_filename): train_network(source_sensor, target_sensor)
			if not os.path.exists(SM_filename):  continue 

			if num_points is not None:
				assert(target.shape[0] == num_points), 'Different number of samples: %s != %s' % (target.shape[0], num_points)
			num_points = target.shape[0]

			target_wave = np.array(wavelengths[target_sensor][:9])
			source_wave = np.array(wavelengths[source_sensor][:9])

			f = {
				'Mélin & Sclep (2015)' : (
					lambda: melin(source, source_wave, target_wave).T
				), 'Spectral Matching' : (
					lambda: np.load(SM_filename)
				), 'Cubic Spline' : (
					lambda: CubicSpline(source_wave, source.T, axis=0, extrapolate=True)(target_wave).T
				), 'Deep Neural Network' : (
					lambda: load_Rrs(DNN_filename)
				), 
			}

			method_data = {}
			for name in names:
				v = f[name]()

				if target_sensor == 'MSI': v = v[:,:4]
				method_data[name] = v
				write_data.append({
					'Method'   : name,
					'Reference': source_sensor,
					'Target'   : target_sensor,
					'RMSE'	   : np.mean([float(rmse(vv, t)) for vv, t in zip(v.T, target.T)]),
					'RRMS' 	   : rrms(target, v),
					'Diff_mean': bias_mean(target, v),
					'Diff_med' : bias_med(target, v),
					'Diff_std' : bias_std(target, v),
					
					'RMSE_band': [float(rmse(vv, t)) for vv, t in zip(v.T, target.T)], 
					'RRMS_band': [float(rrms(vv, t)) for vv, t in zip(v.T, target.T)],
					'Diff_mean_band': [float(bias_mean(vv, t)) for vv, t in zip(v.T, target.T)],
					'Diff_med_band' : [float(bias_med(vv, t)) for vv, t in zip(v.T, target.T)],
					'Diff_std_band' : [float(bias_std(vv, t)) for vv, t in zip(v.T, target.T)],
					
					'Slope_band'    : [linregress(vv, t)[0] for vv,t in zip(v.T, target.T)],
					'Intercept_band': [linregress(vv, t)[1] for vv, t in zip(v.T, target.T)],
					'Rsquared_band' : [linregress(vv, t)[2] for vv, t in zip(v.T, target.T)],
				})

			sensor_data[(source_sensor, target_sensor)] = method_data

	with open(data_filename, 'wb+') as f:
		pkl.dump(sensor_data, f)

	write_keys = ['Reference', 'Target', 'Method', 'RMSE','RRMS', 'Diff_mean', 'Diff_med', 'Diff_std',
					'RMSE_band', 'RRMS_band', 'Diff_mean_band', 'Diff_med_band' , 'Diff_std_band',
					'Slope_band', 'Intercept_band', 'Rsquared_band']
	
	max_num_bands = max(len(wavelengths[k]) for k in sensor_labels)
	with open('Results/stats.csv', 'w+') as f:
		band_keys = [w for w in write_keys if '_band' in w]
		over_keys = [w for w in write_keys if '_band' not in w]
		w_keys = over_keys + [k for w in band_keys for k in 
				['%s_%s'%(w,i) for i in range(max_num_bands)]]

		f.write(','.join(w_keys) + '\n')
		for line in write_data:
			vals = [str(line[w]).replace(',',';') for w in over_keys] 
			for w in band_keys:
				vals += [str(k) for k in line[w]]
				if len(line[w]) < max_num_bands: 
					vals += [''] * (max_num_bands - len(line[w]))
			f.write(','.join(vals) + '\n')


def get_results(overwrite=False):
	''' Reads in data, then wraps to allow 
			e.g. df['MLP'] to get MLP results ) '''

	if overwrite or not os.path.exists(data_filename):
		create_data()

	with open(data_filename, 'rb') as f:
		results = pkl.load(f)

	class Results(object):

		def __init__(self, data):
			self.data = data
			self.sources = [k[0] for k in data.keys()]
			self.targets = [k[1] for k in data.keys()]
			self.methods = list(data[list(data.keys())[0]].keys())

		def __getitem__(self, key):
			if key in self.data:
				return self.data[key]
			
			elif key in names:
				return self.get_method(key)
			
			else: raise KeyError(key)

		def __repr__(self): return self.data.__repr__()
		def __str__(self):  return self.data.__str__()
		

		def get_source(self, source):
			keys = [k for k in self.data.keys() if "'%s', " in str(k)]
			data = {}
			for source, target in keys:
				data[target] = self.data[(source, target)]
			return data 


		def get_target(self, target):
			keys = [k for k in self.data.keys() if ", '%s'" in str(k)]
			data = {}
			for source, target in keys:
				data[source] = self.data[(source, target)]
			return data 


		def get_method(self, method):
			data = {}
			for key in self.data:
				data[key] = self.data[key][method]
			return data

	return Results(results)


if __name__ == '__main__': get_results(True)
