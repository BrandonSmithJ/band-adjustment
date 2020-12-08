from matplotlib import colors, cm, colorbar, ticker, patches
from create_results import get_results, sensor_labels, insitu_file_fmt, load_Rrs
from QAA import wavelengths
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

def RMSE(y, y_hat): return np.nanmean((y - y_hat) ** 2, axis=0) ** 0.5


def format_sensor_label(sensor):
	part1, part2, *_ = (sensor_labels[sensor]+'-').split('-')
	dash   = '-' if part2 else ''
	sensor = sensor.replace('VI', 'VIIRS')
	return fr'$\sf{{ {part1} }}${dash}$\sf{{ {part2}\ /\ {sensor} }}$'


def plt_unavailable(ax, sensor, ylbl):
	x1, x2 = ax.get_xlim()
	y = [ylbl.index(sensor), ylbl.index(sensor)+1]
	ax.add_patch(patches.Rectangle((x1, y[0]), x2 - x1, 1))


if __name__ == '__main__':
	data  = get_results(True)
	order = ['MSI', 'OLI', 'VI', 'OLCI']

	methods   = ['Cubic Spline', 'Mélin & Sclep (2015)', 'Spectral Matching', 'Deep Neural Network']
	n_methods = len(data.methods)-1
	n_sensors = len(sensor_labels)-1
	assert(sorted(methods) == sorted(data.methods)), f'Mismatch between available and expected methods: {data.methods} vs {methods}'
	
	axs  = []
	ks   = [0, 4, 8] # Row locations for target sensors
	rs   = [4, 4, 2] # Row span for target sensors
	cs   = 5         # Column span
	tick = [5e-5,  5e-4,  5e-3]
	loc  = ticker.FixedLocator(tick)
	fmt  = ticker.FixedFormatter(['%.0e' % t for t in tick])
	cmap = cm.Reds
	vkws = {'vmin': min(tick), 'vmax': max(tick)}
	hkws = {'cbar': False, 'cmap': cmap, 'linecolor': 'k', 'linewidth': 1, 'robust': True, 'square': True, 'norm': colors.LogNorm(**vkws)}
	hkws.update(vkws)

	plt.figure(figsize=(13, 6))

	for k, target_sensor in enumerate(order[:-1]):
		for j, method in enumerate(methods):
			ylbl = []
			err  = []
			ax   = plt.subplot2grid((10, cs*len(data.methods)+3), (ks[k], j*cs), colspan=cs, rowspan=rs[k])
			axs.append(ax)

			sensor_lbl  = sensor_labels[target_sensor]
			target_wave = np.array(wavelengths[target_sensor][:9])
			target      = load_Rrs(insitu_file_fmt % target_sensor)
			if target_sensor == 'MSI': target = target[:,:4]

			for source_sensor in sorted(order):
				if source_sensor == target_sensor: continue
				if source_sensor == 'OLI' and target_sensor != 'MSI': continue
				if source_sensor == 'MSI' and target_sensor != 'OLI': continue
				if source_sensor == 'VI'  and target_sensor not in ['MSI', 'OLI']: continue

				if 'Mélin' in method and source_sensor in ['MSI', 'OLI']:
					err.append([np.nan] * len(target_wave))
				else:
					if method == 'Mélin & Sclep (2015)': print(RMSE(target, data[method][(source_sensor, target_sensor)]))
					err.append( RMSE(target, data[method][(source_sensor, target_sensor)]) )
				ylbl.append(source_sensor.replace('VI', 'VIIRS'))

			h = sns.heatmap(err, ax=ax, **hkws)
			
			if 'Mélin' in method:
				if 'MSI' in ylbl: plt_unavailable(ax, 'MSI', ylbl) 
				if 'OLI' in ylbl: plt_unavailable(ax, 'OLI', ylbl)

			ax.set_xticklabels([fr'$\sf{{ {x} }}$' for x in np.round(target_wave).astype(np.int32)], fontsize=14)
			ax.set_xticks(ax.get_xticks(), ax.get_xticklabels())
			
			if k == 0: ax.set_title(method, fontsize=16)
			if j == 0: ax.set_yticklabels(ylbl, rotation='horizontal', fontsize=12)
			elif j == n_methods:	
				ax.set_ylabel(format_sensor_label(target_sensor), fontsize=10 if target_sensor == 'AER' else 14)
				ax.yaxis.set_label_position('right')
				ax.set_yticklabels([])
			else: ax.set_yticklabels([])

	ax = plt.subplot2grid((len(sensor_labels)-1, cs*len(data.methods)+3), (0, cs*j+cs+2), rowspan=len(sensor_labels)-1, colspan=1)
	cb = plt.colorbar(h.get_children()[0], cax=ax, ax=axs, orientation='vertical', ticks=loc)
	cb.ax.yaxis.set_ticks(tick)
	cb.locator   = loc 
	cb.formatter = fmt
	cb.ax.yaxis.set_ticks_position('left')
	cb.update_ticks()
	cb.set_label(r'RMSE $(sr^{-1})$', fontsize=16)

	plt.gcf().text(.5, 0.02, 'Band Center (nm)', fontsize=18, ha='center', weight='bold')
	plt.gcf().text(.03, .55, 'Reference Sensor', fontsize=18, va='center', rotation='vertical', weight='bold')
	plt.gcf().text(.87, .55, 'Target Sensor', fontsize=18, va='center', rotation='vertical', weight='bold')

	plt.subplots_adjust(wspace=1, hspace=0, right=.95, left=.1, bottom=.06, top=.95)#left=.1, wspace=.07, hspace=.07, bottom = .18, top=.9, right=.86)
	# plt.savefig('Results/heatmap.png', dpi=600)
	plt.show()
