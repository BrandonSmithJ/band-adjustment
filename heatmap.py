from matplotlib import colors, cm, colorbar, ticker, patches
from create_results import get_results, sensor_labels, insitu_file_fmt, load_Rrs
from QAA import wavelengths
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np


if __name__ == '__main__':
	data = get_results()
	order = ['MSI', 'OLI', 'VI', 'OLCI']

	vmin = -1e-3
	vmax =  1e-3
	cmap = cm.Reds
	span = 5
	tick = [5e-5,  5e-4,  5e-3]
	loc  = ticker.FixedLocator(tick)
	
	plt.figure(figsize=(13,6))

	n_methods = len(data.methods)-1
	n_sensors = len(sensor_labels)-1
	axs = []
	ks = [0, 4, 8]
	rs = [4, 4, 2]
	for k, target_sensor in enumerate(order[:-1]):

		for j, method in enumerate(list(sorted(data.methods))[:1] + list(sorted(data.methods))[2:]+[list(sorted(data.methods))[1]]):
			ax = plt.subplot2grid((10, span*len(data.methods)+3), (ks[k], j*span), colspan=span, rowspan=rs[k])
			axs.append(ax)

			ylbl= []
			err = []
			for source_sensor in sorted(order):
				if source_sensor == target_sensor: 
					continue

				if source_sensor == 'OLI' and target_sensor != 'MSI': continue
				if source_sensor == 'MSI' and target_sensor != 'OLI': continue
				if source_sensor == 'VI' and target_sensor not in ['MSI', 'OLI']: continue

				target = load_Rrs(insitu_file_fmt % target_sensor)
				if target_sensor == 'MSI': target = target[:,:4]
				target_wave = np.array(wavelengths[target_sensor][:9])

				if 'Mélin' in method and source_sensor in ['MSI', 'OLI']:
					err.append([np.nan] * len(target_wave))
				else:
					err.append( ((data[method][(source_sensor, target_sensor)] - target)**2).mean(axis=0)**.5 )

				ylbl.append(source_sensor.replace('VI', 'VIIRS'))

			ylbl = ylbl[::-1]
			h = sns.heatmap(err, ax=ax, cbar=False, cmap=cmap, linecolor='black', linewidth=1, robust=True, square=True,
							norm=colors.LogNorm(vmin=5e-5, vmax=5e-3), vmin=5e-5, vmax=5e-3, cbar_kws={'ticks':loc})#, norm=colors.SymLogNorm(8e-5))
			
			def plt_x(sensor_name):
				x1, x2 = ax.get_xlim()
				y = [ylbl.index(sensor_name), ylbl.index(sensor_name)+1]
				ax.add_patch(
					patches.Rectangle(
						(x1, y[0]), 
						x2 - x1, # width
						1, # height
				))

			if 'Mélin' in method:
				if 'MSI' in ylbl: plt_x('MSI') 
				if 'OLI' in ylbl: plt_x('OLI')

			ax.set_xticklabels([r'$\mathrm{\bf{%s}}$'%xl for xl in np.round(target_wave).astype(np.int32)], fontsize=14, weight='extra bold')
			ax.set_xticks(ax.get_xticks(), [r'$\mathbf{%s}$' % x for x in np.round(target_wave).astype(np.int32)])
			if k == 0: 	ax.set_title(method, fontsize=16)
			if j == n_methods:	
				ax.set_yticklabels([])
				ax.set_ylabel(r'$\sf{\bf{%s}}$%s$\bf{%s}$$\sf{\bf{ / %s}}$' % (sensor_labels[target_sensor].split('-')[0],
																		'-' if len(sensor_labels[target_sensor].split('-')) == 2 else '',
																		sensor_labels[target_sensor].split('-')[1] if 
																			len(sensor_labels[target_sensor].split('-')) == 2 else '',
																		 target_sensor.replace('VI', 'VIIRS')), 
								fontsize=10 if target_sensor == 'AER' else 14)
				ax.yaxis.set_label_position('right')
			elif j == 0: 
				ax.set_yticklabels([r'$\bf{%s}$'%yl for yl in ylbl], rotation='horizontal', fontsize=12)
			else: ax.set_yticklabels([])


	ax = plt.subplot2grid((len(sensor_labels)-1, span*len(data.methods)+3), (0, span*j+span+2), rowspan=len(sensor_labels)-1, colspan=1)
	cb  = plt.colorbar(h.get_children()[0], cax=ax, ax=axs, orientation='vertical', ticks=loc)
	cb.ax.yaxis.set_ticks(tick)
	cb.locator = loc 
	cb.formatter   = ticker.FixedFormatter(['%.0e' % t for t in tick])
	cb.ax.yaxis.set_ticks_position('left')
	cb.update_ticks()
	cb.set_label(r'RMSE $(\frac{1}{sr})$', fontsize=16)

	plt.gcf().text(.5, 0.02, 'Band Center (nm)', fontsize=18, ha='center')
	plt.gcf().text(.03, .55, 'Reference Sensor', fontsize=18, va='center', rotation='vertical')
	plt.gcf().text(.87, .55, 'Target Sensor', fontsize=18, va='center', rotation='vertical')

	plt.subplots_adjust(wspace=1, hspace=0, right=.95, left=.1, bottom=.06, top=.95)#left=.1, wspace=.07, hspace=.07, bottom = .18, top=.9, right=.86)
	# plt.savefig('Results/heatmap.png', dpi=600)
	plt.show()
