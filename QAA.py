from scipy.interpolate import Akima1DInterpolator as Akima
import numpy as np


'''
Landsat-8 / OLI
Sentinel-2 / MSI
Sentinel-3 / OLCI
Suomi NPP / VIIRS
'''
wavelengths = {
    'OLI'   : [442.98, 482.49, 561.33, 654.61],
    'MSI'   : [443.93, 496.54, 560.01, 664.45],
    'OLCI'  : [411.3999939, 442.63000488, 490.07998657, 510.07000732, 560.05999756, 619.97998047, 664.85998535, 673.61999512, 681.15002441], # 9 band insitu
    'OLCI2' : [400, 412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25, 708.75, 753.75, 761.25, 764.375, 767.5, 778.75], # 16 band LUT
    'VI'    : [412.49, 444.17, 486.81, 549.99, 670.01], 
    'AER'   : [412, 442, 490, 530, 551, 668],
}


# SeaWiFS coefficients
h0 = -1.14590292783408
h1 = -1.36582826429176
h2 = -0.469266027944581 

# Gordon
g0 = 0.0949
g1 = 0.0794

# Lee
g0 = 0.084
g1 = 0.17

# QAA
g0 = 0.08945
g1 = 0.1247  

water_absorption_file = 'Data/IOP/aw_bw'
water_scattering_file = 'Data/IOP/bbw'
def water_interpolators():
	a_data = np.loadtxt(water_absorption_file, delimiter=',')
	s_data = np.loadtxt(water_scattering_file, delimiter=',')
	return Akima(*a_data.T), Akima(*s_data.T)
absorb, scatter = water_interpolators()

find = lambda k, wavelength: np.abs(wavelength - k).argmin() 	# Index of closest wavelength
key  = lambda k, wavelength: wavelength[find(k, wavelength)]	# Value of closest wavelength

to_rrs = lambda Rrs: Rrs / (0.52 + 1.7 * Rrs)
to_Rrs = lambda rrs: (rrs * 0.52) / (1 - rrs * 1.7)


def QAA(data, wavelength, lambda_reference=443):
	# QAAv5: http://www.ioccg.org/groups/Software_OCA/QAA_v5.pdf
	wavelength = np.array(wavelength)

	# Functional interface into matrix
	idx  = lambda f: (lambda k: f[:, find(k, wavelength)][:, None] if k is not None else f)

	Rrs = idx( np.atleast_2d(data) )
	rrs = idx( to_rrs(Rrs(None)) )

	# Invert rrs formula to find u
	u  = idx( (-g0 + (g0**2 + 4 * g1 * rrs(None)) ** 0.5) / (2 * g1) )

	# Next couple steps depends on if Rrs(670) is lt/gt 0.0015
	QAA_v5 = Rrs(670) < 0.0015
	a_full = np.zeros(QAA_v5.shape) # a(lambda_0)
	b_full = np.zeros(QAA_v5.shape) # b_bp(lambda_0)
	l_full = np.zeros(QAA_v5.shape) # lambda_0

	# --------------------
	# If Rrs(670) < 0.0015
	if QAA_v5.sum():
		lambda0 = key(555, wavelength)
		a_w = absorb(lambda0)
		b_w = scatter(lambda0)
		chi = np.log10( (rrs(443) + rrs(489)) / 
						(rrs(lambda0) + 5 * (rrs(670) / rrs(489)) * rrs(670)) )

		a = a_w + 10 ** (h0 + h1 * chi + h2 * chi**2)
		b = (u(lambda0) * a) / (1 - u(lambda0)) - b_w

		a_full[QAA_v5] = a[QAA_v5]
		b_full[QAA_v5] = b[QAA_v5]
		l_full[QAA_v5] = lambda0
	# --------------------

	# --------------------
	# else
	if (~QAA_v5).sum():
		lambda0 = key(670, wavelength)
		a_w = absorb(lambda0)
		b_w = scatter(lambda0)

		a = a_w + 0.39 * ( Rrs(670) / (Rrs(443) + Rrs(489)) ) ** 1.14
		b = (u(lambda0) * a) / (1 - u(lambda0)) - b_w

		a_full[~QAA_v5] = a[~QAA_v5]
		b_full[~QAA_v5] = b[~QAA_v5]
		l_full[~QAA_v5] = lambda0
	# --------------------

	# Back to the same steps for all data
	a0 = a_full
	b0 = b_full
	l0 = l_full

	eta = 2 * (1 - 1.2 * np.exp(-0.9 * rrs(443) / rrs(555)))

	b = b0 * (l0 / wavelength) ** eta
	a = (1 - u(None)) * (scatter(wavelength) + b) / u(None)
	a = idx(a)

	# Now decompose the absorption
	zeta = 0.74 + (0.2 / (0.8 + rrs(443) / rrs(555)))

	S  = 0.015 + (0.002 / (0.6 + rrs(443) / rrs(555)))
	xi = np.exp(S * (key(443, wavelength) - key(412, wavelength))) 

	a_dg443 =  (a(412) - zeta * a(443)) / (xi - zeta) \
			- (absorb(412) - zeta * absorb(443)) / (xi - zeta)

	a_dg = a_dg443 * np.exp(S * (key(443, wavelength) - wavelength))
	a_ph = a(None) - a_dg - absorb(443)

	# QAA-CDOM - Zhu & Yu 2013
	# b[b < 0] = 1e-5
	# a_ph[a_ph < 0] = 1e-5
	# a_dg[a_dg < 0] = 1e-5
	# a_p = 0.63 * b ** 0.88
	# a_g = a(None) - absorb(440) - a_p  
	
	return (idx(b)(lambda_reference), 
			idx(a_ph)(lambda_reference), 
			idx(a_dg)(lambda_reference), 
			eta, S)


def melin(source_data, source_wavelengths, target_wavelengths):
	source_data = np.atleast_2d(source_data)
	source_wave = np.array(source_wavelengths)
	target_wave = np.array(target_wavelengths)
	assert(source_wave.shape[0] == source_data.shape[1]), \
		'Data / Wavelength mismatch: %s' % str([source_wave.shape[0], source_data.shape[1]])

	AB = np.loadtxt('Data/IOP/AB_Bricaud.csv', delimiter=',')
	A_interp = Akima(AB[:, 0], AB[:, 1])
	B_interp = Akima(AB[:, 0], AB[:, 2])

	lambda_reference = 443
	b, a_ph, a_dg, eta, S = QAA(source_data, source_wave, lambda_reference)

	melin_out = []
	for lambda_target in target_wave:
		lambda_source_i  = [find(lambda_target, source_wave)]
		lambda_sources   = [source_wave[lambda_source_i[0]]]

		# Distance too great - use weighted average if not first / last index
		if abs(lambda_target - lambda_sources[0]) > 3 and lambda_source_i[0] not in [0, len(source_wave)-1]:
			lambda_source_i.append(lambda_source_i[0] + (1 if lambda_sources[0] < lambda_target else -1))
			lambda_sources.append(source_wave[lambda_source_i[1]])

		Rrs_es = []
		for lambda_source_idx, lambda_source in zip(lambda_source_i, lambda_sources):

			Rrs_fs = []
			for lmbda in [lambda_source, lambda_target]:
				bbp = b * (lambda_reference / lmbda) ** eta
				aph = A_interp(lmbda) * (a_ph / A_interp(lambda_reference)) ** ((1 - B_interp(lmbda)) / (1 - B_interp(lambda_reference)))
				acd = a_dg * np.exp(-S * (lmbda - lambda_reference))
				
				a = aph + acd + absorb(lmbda)
				b = bbp + scatter(lmbda)
				rrs_f = g0 * (b / (b + a)) + g1 * (b / (b + a)) ** 2
				Rrs_fs.append( to_Rrs(rrs_f).flatten() )

			Rrs_source, Rrs_target = Rrs_fs
			Rrs_es.append( Rrs_target * (source_data[:, lambda_source_idx] / Rrs_source) )
		
		if len(lambda_sources) > 1:
			Rrs_e = np.abs(lambda_sources[1] - lambda_target) * Rrs_es[0] + np.abs(lambda_sources[0] - lambda_target) * Rrs_es[1]
			Rrs_e/= np.abs(lambda_sources[0] - lambda_sources[1])
		else:
			Rrs_e = Rrs_es[0]

		melin_p = Rrs_e
		melin_out.append(melin_p)

	return np.array(melin_out)