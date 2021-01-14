# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:22:10 2020
# Some of the subroutines in this model have been translated from fortran to Python . Original fortran code is in the MSc thesis of Jonica Vidrine (1996), only a printed version is available, and can be 
# obtained from University of Delaware upon request.
rebecca.rolph@awi.de
"""

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from pathlib import Path
import os
import datetime

import global_variables_for_batch as global_variables


''' Definitions of model parameters and forcing variables

# Beach 
Wb = initial beach width
tanthb = initial beach slope
Bb = initial unfrozen beach sediment beach thickness
Pb = coarse sediment volume per unit volume of unfrozen beach sediment
vb = coarse sediment volume per unit volume of frozen beach sediment
ksb = equivalent sand roughness of the frozen beach sediment melting surface where ksb=2.5*d50, where d50 = mean diameter of the frozen beach sediment
nib = ice voulme per unit volume of the frozen beach sediment
rhoi = density of bubble-free pure ice = 920 [kg/m^3]
Li = latent heat of melting = 3.3e5 [J/kg]
Lb = nib*rhoi*Li = volumetric latent heat of melting per unit volume of the frozen beach sediment [J/m^3]

# Cliff
Hc = cliff height above MSL
thetac = seaward cliff slope angle in degrees as input
Bc = unfrozen cliff sediment thickness
Pc = coarse sediment volume per unit volume of unfrozen cliff sediment
vc = coarse sediment volume per unit volume of frozen cliff sediment
nic = ice volume per unit volume of frozen cliff sediment
Lc = volumetric latent heat of melting per unit volume of frozen cliff sediment [J/m^3]
ksc = equivalent sand roughness of frozen cliff sediment melting surface where ksc = 2.5d50

# Seawater
Tw = ocean temperature
Sw = seawater salinity < 35%
Tm = melting temperature [C] of ice
Cw = volumetric heat capacity [J/m^3]
v = kinematic viscosity
Kw = thermal conductivity

# Waves
Hrms = incident rms wave height
Tr = representative wave period [sec] during a storm
wr = angular frequency
gammab = emperical breaking parameter = 0.4
beta = emperical runup parameter = 1.0

# Sediment transport
A = empirical parameter for equilibrium beach profile
alpha = empirical parameter for sediment transport rate parameter
# Heat transfer
adjust = empirical adjustment for hw (heat transfer coefficient)

# Storm surge
STIME = storm duration [hours]
SURGE = storm surge above the mean sea level (MSL)

# Other parameters
db = water depth at seaward boundary
dr = representative water depth where friction factors are calculated in subroutine FWROOT (added)
ks = bottom roughness specification used to calculate near bottom fluid velocity in subroutine FWROOT (added)
dbc = water depth at toe of cliff
WBDB = beach width from the seaward boundary to the toe of the cliff
'''

def erosion_main(water_level_offset):

	# call/rename the global variables for clarity
	basepath = global_variables.basepath
	datapath_io = global_variables.datapath_io

	# add the offset to the water level.
	water_level_meters = global_variables.water_level_meters + water_level_offset

	## call global variables

	# beach input variables
	Wb = global_variables.beach_params.Wb
	tanthb = np.tan(np.radians(global_variables.beach_params.thb))
	Bb = global_variables.beach_params.Bb
	Pb = global_variables.beach_params.Pb
	vb = global_variables.beach_params.vb
	ksb = global_variables.beach_params.ksb

	# cliff input variables
	Hc = global_variables.cliff_params.Hc
	sinthc = np.sin(np.radians(global_variables.cliff_params.thc))
	Bc = global_variables.cliff_params.Bc
	Pc = global_variables.cliff_params.Pc
	vc = global_variables.cliff_params.vc
	nic = global_variables.cliff_params.nic
	ksc = global_variables.cliff_params.ksc

	# other globals
	Sw = global_variables.Sw
	gammab = global_variables.gammab
	beta = global_variables.beta
	A = global_variables.A
	alpha = global_variables.alpha
	swh = global_variables.Hrms
	wave_period_site = global_variables.Tr
	sst = global_variables.Tw
	adjust = global_variables.adjust
	start_month = global_variables.start_month
	start_day = global_variables.start_day
	year = global_variables.year

	# set constants
	STIME = water_level_meters.shape[0]
	rhoi = 920.0
	nib = 1-vb
	Li = 330000.0
	Lb = nib*rhoi*Li
	Tm = -0.06*Sw
	Lc = nic*rhoi*Li
	db = 4/9*A**3/(tanthb**2)
	WBDB = Wb + db/tanthb
	XACC = 0.0000001
	X1 = 0.002
	X2 = 0.10
	switch = 'init'

	# Function Friction for fws
	def FRICTION(fw,R):
		friction = 1.0/(8.1*fw**0.5)+np.log10(1/(fw**0.5)) + 0.135 - np.log10(R**0.5)
		return friction

	def FWROOT(X1, X2, XACC, dr, ks, Tw, Tr, Hrms):
		# Calculate krdr using Newton-Raphson method to get Ub Ab
		g = 9.81
		wr  = 2.0*np.pi/Tr
		A = (wr**2)*dr/g
		x = 0.25
		f = x*np.tanh(x) - A
		df = np.tanh(x) + x*(1/np.cosh(x))**2
		xx1 = x - f/df
		err = abs(x - xx1)
		while np.greater(err,1e-6):
			x = xx1
			f = x*np.tanh(x) - A
			df = np.tanh(x) + x*(1/np.cosh(x))**2
			xx1 = x - f/df
			err = abs(x - xx1)
		krdr = x
		Hr = np.amin([gammab * dr, Hrms])
		Ub = Hr*wr/(2*np.sinh(krdr))
		Ab = Ub/wr
		v = 1.787e-6 - 6.25e-8 * Tw + 1.34e-9 * Tw**2
		R = Ub*Ab/v
		# Calculate fws for smooth bed using the bisection method. Range of fws is: 0.002 < fws < 0.02
		IMAX = 40
		FMID = FRICTION(X2, R)
		F = FRICTION(X1, R)
		#print(R)
		if np.greater(F*FMID,0.0):
			print('F times FMID is pos.')
		if np.less(F,0.0):
			fws = X1
			DX = X2 - X1
		else:
			fws = X2
			DX = X1 - X2
		for I in np.arange(0,IMAX):
			DX = DX * 0.5
			XMID = fws + DX
			FMID = FRICTION(XMID, R)
			if np.less_equal(FMID,0):
				fws = XMID
			if np.less(abs(DX),XACC) or FMID==0:
				fwr = np.exp(5.213*(ks/Ab)**0.194 - 5.977)
				v = 1.787e-6 - 6.25e-8 * Tw + 1.34e-9 * Tw**2
				Rsr = ks*Ub/v*(0.5*fwr)**0.5
				Rss = ks*Ub/v*(0.5*fws)**0.5
				if np.less_equal(Rss,5.0): # smooth flow
					fw = fws
				elif np.greater_equal(Rsr,70.0):  # rough flow 
					fw = fwr
				else: # transitional flow
					Rst = ks*Ub/v*(((fws + fwr)/4.0)**0.5)
					fw = fws + (fwr-fws)*(Rst - 5.0)/65.0
		return Ub, fw, fws, fwr

	# Calculate heat transfer coefficient
	def HEATRA(ks, Ub, fw, Tw):
		Cw = -2500*Tw + 4.217e6 # this has been added as an input to this function becuase Tw is no longer assumed constant
		v = 1.787e-6 - 6.25e-8 * Tw + 1.34e-9 * Tw**2 # added bc no longer assumed constant
		Kw = 0.0014*Tw + 0.564 # added bc no longer assumed constant
		P = v*Cw/Kw
		Rs = ks*((0.5*fw)**0.5)*Ub/v
		Es = 5.0*(P - 1 + np.log(1+5/6*(P-1)))  
		if np.less_equal(Rs,5.0):  # Rs needs to be small for Es to be invoked
			E = Es
		elif np.greater_equal(Rs,70):
			E = 0.52*(P**0.8)*(Rs**0.45)
		else:
			E = Es - (Es - 3.52*P**0.8)*(Rs - 5.0)/65.0
		hw = adjust*fw*Cw*Ub/(1 + E*(0.5*fw)**0.5)
		return hw
	'''
	# write some input data for current run to .csv file (e.g. for forcing animation)
	df = pd.DataFrame(np.array([Wb]),columns=['Wb: prescribed initial beach width [m]'])
	df.to_csv(datapath_io+'initial_beach_width_Wb.csv')

	df = pd.DataFrame(([np.array(np.degrees(np.arctan(tanthb)))]),columns=['thb: Beach angle [degrees]'])
	df.to_csv(datapath_io+'beach_angle_degrees_thb.csv')
	'''
	#df = pd.DataFrame(np.array([Bb]),columns=['Bb: prescribed initial unfrozen beach sediment thickness [m]'])
	#df.to_csv(global_variables.path_parameters_tested_outputs_per_year +'initial_unfrozen_beach_sediment_thickness.csv')
	'''
	df = pd.DataFrame(np.array([Hc]),columns=['Hc: Cliff height [m]'])
	df.to_csv(datapath_io+'initial_cliff_height.csv')

	df = pd.DataFrame(([np.array(np.degrees(np.arcsin(sinthc)))]),columns=['thc: Cliff angle [degrees]'])
	df.to_csv(datapath_io+'cliff_angle_degrees_thc.csv')

	df = pd.DataFrame(np.array([Bc]),columns=['Bc: prescribed initial unfrozen sediment thickness on top of cliff [m]'])
	df.to_csv(datapath_io+'initial_unfrozen_cliff_sediment_thickness.csv')

	df = pd.DataFrame(np.array([db]),columns=['db: water depth at seaward boundary [m]'])
	df.to_csv(datapath_io+'initial_water_depth_at_seaward_boundary.csv')
	'''

	number_timesteps_per_storm_timestep = 2
	NTSTEP = STIME*number_timesteps_per_storm_timestep # NTSTEP should be accounted for such that you consider, for example, if you have 1 hour storm surge data, if you have over an hour per timestep it doesnt work . 
	TSTEP = STIME*3600.0/NTSTEP  # [seconds per timestep] based on number of hours you have for storm surge data and number of timesteps. 
	hours_per_timestep = STIME/NTSTEP # [hours per timestep]
	print('NTSTEP:' + str(NTSTEP))
	print('hours per timestep: ' + str(hours_per_timestep))
	TSEC = 0.0 # initialize how many seconds have passed since start of loop
	R = 0.0
	B = Bb
	D = 0.0
	E = 0.0
	NCOUNT = 0

	# initialize timeseries arrays.
	E_all = np.array([99999])
	R_all = np.array([99999])
	S_all = np.array([99999])
	D_all = np.array([99999])
	dc_all = np.array([99999])
	B_all = np.array([99999]) # for calibrating alpha
	qp_all = np.array([99999]) # for developing over longer timescales
	W_all = np.array([99999])
	H_all = np.array([99999]) # bluff height timeseries
	qb_all = np.array([99999])
	qmelt_all = np.array([99999])
	#qmelt_flag = np.array([99999])
	qc_all = np.array([99999]) # corresponding coarse sediment supply rate qc

	# initialize saved arrays of NCOUNTs of storm starts and storm peaks, so you can overlay them onto the surge values to check they are right . also overlay them on qp_all[1:]. 
	NCOUNT_saved_array_of_storm_peaks = np.zeros(1) # timesteps where there is a peak in the storm surge data.
	NCOUNT_saved_array_of_storm_start = np.zeros(1) # timesteps where the storm surge is deemed to have started.

	# initialize for when NCOUNT = 1
	NCOUNT_at_currentloop_storm_peak = 0
	time_in_hours_since_storm_peak = 0
	q_mo = 0

	while NCOUNT != NTSTEP:
		#print(str(NCOUNT) + ' of ' + str(NTSTEP))
		NCOUNT = NCOUNT + 1 # index of timestep
		THOUR = TSEC/3600.00 # [hr] number of hours elapsed in loop. you have to use THOUR as a way to find the nearest relevant index in the storm surge data in the loop below. 
		W = R + WBDB
		H = Hc + db + E - W*tanthb

		if np.less(H,Bc):
			print('Computation stopped because H < Bc at THOUR = ' + str(THOUR) + '. In other words, the cliff height above its toe is less than the unfrozen cliff sediment thickness, which is not physically possible.')
			raise SystemExit

		# Find surge level
		# extract the subset of the water level data available from the pre-processing step called in the input file.
		# find datetime that corresponds to current loop index. use THOUR elapsed (add timedelta) to start date of storm surge timeseries.
		start_datetime = pd.to_datetime(start_month+'-'+start_day+'-'+str(year))
		datetime_hour_present_loop = start_datetime + pd.DateOffset(hours=THOUR)
		#print(datetime_hour_present_loop)
		# find closest index of the surge array that corresponds with the datetime hour in the current loop
		index_surge_present_loop = water_level_meters.index.get_loc(datetime_hour_present_loop,method='nearest')
		#print('index S present loop: ' + str(index_surge_present_loop) + ' NCOUNT: ' + str(NCOUNT))
		S = water_level_meters[index_surge_present_loop] # index must correspond to hour number as given by thour.

		# find current timestep's value of Tr
		index_wave_period_site_present_loop = wave_period_site.index.get_loc(datetime_hour_present_loop,method='nearest')
		#print('index Tr present loop: ' + str(index_wave_period_site_present_loop) + ' NCOUNT: ' + str(NCOUNT))
		Tr = wave_period_site[index_wave_period_site_present_loop]

		# find current timestep's value of Hrms
		index_swh_present_loop = swh.index.get_loc(datetime_hour_present_loop,method='nearest')
		#print('index swh present loop: ' + str(index_swh_present_loop) + ' NCOUNT: ' + str(NCOUNT))
		Hrms = swh[index_swh_present_loop]

		# find current timestep's value of Tw
		index_sst_present_loop = sst.index.get_loc(datetime_hour_present_loop,method='nearest')
		#print('index sst present loop: ' + str(index_sst_present_loop) + ' NCOUNT: ' + str(NCOUNT))
		Tw = sst[index_sst_present_loop]

		input_variables =  np.array([S, Tr, Hrms, Tw])

		## put a check here that makes the appended values nan if the S is masked (if there is sea ice), becuase otherwhise the if statements will not run below and nothing will be appended.
		if np.isnan(input_variables).any() == False:  # since the input values are interpolated from different time resolutions, they will have a different number of nan values (due to interp).
			# find the depth of the clff toe
			dc = S + db + E - W*tanthb

			if np.greater(dc,0.0):
				Ru = beta*np.amin([dc,Hrms])
				llc = (np.amin([(Ru+dc),(H-Bc)]))/sinthc
				# Calculate cliff and beach erosion
				Ubc, fwc, fwcs, fwcr = FWROOT(X1, X2, XACC, dc, ksc, Tw, Tr, Hrms)
				hhc = HEATRA(ksc, Ubc, fwc, Tw)
				qc = (Pc*Bc + vc*(H-Bc))*hhc*llc*(Tw-Tm)/(Lc*(H-Bc))
				R = R + llc*hhc*(Tw - Tm)/(Lc*(H - Bc))*TSTEP
			else:
				R = R
				qc = 0.0

				# find the potential sediment transport from the beach, if there is no storm in the storm surge forcing
			if (S + E + db) < 0.0:
				'the storm surge is below the water level of the assumed offshore depth where sediment is transported away.  the assumed water depth is realistically too shallow in this case and beach angle needs to be increased '

			qp = alpha * Pb * (((S + E + db)**0.5)*tanthb - (2/3)*(A**(3/2)))

			if np.greater(B,0.0):
				D = D
				qb = qp
				qmelt = 0.0
				B = B + (qc - qb)/(Pb * W)*TSTEP  
				if np.less(B,0.0):
					B = 0.0
				#print(str(B) + 'b in loop')
				#qmelt_flag = np.append(qmelt_flag,1) # when 1 ..  no melt
			else:
				dbc = 0.5*(S + E + dc + db)
				Ubb, fwb, fwbs, fwbr = FWROOT(X1, X2, XACC, dbc, ksb, Tw, Tr, Hrms)
				hb = HEATRA(ksb, Ubb, fwb, Tw)
				D = D + (hb*(Tw - Tm))/Lb *TSTEP
				qmelt = vb * hb * (Tw - Tm)*W / Lb
				if np.greater_equal(qp,(qmelt + qc)):
					qb = qmelt + qc
					B = 0.0
					#qmelt_flag = np.append(qmelt_flag,2)
				else:
					qb = qp
					B = B + (qmelt + qc - qb)/(Pb*W)*TSTEP
					#qmelt_flag = np.append(qmelt_flag,3)

			E = Bb - B + D

			W_all = np.append(W_all,W) # save beach width as output array
			H_all = np.append(H_all,H) # save change in bluff height as output array
			qp_all = np.append(qp_all,qp)
			E_all = np.append(E_all,E)
			R_all = np.append(R_all,R)
			S_all = np.append(S_all,S)
			D_all = np.append(D_all,D)
			dc_all = np.append(dc_all,dc)
			B_all = np.append(B_all,B)
			qb_all = np.append(qb_all,qb)
			qmelt_all = np.append(qmelt_all,qmelt)
			qc_all = np.append(qc_all,qc)

			last_time_nan_was_not_there = NCOUNT
			switch = 'now some erosion variables have been produced'
		if np.isnan(input_variables).any() == True:
			if switch== 'now some erosion variables have been produced': # if the first timestep has sea ice, then there are no erosion variables calculated yet, thus this switch is necessary
				W_all = np.append(W_all,W)
				H_all = np.append(H_all,H)
				qp_all = np.append(qp_all,qp)
				E_all = np.append(E_all,E)
				R_all = np.append(R_all,R)
				S_all = np.append(S_all,S)
				D_all = np.append(D_all,D)
				dc_all = np.append(dc_all,dc)
				B_all = np.append(B_all,B)
				qb_all = np.append(qb_all,qb)
				qmelt_all = np.append(qmelt_all,qmelt)
				qc_all = np.append(qc_all,qc)
			else:
				W_all = np.append(W_all,np.nan)
				H_all = np.append(H_all,np.nan)
				qp_all = np.append(qp_all,np.nan)
				E_all = np.append(E_all,np.nan)
				R_all = np.append(R_all,np.nan)
				S_all = np.append(S_all,np.nan)
				D_all = np.append(D_all,np.nan)
				dc_all = np.append(dc_all,np.nan)
				B_all = np.append(B_all,np.nan)
				qb_all = np.append(qb_all,np.nan)
				qmelt_all = np.append(qmelt_all,np.nan)
				qc_all = np.append(qc_all,np.nan)

		# Proceed to next timestep
		TSEC = TSEC + TSTEP

		# end of while loop for timesteps.

	# remove placeholder that was used to initialize each array
	E_all = E_all[1:]
	R_all = R_all[1:]
	S_all = S_all[1:]
	D_all = D_all[1:]
	dc_all = dc_all[1:]
	W_all = W_all[1:]
	H_all = H_all[1:]
	B_all = B_all[1:]
	qp_all = qp_all[1:]
	qb_all = qb_all[1:]
	qmelt_all = qmelt_all[1:]
	qc_all = qc_all[1:]


	# write output data for current run
	'''
	# beach width timeseries.
	df = pd.DataFrame(W_all,columns=['W: beach width [m]'])
	df.to_csv(global_variables.path_parameters_tested_outputs_per_year+'beach_width_W.csv')
	# coarse sediment thickness on beach timeseries.
	df = pd.DataFrame(B_all,columns=['Unfrozen coarse sediment thickness on beach [m]'])
	df.to_csv(global_variables.path_parameters_tested_outputs_per_year+'unfrozen_coarse_sediment_thickness_on_beach_B.csv')
	# bluff height
	df = pd.DataFrame(H_all,columns=['Bluff height [m]'])
	df.to_csv(global_variables.path_parameters_tested_outputs_per_year+'bluff_height_timeseries_H.csv')
	# cliff toe depth
	df = pd.DataFrame(dc_all,columns=['Cliff toe depth [m]'])
	df.to_csv(global_variables.path_parameters_tested_outputs_per_year+'cliff_toe_depth_timeseries_dc.csv')
	# relative water level ('storm surge (MLLW)') tide gauge data
	df = pd.DataFrame(S_all,columns=['Relative water level (MLLW) [m]'])
	df.to_csv(global_variables.path_parameters_tested_outputs_per_year+'relative_water_level_S.csv')
	# beach melting depth
	df = pd.DataFrame(D_all,columns=['Beach thaw depth [m]'])
	df.to_csv(global_variables.path_parameters_tested_outputs_per_year+'beach_thaw_depth_D.csv')
	# cliff retreat rate
	df = pd.DataFrame(R_all,columns=['Cliff retreat [m]'])
	df.to_csv(global_variables.path_parameters_tested_outputs_per_year+'cliff_retreat_R.csv')
	# beach erosion depth
	df = pd.DataFrame(E_all,columns=['Beach erosion [m]'])
	df.to_csv(global_variables.path_parameters_tested_outputs_per_year+'beach_erosion_E.csv')
	# beach erosion depth
	df = pd.DataFrame(qp_all,columns=['Potential beach erosion (qp)'])
	df.to_csv(global_variables.path_parameters_tested_outputs_per_year+'potential_beach_sedimentflux_qp.csv')
	# beach erosion depth
	df = pd.DataFrame(qb_all,columns=['Actual sediment transport off beach (qb)'])
	df.to_csv(global_variables.path_parameters_tested_outputs_per_year+'actual_beach_transport_qb.csv')
	# beach erosion depth
	df = pd.DataFrame(qmelt_all,columns=['Sediment released from thawing frozen beach (qmelt)'])
	df.to_csv(global_variables.path_parameters_tested_outputs_per_year+'sediment_released_from_thawing_beach_qmelt.csv')
	'''
	# load the observed retreat
	observed_retreat_years = np.load(basepath + 'input_data/observed_retreat_years_' + global_variables.community_name + '.npy')
	retreat_observed = np.load(basepath + 'input_data/observed_retreat_rates_allyears_' + global_variables.community_name + '.npy')
	retreat_observed_year_selected = retreat_observed[int(np.where(observed_retreat_years==year)[0])]

	return R - retreat_observed_year_selected
