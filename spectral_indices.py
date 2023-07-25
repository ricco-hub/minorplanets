import argparse, os
import numpy as np, ephem
from numpy.lib import recfunctions
from pixell import utils, enmap, bunch, reproject, colors, coordinates, mpi
from scipy import interpolate, optimize
import glob
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import h5py
from datetime import datetime
import matplotlib.dates as mdates
import pickle as pk
import pandas as pd
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import ephem
import scipy.signal as signal
from palettable.colorbrewer.sequential import Blues_9
from scipy.stats import binned_statistic
from scipy import stats
from statistics import mean
import seaborn as sns
import sys
import requests
from scipy.stats import chi2
from operator import add
sns.set_theme(style='ticks')

def inv_var(data, variances):
  '''
    Inputs
      data, type: array of ints/floats, data to be weighted
      variances, type: array of ints/floats, variances of data
    Output
      Inverse-variance weighted average
  '''
  ave = 0
  var = 0
  for i in range(len(data)):
    ave += data[i]/variances[i]
    var += 1/variances[i]
  return ave/var, 1/var    

def inv_var_weight(n,min_bin,max_bin,errs,x_data,y_data):
  '''
    Inputs
      n, type: integer, number of weighted bins
      errs, type: float, errors for binning
      x_data, type: float, x data to be binned 
      y_data, type: float, y data to be binned 
    Outputs
      ave_var, type: float, inverse variance weighted average
      err_var, type: float, associated errors from inverse variance weighted average
      phase_bins, type: float, bins 
  '''
  interval = 1/n
  center = interval * 0.5      
  bins = np.arange(min_bin,max_bin,interval)# + center 
            
  #inverse weighting
  err_prop = []
  ave_var = []
      
  for i in np.arange(min_bin,max_bin,interval):
    try: 
      #print(i)
      err_bin = []
      res_bin = []
      err_data_sqr = []
      
      err_bin = [errs[f] for f in range(len(x_data)) if x_data[f] > i and x_data[f] <= (i+interval)]
      res_bin = [y_data[r] for r in range(len(x_data)) if x_data[r] > i and x_data[r] <= (i+interval)]
      err_data_sqr = [err_bin[e]**2 for e in range(len(err_bin))]
      ave_var_temp, i_var = inv_var(res_bin, err_data_sqr)
      new_err = i_var**0.5
          
      ave_var.append(ave_var_temp)
      err_prop.append(new_err)
    except ZeroDivisionError:      
      ave_var.append(np.nan)
      err_prop.append(np.nan)
      
  return ave_var, err_prop, bins

def spec_fit(nu,alpha,A):
  return A*(nu**(alpha))
  
def rot_phase(name,times):
  '''
    Inputs
      name, type: string, name of asteroid
      times, type: float, observation times of asteroid in MJD
    Outputs
      phi, type: float, phase numbers for rotation period of asteroid
  '''
  period = get_period(name)
  
  pos = period / 24
  phi = []
    
  #get times
  for t in range(len(times)):
    num = (times[t]) % pos
    phi.append(num/pos)
    
  return phi  
  
def orb_phase(name,times):
  '''
    Inputs
      name, type: string, name of asteroid
      obs, type: floats, observation times (MJD) of asteroid
    Output
      Theta, type: float, array of phase numbers for asteroid orbital period
      T, type: float, period of object in days
  '''
  time_temp = Time(times[0], format='mjd')
  time_iso = time_temp.iso  
  obj = Horizons(id=name, epochs={time_iso},id_type='asteroid_name')        
  el = obj.elements()     
  
  #orbital period in days
  T = el['P']
  T = float(T)  
  
  times = np.array(times)
  theta = ((times / T) % T) % 1
  theta = np.ravel(theta)
  
  return theta, T
  
def sunang_phase(name,times):
  '''
    Inputs
      name, type: string, name of asteroid
      times, type: float, times of asteroid observations in MJD
    Outputs
      alpha, type: float, solar phase angle between 0 to 1
  '''
  alpha = []
  
  for t in times:  
    #convert times
    time_temp = Time(t, format='mjd')
    time_iso = time_temp.iso
      
    obj = Horizons(id=name, epochs={time_iso},id_type='asteroid_name')    
    eph = obj.ephemerides()
      
    alpha_best = eph['alpha']
    alpha_best = float(alpha_best)    
    alpha.append(alpha_best) #solar phase angle 
  
  alpha = np.array(alpha)
   
  return (alpha/180)    

def get_period(name):
  '''
    Input
      name, type: string, capitalized name of object we want rotation period of
    Output
      Rotation period of object name
  '''
  obj = requests.get('https://ssd-api.jpl.nasa.gov/sbdb.api?sstr='+name+'&phys-par=1')
  phys_par = obj.json()['phys_par']
  
  ind = 0
  for i in phys_par:
    if i['title'] == 'rotation period':
      break
    else:
      ind += 1
  
  period = obj.json()['phys_par'][ind]['value']
  return float(period)

def light_curve(name,freq):
  '''
  Inputs:
    freq, type: integer, frequency of ACT measurement
    name, type: string, name of object (must be capitalized)
  Output:
    min(times), type: float, minimum time in data (i.e. start time)
    max(times), type: float, maximum time in data (i.e. end time)
    times, type: np.array, individual observation times (MJD)
    flux, type: np.array, normalized flux of asteroid (arbitary units)
    err, type: np.array, normalized error bar of each observation (arbitrary units)
    freq, type: np.ndarray, frequency of observation (GHz)  
  '''
  pa_dict = {'090':['pa5', 'pa6'], '150':['pa4', 'pa5', 'pa6'], '220':['pa4']}
  pas = ['pa4', 'pa5', 'pa6']
  if freq == 90:
    freq = '090'
  else:
    str(freq)

  flux = np.array([])
  times = np.array([])
  err = np.array([])
  F = np.array([])
  for pa in pas:
    try:
      with open('/scratch/r/rbond/jorlo/actxminorplanets/sigurd/lightcurves/{}_lc_{}_{}_{}.pk'.format(name, 'night', pa, freq), 'rb') as f:
          lc_dict = pk.load(f)
  
      flux = np.hstack([flux, np.array(lc_dict['flux'])])
      cur_times= np.array(lc_dict['time'])
      cur_times = Time(cur_times, format='unix')
      cur_times = cur_times.mjd
      
      times = np.hstack([times, np.array(cur_times)])
      
      err = np.hstack([err, np.array(lc_dict['err'])])
      F = np.hstack([F, np.array(lc_dict['F'])])      
    except FileNotFoundError:
      continue
  
  norm = np.mean(flux*F)
  flux = flux*F/norm
  err = err*F/norm    
  
  if freq == '090':
    freq = np.ndarray([90])
  else:
    freq = np.ndarray([int(freq)])
    
  #freq = [freq for i in range(len(times))]      
  return min(times),max(times),times,flux,err,freq  

def all_lcurves_stats(name,bin_number):
  '''
    Inputs:
      name, type: string, name of object/asteroid
      bin_number, type: integer, number of bins
    Outputs:
      f090_sums, type: float, mean of flux at 90 GHz
      f150_sums, type: float, mean of flux at 150 GHz
      f220_sums, type: float, mean of flux at 220 GHz
  '''
  min90,max90,f090_times,f090_flux,f090_err,f090_freq = light_curve(name,90)
  min150,max150,f150_times,f150_flux,f150_err,f150_freq = light_curve(name,150)
  min220,max220,f220_times,f220_flux,f220_err,f220_freq = light_curve(name,220)
  
  #find global minimum/maximum
  min_bin = min(min90,min150,min220)
  max_bin = max(max90,max150,max220)
  
  f090_sums,f090_edges,f090_index = stats.binned_statistic(f090_times,f090_flux,statistic='mean',bins=bin_number,range=(min_bin,max_bin))
  f150_sums,f150_edges,f150_index = stats.binned_statistic(f150_times,f150_flux,statistic='mean',bins=bin_number,range=(min_bin,max_bin))
  f220_sums,f220_edges,f220_index = stats.binned_statistic(f220_times,f220_flux,statistic='mean',bins=bin_number,range=(min_bin,max_bin))
  
  return f090_sums,f150_sums,f220_sums,min_bin,max_bin

def phase_curve(name,freq,pas = ['pa4', 'pa5', 'pa6']):
  '''
  '''
  pa_dict = {'090':['pa5', 'pa6'], '150':['pa4', 'pa5', 'pa6'], '220':['pa4']}
  if freq == 90:
    freq = '090'
  else:
    str(freq)

  flux = np.array([])
  times = np.array([])
  err = np.array([])
  F = np.array([])
  phi = np.array([])
  theta = np.array([])
  alpha = np.array([])
  for pa in pas:
    try:
      with open('/scratch/r/rbond/jorlo/actxminorplanets/sigurd/lightcurves/{}_lc_{}_{}_{}.pk'.format(name, 'night', pa, freq), 'rb') as f:
          lc_dict = pk.load(f)
  
      flux = np.hstack([flux, np.array(lc_dict['flux'])])
      cur_times= np.array(lc_dict['time'])
      cur_times = Time(cur_times, format='unix')
      cur_times = cur_times.mjd
      
      times = np.hstack([times, np.array(cur_times)])
      
      err = np.hstack([err, np.array(lc_dict['err'])])
      F = np.hstack([F, np.array(lc_dict['F'])])
  
      cur_phi = rot_phase(name, cur_times) #rotation phase numbers
      phi = np.hstack([phi, np.array(cur_phi)])
      cur_theta, ignore_T = orb_phase(name, cur_times) #orbital phase numbers
      theta = np.hstack([theta, np.array(cur_theta)])
      cur_alpha = sunang_phase(name, cur_times) #solar phase angle numbers
      alpha = np.hstack([alpha, np.array(cur_alpha)])
      
    except FileNotFoundError:
      continue
  
  eta = phi + theta 
  eta = eta % 1
  
  norm = np.mean(flux*F)
  flux = flux*F/norm
  err = err*F/norm  
  
  if freq == '090':
    freq = np.ndarray([90])
  else:
    freq = np.ndarray([int(freq)])
    
  #freq = [freq for i in range(len(times))]      
  return min(eta),max(eta),eta,flux,err,freq
  
def all_phase_stats(name,bin_number,pas = ['pa4', 'pa5', 'pa6']):
  '''
    Inputs:
      bin_number, type: integer, number of bins
    Outputs:
      f090_sums, type: float, mean of flux at 90 GHz
      f150_sums, type: float, mean of flux at 150 GHz
      f220_sums, type: float, mean of flux at 220 GHz
  '''
  min90,max90,f090_phase,f090_flux,f090_err,f090_freq = phase_curve(name,90)
  min150,max150,f150_phase,f150_flux,f150_err,f150_freq = phase_curve(name,150)
  min220,max220,f220_phase,f220_flux,f220_err,f220_freq = phase_curve(name,220)
  
  #find global minimum/maximum
  min_bin = min(min90,min150,min220)
  max_bin = max(max90,max150,max220)
  
  f090_var,f090_err_prop,f090_bins = inv_var_weight(bin_number,min_bin,max_bin,f090_err,f090_phase,f090_flux)
  f150_var,f150_err_prop,f150_bins = inv_var_weight(bin_number,min_bin,max_bin,f150_err,f150_phase,f150_flux)
  f220_var,f220_err_prop,f220_bins = inv_var_weight(bin_number,min_bin,max_bin,f220_err,f220_phase,f220_flux)
  
  return f090_var,f090_err_prop,f150_var,f150_err_prop,f220_var,f220_err_prop,min_bin,max_bin

def spec_index(name,bin_number,x_axis):
  '''
    Inputs:
      name, type: string, name of object (capitalized)
      bin_number, type: integer, number of bins 
      x_axis, type: string, plot spectral index against 'time' or 'phase'
    Outputs: 
      Plot of spectral index vs. bin number
  '''
  f090_var,f090_err,f150_var,f150_err,f220_var,f220_err,min_bin_phase,max_bin_phase = all_phase_stats(name,bin_number)
  f090_sums,f150_sums,f220_sums,min_bin_time,max_bin_time = all_lcurves_stats(name,bin_number)
  freq = [90,150,220]
  indices = []
  for i in range(bin_number):
    try:
      flux = [f090_var[i],f150_var[i],f220_var[i]]
      err = [f090_err[i],f150_err[i],f220_err[i]]
        
      #fit
      params, params_covariance = optimize.curve_fit(spec_fit, freq, flux, sigma=err, maxfev=10000)
      perr = np.sqrt(np.diag(params_covariance))
      alpha = params[0]
      indices.append(alpha)
    except ValueError:
      indices.append(np.nan)
      continue  
  
  if x_axis == 'time':
    interval = (max_bin_time - min_bin_time)/bin_number
    bins = np.arange(min_bin_time,max_bin_time,interval) 
    
    plt.xlabel('Time (MJD)')
  elif x_axis == 'phase':
    interval = (max_bin_phase - min_bin_phase)/bin_number
    bins = np.arange(min_bin_phase,max_bin_phase,interval)
    
    plt.xlabel('Phase')    
  else:
    print('Currently not supported') 
  plt.scatter(bins,indices)
  plt.ylabel('Spectral Index')
  plt.title('Spectral Index of {}'.format(name))
  plt.show()  
  
  return bins, indices
   
def spec_flux(name,bins,freq):
  '''
    Inputs
    Outputs
      Combined plot of spectral index and flux against time
  '''
  spec_bins, indices = spec_index(name,bins,'phase')
  min_eta,max_eta,eta,flux,err,ignore_freq = phase_curve(name,freq)
  ave_var, err_prop, phase_bins = inv_var_weight(bins,min_eta,max_eta,err,eta,flux)
  
  fig, ax = plt.subplots()
  ax.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='.', label='Bin Flux', zorder=1, capsize=5, alpha=1)
  ax.set_xlabel('Phase Number')
  ax.set_ylabel('Flux (Arbitrary)')
  ax2 = ax.twinx()
  ax2.scatter(spec_bins,indices,label='Spectral Index',color='g')
  ax2.set_ylabel('Spectral Index')
  plt.title('Flux and Spectral Index of {name} at {freq}'.format(name=name,freq=freq))
  flux_plot, flux_labels = ax.get_legend_handles_labels()
  spec_plot, spec_labels = ax2.get_legend_handles_labels()
  ax2.legend(flux_plot + spec_plot, flux_labels + spec_labels, loc='best')
  plt.show()
  
def spec_index_arr(name,bin_number,pas):
  '''
    Inputs:
      name, type: string, name of object (capitalized)
      bin_number, type: integer, number of bins 
      x_axis, type: string, plot spectral index against 'time' or 'phase'
      pas, type: string array, ACT array
    Outputs: 
      Plot of spectral index over different arrays vs. bin number
  '''
  f090_var,f090_err,f150_var,f150_err,f220_var,f220_err,min_bin_phase,max_bin_phase = all_phase_stats(name,bin_number,pas)
  #f090_sums,f150_sums,f220_sums,min_bin_time,max_bin_time = all_lcurves_stats(name,bin_number)
  indices = []
  if pas == ['pa4']:
    freq = [150,220]
    for i in range(bin_number):
      try:
        flux = [f150_var[i],f220_var[i]]
        err = [f150_err[i],f220_err[i]]
        
        #fit
        params, params_covariance = optimize.curve_fit(spec_fit, freq, flux, sigma=err, maxfev=100000)
        perr = np.sqrt(np.diag(params_covariance))
        alpha = params[0]
        indices.append(alpha)
      except ValueError:
        indices.append(np.nan)
        continue  
  else:
    freq = [90,150]
    for i in range(bin_number):
      try:
        flux = [f090_var[i],f150_var[i]]
        err = [f090_err[i],f150_err[i]]
        
        #fit
        params, params_covariance = optimize.curve_fit(spec_fit, freq, flux, sigma=err, maxfev=100000)
        perr = np.sqrt(np.diag(params_covariance))
        alpha = params[0]
        indices.append(alpha)
      except ValueError:
        indices.append(np.nan)
        continue  
          
  #if x_axis == 'time':
  #  interval = (max_bin_time - min_bin_time)/bin_number
  #  bins = np.arange(min_bin_time,max_bin_time,interval) 
    
  #  plt.xlabel('Time (MJD)')
  interval = (max_bin_phase - min_bin_phase)/bin_number
  bins = np.arange(min_bin_phase,max_bin_phase,interval)
    
  #plt.xlabel('Phase')     
  #plt.scatter(bins,indices)
  #plt.ylabel('Spectral Index')
  #plt.title('Spectral Index of {}'.format(name))
  #plt.show()  
  
  return bins, indices  
 
 
def plot_spec_index_arr(name,bin_number):
  bins_pa4, indices_pa4 = spec_index_arr(name,bin_number,pas=['pa4'])
  bins_pa5, indices_pa5 = spec_index_arr(name,bin_number,pas=['pa5'])
  bins_pa6, indices_pa6 = spec_index_arr(name,bin_number,pas=['pa6'])
  
  plt.xlabel('Phase')     
  plt.scatter(bins_pa4,indices_pa4,label='pa4')
  plt.scatter(bins_pa5,indices_pa5,label='pa5')
  plt.scatter(bins_pa6,indices_pa6,label='pa6')
  plt.ylabel('Spectral Index')
  plt.title('Spectral Index of {} Across Arrays'.format(name))
  plt.legend(loc='best')  
  plt.show()    
##############################RUN BELOW#################################################################################################################
#light_curve(150)
#all_lcurves_stats(50)
#tuples()
#spec_index('Hebe',20,'phase')
#spec_flux('Hebe',20,220)
plot_spec_index_arr('Hebe',20)