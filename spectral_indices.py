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

def inv_var_weight(n,errs,x_data,y_data):
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
  bins = np.arange(min(x_data),max(x_data),interval) + center 
            
  #inverse weighting
  err_prop = []
  ave_var = []
      
  for i in np.arange(min(x_data),max(x_data),interval):
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

def spec_index(name,bin_number):
  '''
    Inputs:
      name, type: string, name of object (capitalized)
      bin_number, type: integer, number of bins 
    Outputs: 
      Plot of spectral index vs. bin number
  '''
  f090_sums,f150_sums,f220_sums,min_bin,max_bin = all_lcurves_stats(name,bin_number)
  freq = [90,150,220]
  indices = []
  interval = (max_bin - min_bin)/bin_number
  bins = np.arange(min_bin,max_bin,interval) 
  #bins = []
  for i in range(len(f090_sums)):
    try:
      flux = [f090_sums[i],f150_sums[i],f220_sums[i]]
        
      #fit
      params, params_covariance = optimize.curve_fit(spec_fit, freq, flux,maxfev=10000)
      perr = np.sqrt(np.diag(params_covariance))
      alpha = params[0]
      indices.append(alpha)
      #bins.append(i)
    except ValueError:
      indices.append(np.nan)
      continue
            
  plt.scatter(bins,indices)
  plt.xlabel('Time (MJD)')
  #plt.xlabel('Bin Number')
  plt.ylabel('Spectral Index')
  plt.title('Spectral Index of {}'.format(name))
  plt.show()  
  
##############################RUN BELOW#################################################################################################################
#light_curve(150)
#all_lcurves_stats(50)
#tuples()
spec_index('Vesta',50)