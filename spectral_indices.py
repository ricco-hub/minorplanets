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
      print(i)
      err_bin = []
      res_bin = []
      err_data_sqr = []
      
      err_bin = [errs[f] for f in range(len(x_data)) if x_data[f] > i and x_data[f] <= (i+interval)]
      res_bin = [y_data[r] for r in range(len(x_data)) if x_data[r] > i and x_data[r] <= (i+interval)]
      #print('res_bin', res_bin)
      err_data_sqr = [err_bin[e]**2 for e in range(len(err_bin))]
      #print('err_data_sqr', err_data_sqr)
      ave_var_temp, i_var = inv_var(res_bin, err_data_sqr)
      new_err = i_var**0.5
      #print('new_err', new_err)
          
      ave_var.append(ave_var_temp)
      err_prop.append(new_err)
    except ZeroDivisionError:
      print('Zero Division Error')      
      ave_var.append(np.nan)
      err_prop.append(np.nan)
      
  return ave_var, err_prop, bins

def spec_fit(nu,alpha,A):
  return A*(nu**(-1*alpha))

def light_curve(pas,freq):
  '''
  Inputs:
    pas, type: array, arrays for each frequency
    freq, type: string, frequency of measurement
    
  Output:
    Light curve using Jack's data
    
  '''
  pa_dict = {'090':['pa5', 'pa6'], '150':['pa4', 'pa5', 'pa6'], '220':['pa4']}
  name = 'Vesta'
  #pas = [ 'pa5', 'pa4']
  #freq = '150'

  flux = np.array([])
  times = np.array([])
  err = np.array([])
  F = np.array([])
  phi = np.array([])
  theta = np.array([])
  alpha = np.array([])
  for pa in pas:
      with open('/scratch/r/rbond/jorlo/actxminorplanets/sigurd/lightcurves/{}_lc_{}_{}_{}.pk'.format(name, 'night', pa, freq), 'rb') as f:
          lc_dict = pk.load(f)
  
      flux = np.hstack([flux, np.array(lc_dict['flux'])])
      cur_times= np.array(lc_dict['time'])
      cur_times = Time(cur_times, format='unix')
      cur_times = cur_times.mjd
      
      times = np.hstack([times, np.array(cur_times)])
      
      err = np.hstack([err, np.array(lc_dict['err'])])
      F = np.hstack([F, np.array(lc_dict['F'])])
  
  norm = np.mean(flux*F)
  flux = flux*F/norm
  err = err*F/norm  
  
  #fit
  times,flux,err = zip(*sorted(zip(times,flux,err)))
  params, params_covariance = optimize.curve_fit(spec_fit, times, flux, sigma=err,maxfev=10000)
  perr = np.sqrt(np.diag(params_covariance))  
  x = 150
  y_fit = spec_fit(x, params[0], params[1])
  
  #binning
  ave_var, err_prop, bins = inv_var_weight(50,err,times,flux)  
  
  #plot
  #plt.errorbar(times, flux, yerr=err, fmt='o', label='Flux', zorder=0,alpha=0.3)
  #plt.errorbar(bins, ave_var, yerr=err_prop, fmt='.', label='Bin Flux at {}'.format(freq), zorder=1, capsize=5, alpha=1)
  plt.plot(times,y_fit, label='Spectral Indices', alpha=1)   
  plt.tick_params(direction='in')
  plt.xlabel("Time (MJD)")
  #plt.ylabel("Normalized Flux")
  plt.ylabel("Alpha")
  #plt.title('Phase Curve of {name} at {freq}'.format(name=name,freq=freq))
  plt.legend(loc='best')       
  plt.show()
  #plt.savefig("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/{name}_light_curve_all_{freq}.pdf".format(name=name, freq=freq))

def all_lcurves():
  '''
    Inputs:
    Outputs: 
      Inverse-variance weighted average 
  '''
  
  light_curve(['pa5','pa6'],'090')
  light_curve(['pa5','pa4'],'150')
  light_curve(['pa4'],'220')
  plt.show()

##############################RUN BELOW#################################################################################################################
light_curve([ 'pa5', 'pa4'],'150')
#all_lcurves()

