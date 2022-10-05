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

def get_desig(id_num):
  '''
  Input:
    id_num, type: integer, number between 0 - 799
  Output:
    desig, type: integer, designation number of object
    name, type: string, name of object
    semimajor, type: float, semimajor axis of object
    
    Gets semimajor axis of object
  '''
  with open('/home/r/rbond/ricco/minorplanets/asteroids.pk', 'rb') as f:
    df = pk.load(f)
    name = df['name'][id_num]
    desig = df['designation'][id_num]
    semimajor = df['semimajor'][id_num]
  return desig, name, semimajor
  
def get_index(name):
  '''
    Inputs:
    name, type: string, name of object
    
    Output:
    desig, type: integer, index of object in asteroids.pk file
  '''
  
  with open('/home/r/rbond/ricco/minorplanets/asteroids.pk', 'rb') as f:
    df = pk.load(f)    
    idx = np.where((df['name'] == name))[0]    
    desig = df['designation'][idx]
  
  string = desig.to_string()
  
  num_string = ''
  
  for s in string:
    if s == ' ':
      break
    else:
      num_string += s
  
  indx = int(num_string)
  return indx
  
def one_lcurve(name, arr, freq, directory = None, show = False):
  '''
    Inputs:
      name, type: string, name of object we want
      arr, type: string, ACT array we want
      freq, type: string, frequency band
      directory, type: string, optionally save plots in specified directory
      show, type: boolean, optionally display plot right after calling one_lcurve()
    
    Output:
      Single light curve for desired object
  '''
  
  index = get_index(name)
  
  #get semimajor axis and name
  ignore_desig, name, semimajor_sun = get_desig(index)
  ignore_desig, ignore_name, semimajor_earth = get_desig(index)
  
  #Jack's maps
  path = "/scratch/r/rbond/jorlo/actxminorplanets/sigurd/asteroids/" + name
  
  #get rho and kappa files
  rho_files = glob.glob(path + "/*" + arr + "_" + freq + "_" + "rho.fits")
  kap_files = [utils.replace(r, "rho.fits", "kappa.fits") for r in rho_files]
  
  if len(rho_files) != 0:
    #find time
    str_times = []
    t_start = len(path) + len(name) + 9
    t_end = t_start + 10
    for time in rho_files:
      str_times.append(time[t_start:t_end])
    int_times = [int(t) for t in str_times]
    
    #get geocentric dist
    eph = np.load("/gpfs/fs0/project/r/rbond/sigurdkn/actpol/ephemerides/objects/" + name + ".npy").view(np.recarray)
    orbit = interpolate.interp1d(eph.ctime, [utils.unwind(eph.ra*utils.degree), eph.dec*utils.degree, eph.r, eph.rsun, eph.ang*utils.arcsec], kind=3)
    
    flux_data = []
    err_data = []
    times_data = []
    
    Fs = []
    for count, t in enumerate(int_times):
      #get distances
      pos = orbit(t)
      
      d_sun_0, d_earth_0 = semimajor_sun, semimajor_earth
      
      ignore_ra, ignore_dec, delta_earth, delta_sun, ignore_ang = pos
      
      #F weighting
      F = (d_sun_0)**2 * (d_earth_0)**2 / ((delta_earth)**2*(delta_sun)**2) #* 491
      Fs.append(F)
      
      #open files
      hdu_rho = fits.open(rho_files[count])
      hdu_kap = fits.open(kap_files[count])
      
      #get data
      data_rho = hdu_rho[0].data
      data_kap = hdu_kap[0].data
      
      #get flux, error, and time
      flux = data_rho / data_kap
      good_flux = flux[0, 40, 40]
      flux_data.append(good_flux)
      
      err = np.abs(data_kap)**(-0.5)
      err_data.append(err[0,40,40])
      
      times_data.append(t)
      
    mjd_date = utils.ctime2mjd(times_data)
    
    plt.errorbar(mjd_date, flux_data, yerr=err_data, fmt='o', capsize=4, label='Flux')
    plt.scatter(mjd_date, Fs, label='F weighting', c='r')
    plt.xlabel("Time (MJD)")
    plt.ylabel("Flux (mJy)")
    plt.legend(loc = 'best')
    plt.title("Light curve of {name} on {arr} at {freq}".format(name=name, arr=arr, freq=freq))
    
    if show is not False:
      plt.show()
      
    if directory is not None:
      plt.savefig(directory + "{name}_light_curve_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))
      
  else:
      print("No hits")

def lcurves(arr, freq, directory = None, show = False):
  '''
    Inputs:
    arr, type: arr, ACT array
    freq, type: freq, frequency we want
    directory, type: string, optionally save file in directory
    show, type: boolean, if true, display light curve after calling lcurve
  
    Outputs:
      figure, creates light curve for object based on hits after running get_maps
      also plots F weighting
  '''

  for i in range(10):
    #get semimajor axis and name
    ignore_desig, name, semimajor_sun = get_desig(i)
    ignore_desig, ignore_name, semimajor_earth = get_desig(i)    
    
    #Jack's maps
    path = "/scratch/r/rbond/jorlo/actxminorplanets/sigurd/asteroids/" + name 
    
    #get rho and kappa files
    rho_files = glob.glob(path + "/*" + arr + "_" + freq + "_" + "rho.fits")
    kap_files = [utils.replace(r, "rho.fits", "kappa.fits") for r in rho_files] 
    
    if len(rho_files) != 0:
      #find time
      str_times = []
      t_start = len(path) + len(name) + 9
      t_end = t_start + 10
      for time in rho_files:
        str_times.append(time[t_start:t_end]) 
      int_times = [int(t) for t in str_times]
      
      #get geocentric dist
      eph = np.load("/gpfs/fs0/project/r/rbond/sigurdkn/actpol/ephemerides/objects/" + name + ".npy").view(np.recarray)
      orbit = interpolate.interp1d(eph.ctime, [utils.unwind(eph.ra*utils.degree), eph.dec*utils.degree, eph.r, eph.rsun, eph.ang*utils.arcsec], kind=3)
      
      flux_data = []
      err_data = []
      times_data = []
      Fs = []
      for count, t in enumerate(int_times):
        #get distances
        pos = orbit(t)
        
        d_sun_0, d_earth_0 = semimajor_sun, semimajor_earth 
        
        ignore_ra, ignore_dec, delta_earth, delta_sun, ignore_ang = pos      
        
        #F weighting
        F = (d_sun_0)**2 * (d_earth_0)**2 / ((delta_earth)**2*(delta_sun)**2) #* 491.92334
        Fs.append(F)    
      
        #open files
        hdu_rho = fits.open(rho_files[count])
        hdu_kap = fits.open(kap_files[count])
        
        #get data
        data_rho = hdu_rho[0].data
        data_kap = hdu_kap[0].data
        
        #get flux, error, and time
        flux = data_rho / data_kap
        good_flux = flux[0, 40, 40]
        #print(good_flux)
        flux_data.append(good_flux)
        
        err = np.abs(data_kap)**(-0.5)
        err_data.append(err[0,40,40]) 
        
        times_data.append(t)
      
      mjd_date = utils.ctime2mjd(times_data)
      
      plt.clf()
      plt.errorbar(mjd_date, flux_data, yerr=err_data, fmt='o', capsize=4, label='Flux')
      plt.scatter(mjd_date, Fs, label='F weighting', c='r')
      plt.xlabel("Time (MJD)")
      plt.ylabel("Flux (mJy)")
      plt.legend(loc='best')      
      plt.title("Light curve of {name} on {arr} at {freq}".format(name=name, arr=arr, freq=freq))
      
      if show is not False:
        plt.show()
        
      if directory is not None:
        plt.savefig(directory + "{name}_light_curve_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))
    
    else:
      print("No hits")     


lcurves("pa5", "f150", show=True)
#one_lcurve("Eros", "pa5", "f150", show=True)