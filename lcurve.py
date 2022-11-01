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
  
#get theory fluxes
def get_theory(name, freq):
  with open("/home/r/rbond/ricco/minorplanets/theory_flux_dict.pk", 'rb') as f:
    '''
    Format: [f090, f150, f220]
    '''
    theory_flux_dict = pk.load(f)
    
  try:
    if freq == "f090":
      return theory_flux_dict[name]["run5a"][0]
    elif freq == "f150":
      return theory_flux_dict[name]["run5a"][1]
    elif freq == "f220":
      return theory_flux_dict[name]["run5a"][2]
    else:
      print("Not a valid frequency, please try again")
  
  except KeyError:
    print("Object " + name + " not currently in flux theory file")    

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
  
  #get ref flux
  ref_flux = get_theory(name, freq)
  
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
      try:
        F = (d_sun_0)**2 * (d_earth_0)**2 / ((delta_earth)**2*(delta_sun)**2) * ref_flux
        Fs.append(F)
      except:
        print("Unable to generate light curve")
        break
      
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
    
    plt.errorbar(mjd_date, flux_data, yerr=err_data, fmt='o', capsize=4, label='Flux', zorder=0)
    plt.scatter(mjd_date, Fs, label='F weighting', c='r', zorder=1)
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
  Objects = []
  Array = []
  Frequency = []
  Fluxes = []
  Times = []
  Errors = []
  Weights = []
  Theory = []

  for i in range(100):    
    #get semimajor axis and name
    ignore_desig, name, semimajor_sun = get_desig(i)
    ignore_desig, ignore_name, semimajor_earth = get_desig(i)
    Objects.append(name)
    Array.append(arr)
    Frequency.append(freq)
    
    #get ref flux
    ref_flux = get_theory(name, freq)
    Theory.append(ref_flux)    
    
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
        try:
          F = (d_sun_0)**2 * (d_earth_0)**2 / ((delta_earth)**2*(delta_sun)**2) * ref_flux
          Fs.append(F)
        except TypeError:
          print("Unable to generate light curve")
          break    
      
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
      plt.errorbar(mjd_date, flux_data, yerr=err_data, fmt='o', capsize=4, label='Flux', zorder=0)
      plt.scatter(mjd_date, Fs, label='F weighting', c='r', zorder=1)
      plt.xlabel("Time (MJD)")
      plt.ylabel("Flux (mJy)")
      plt.legend(loc='best')      
      plt.title("Light curve of {name} on {arr} at {freq}".format(name=name, arr=arr, freq=freq))
      
      Fluxes.append(flux_data)
      Errors.append(err_data)
      Times.append(mjd_date)
      Weights.append(Fs)
      
      if show is not False:
        plt.show()
        
      if directory is not None:
        plt.savefig(directory + "{name}_light_curve_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))
    
    else:
      print("No hits")
      
def all_lcurves(show = False, directory = None):
  '''
  Inputs:
    directory, type: string, optionally save file in directory
    show, type: boolean, if true, display light curve after calling lcurve
  
  Outputs:
    figure, creates light curves for all frequencies object based on hits after running get_maps
    also plots F weighting
  '''
  
  #get f090
  infile_f090 = open('/scratch/r/rbond/ricco/minorplanets/lcurve_data_f090.pk', 'rb')
  dict_f090 = pk.load(infile_f090)
  infile_f090.close()
  
  #get f150
  infile_f150 = open('/scratch/r/rbond/ricco/minorplanets/lcurve_data_f150.pk', 'rb')
  dict_f150 = pk.load(infile_f150)
  infile_f150.close()
    
  #get f220
  infile_f220 = open('/scratch/r/rbond/ricco/minorplanets/lcurve_data_f220.pk', 'rb')
  dict_f220 = pk.load(infile_f220)
  infile_f220.close()
  
  names = dict_f090['Name']  
      
  for i in range(100):
    #get times, fluxs, Fs, errors
    times_f090 = dict_f090['Time'][i]
    flux_f090 = dict_f090['Flux'][i]
    fWeights_f090 = dict_f090['F'][i]
    error_f090 = dict_f090['Error'][i]
  
    times_f150 = dict_f150['Time'][i]
    flux_f150 = dict_f150['Flux'][i]
    fWeights_f150 = dict_f150['F'][i]
    error_f150 = dict_f150['Error'][i]
    
    times_f220 = dict_f220['Time'][i]
    flux_f220 = dict_f220['Flux'][i]
    fWeights_f220 = dict_f220['F'][i]
    error_f220 = dict_f220['Error'][i]  
    
    #plot together
    plt.clf()
    plt.errorbar(times_f090, flux_f090, yerr=error_f090, fmt='o', capsize=4, label='Flux at f090', zorder=0)
    plt.scatter(times_f090, fWeights_f090, c='r', marker='^', label='f090 weighting', zorder=1)
    
    plt.errorbar(times_f150, flux_f150, yerr=error_f150, fmt='o', capsize=4, label='Flux at f150', zorder=0)
    plt.scatter(times_f150, fWeights_f150, c='b', marker='^', label='f150 weighting', zorder=1)
    
    plt.errorbar(times_f220, flux_f220, yerr=error_f220, fmt='o', capsize=4, label='Flux at f220', zorder=0)
    plt.scatter(times_f220, fWeights_f220, c='g', marker='^', label='f220 weighting', zorder=1)
        
    plt.xlabel("Time (MJD)")
    plt.ylabel("Flux (mJy)")
    plt.legend(loc='best')      
    plt.title("Light curves of {name}".format(name=names[i]))
    
    if show is not False:
      plt.show()  
      
    if directory is not None:
      plt.savefig(directory + "{name}_light_curves.pdf".format(name=names[i]))    
  
def ratios(show = False, directory = None):
  '''
  Inputs:
    directory, type: string, optionally save file in directory
    show, type: boolean, if true, display light curve after calling lcurve
  
  Outputs:
    figure, creates flux/F weight ratio scatter plot for all frequencies object based on hits after running get_maps    
  '''
  
  #get f090
  infile_f090 = open('/scratch/r/rbond/ricco/minorplanets/lcurve_data_f090.pk', 'rb')
  dict_f090 = pk.load(infile_f090)
  infile_f090.close()
  
  #get f150
  infile_f150 = open('/scratch/r/rbond/ricco/minorplanets/lcurve_data_f150.pk', 'rb')
  dict_f150 = pk.load(infile_f150)
  infile_f150.close()
    
  #get f220
  infile_f220 = open('/scratch/r/rbond/ricco/minorplanets/lcurve_data_f220.pk', 'rb')
  dict_f220 = pk.load(infile_f220)
  infile_f220.close()
  
  names = dict_f090['Name']  
      
  for i in range(100):
    #get times, fluxs, Fs, errors
    times_f090 = dict_f090['Time'][i]
    flux_f090 = dict_f090['Flux'][i]
    fWeights_f090 = dict_f090['F'][i]
    error_f090 = dict_f090['Error'][i]
  
    times_f150 = dict_f150['Time'][i]
    flux_f150 = dict_f150['Flux'][i]
    fWeights_f150 = dict_f150['F'][i]
    error_f150 = dict_f150['Error'][i]
    
    times_f220 = dict_f220['Time'][i]
    flux_f220 = dict_f220['Flux'][i]
    fWeights_f220 = dict_f220['F'][i]
    error_f220 = dict_f220['Error'][i]  
    
    #get ratios
    ratio_f090 = []
    ratio_f150 = []    
    ratio_f220 = []    
    for n in range(len(flux_f090)):
      r_f090 = abs(flux_f090[n] / fWeights_f090[n])
      ratio_f090.append(r_f090)      
    for o in range(len(flux_f150)):
      r_f150 = abs(flux_f150[o] / fWeights_f150[o])
      ratio_f150.append(r_f150)      
    for t in range(len(flux_f220)):
      r_f220 = abs(flux_f220[t] / fWeights_f220[t])
      ratio_f220.append(r_f220)     
    
    #plot together
    plt.clf()
    #f090 ratios
    plt.scatter(times_f090, ratio_f090, label='f090 ratio')
    
    #f150 ratios
    plt.scatter(times_f150, ratio_f150, label='f150 ratio')
    
    #f220 ratios
    plt.scatter(times_f220, ratio_f220, label='f220 ratio')
        
    plt.xlabel("Time (MJD)")
    plt.ylabel("Flux / F Weighting")
    plt.legend(loc='best')      
    plt.title("Ratios of {name}".format(name=names[i]))
    
    if show is not False:
      plt.show()  
      
    if directory is not None:
      plt.savefig(directory + "{name}_light_curves.pdf".format(name=names[i]))    

def test_weights(name, arr, freq, show=False, directory=None):
  '''
  Inputs:
    name, type: string, name of object
    arr, type: string, ACT array
    freq, type: string, frequency we want
    directory, type: string, optionally save file in directory
    show, type: boolean, if true, display light curve after calling lcurve    
    
  Output:
    plot of F weighting using semimajor axis from astroquery along with Jack's theory fluxes
  '''
  index = get_index(name)
  
  #get semimajor axis and name
  ignore_desig, name, semimajor_sun = get_desig(index)
  ignore_desig, ignore_name, semimajor_earth = get_desig(index)
  
  #get ref flux
  ref_flux = get_theory(name, freq)  
  
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
    start = Time(min(int_times), format='unix')
    end = Time(max(int_times), format='unix')
    start_iso = start.iso
    end_iso = end.iso
    
    #get astroquery data  
    obj_earth = Horizons(id=name, location='W99', epochs={'start':start_iso, 'stop':end_iso, 'step':'1d'},id_type='asteroid_name')
    obj_sun = Horizons(id=name, location='500@10', epochs={'start':start_iso, 'stop':end_iso, 'step':'1d'},id_type='asteroid_name')
    
    el = obj_sun.elements()
    eph = obj_earth.ephemerides()
    
    time = eph['datetime_jd']
    mjd_times = utils.jd2mjd(time)
    
    d_sun_0_test = el['a']
    d_earth_0_test = el['a']
    
    delta_earth_test = eph['delta']
    delta_sun_test = eph['r']
    
    test_F = (d_sun_0_test)**2 * (d_earth_0_test)**2 / ((delta_earth_test)**2*(delta_sun_test)**2) * ref_flux

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
      try:
        F = (d_sun_0)**2 * (d_earth_0)**2 / ((delta_earth)**2*(delta_sun)**2) * ref_flux
        Fs.append(F)
      except TypeError:
        print("Unable to generate light curve")
        break    
      
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
    plt.errorbar(mjd_date, flux_data, yerr=err_data, fmt='o', capsize=4, label='Flux', zorder=0)
    plt.scatter(mjd_date, Fs, label='F weighting', c='r', zorder=1)
    plt.plot(mjd_times, test_F, label='Test F weighting', c='k')
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

def phases(name, period, freq, show = False, directory = None):
  '''
  Inputs:
    name, type: string, name of object we want phase curve for
    period, type: float, rotational period of object in hours
    freq, type: string, frequency we want
    directory, type: string, optionally save file in directory
    show, type: boolean, if true, display light curve after calling lcurve
  Output:
    Phase curve of object given a period
  '''
  
  if freq == "f090":
    infile_f090 = open('/scratch/r/rbond/ricco/minorplanets/lcurve_data_f090.pk', 'rb')
    dict_f090 = pk.load(infile_f090)
    infile_f090.close()

    obj =  dict_f090['Name'][get_index(name)]
    
    #get times
    pos = period / 24
    phases = [] 
    for t in range(len(dict_f090['Time'][2])):
      num = (dict_f090['Time'][2][t]) % pos
      phases.append(num/pos)
      
    #get data
    fluxs_f090 = dict_f090['Flux'][2]
    error_f090 = dict_f090['Error'][2]
    fWeights = dict_f090['F'][2]
    
    res = [(fluxs_f090[f]- fWeights[f]) for f in range(len(fluxs_f090))]
    
    plt.errorbar(phases, res, yerr=error_f090, fmt='o', capsize=4, label='Flux at f090', zorder=0)    
    
  elif freq == "f150":
    #get f150
    infile_f150 = open('/scratch/r/rbond/ricco/minorplanets/lcurve_data_f150.pk', 'rb')
    dict_f150 = pk.load(infile_f150)
    infile_f150.close()  
  
    obj =  dict_f150['Name'][get_index(name)]
    
    #get times
    pos = period / 24
    phases = [] 
    for t in range(len(dict_f150['Time'][2])):
      num = (dict_f150['Time'][2][t]) % pos
      phases.append(num/pos)
      
    #get data
    fluxs_f150 = dict_f150['Flux'][2]
    error_f150 = dict_f150['Error'][2]
    fWeights = dict_f150['F'][2]
    
    res = [(fluxs_f150[f]- fWeights[f]) for f in range(len(fluxs_f150))]
    
    plt.errorbar(phases, res, yerr=error_f150, fmt='o', capsize=4, label='Flux at f150', zorder=0)    
    
  elif freq == "f220":
    #get f220
    infile_f220 = open('/scratch/r/rbond/ricco/minorplanets/lcurve_data_f220.pk', 'rb')
    dict_f220 = pk.load(infile_f220)
    infile_f220.close()  

    obj =  dict_f220['Name'][get_index(name)]
    
    #get times
    pos = period / 24
    phases = [] 
    for t in range(len(dict_f220['Time'][2])):
      num = (dict_f220['Time'][2][t]) % pos
      phases.append(num/pos)
      
    #get data
    fluxs_f220 = dict_f220['Flux'][2]
    error_f220 = dict_f220['Error'][2]
    fWeights = dict_f220['F'][2]
    
    res = [(fluxs_f220[f]- fWeights[f]) for f in range(len(fluxs_f220))]
    
    plt.errorbar(phases, res, yerr=error_f220, fmt='o', capsize=4, label='Flux at f220', zorder=0)

  else:
    print("No data for this frequency. Please select from 'f090, f150, or f220'") 

  
  #plt.errorbar(phases, fluxs_f150, yerr=error_f150, fmt='o', capsize=4, label='Flux at f150', zorder=0)

  #plt.ylim([-100,100])
  plt.xlabel("Phase Number")
  plt.ylabel("Flux Residual (mJy)")
  plt.title("Phase Curve of {name} at {freq}".format(name=obj,freq=freq))
  plt.legend(loc='best')
  
  if show is not False:
    plt.show()
        
  if directory is not None:
    plt.savefig(directory + "{name}_phase_curve_{freq}.pdf".format(name=obj, freq=freq))  
  
def all_arrays(name, freq, directory = None, show = False):
  '''
    Inputs:
      name, type: string, name of object we want
      freq, type: string, frequency band
      directory, type: string, optionally save plots in specified directory
      show, type: boolean, optionally display plot right after calling one_lcurve()
    
    Output:
      Light curves for desired object on all arrays for frequency freq
  '''
  
  #index = get_index(name)
  
  #get semimajor axis and name
  #ignore_desig, name, semimajor_sun = get_desig(index)
  #ignore_desig, ignore_name, semimajor_earth = get_desig(index)
  
  #get ref flux
  #ref_flux = get_theory(name, freq)
  
  #Jack's maps
  path = "/scratch/r/rbond/jorlo/actxminorplanets/sigurd/asteroids/" + name
  
  arrays = ["pa4","pa5","pa6"]  
  flux_data = [[],[],[]]
  err_data = [[],[],[]]
  times_data = [[],[],[]]
  for i, arr in enumerate(arrays):
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
        
      Fs = []
      for count, t in enumerate(int_times):
        #get distances
        pos = orbit(t)
          
        #d_sun_0, d_earth_0 = semimajor_sun, semimajor_earth
          
        #ignore_ra, ignore_dec, delta_earth, delta_sun, ignore_ang = pos
          
        #F weighting
        try:
          pass
          #F = (d_sun_0)**2 * (d_earth_0)**2 / ((delta_earth)**2*(delta_sun)**2) #* ref_flux
          #Fs.append(F)
        except:
          print("Unable to generate light curve")
          break
          
        #open files
        hdu_rho = fits.open(rho_files[count])
        hdu_kap = fits.open(kap_files[count])
          
        #get data
        data_rho = hdu_rho[0].data
        data_kap = hdu_kap[0].data
          
        #get flux, error, and time
        flux = data_rho / data_kap
        good_flux = flux[0, 40, 40]
        flux_data[i].append(good_flux)
          
        err = np.abs(data_kap)**(-0.5)
        err_data[i].append(err[0,40,40])
          
        times_data[i].append(t)
      
  mjd_date_pa4 = utils.ctime2mjd(times_data[0])
  mjd_date_pa5 = utils.ctime2mjd(times_data[1])
  mjd_date_pa6 = utils.ctime2mjd(times_data[2])
        
  plt.errorbar(mjd_date_pa4, flux_data[0], yerr=err_data[0], fmt='o', capsize=4, label='pa4', zorder=0)
  plt.errorbar(mjd_date_pa5, flux_data[1], yerr=err_data[1], fmt='o', capsize=4, label='pa5', zorder=0)
  plt.errorbar(mjd_date_pa6, flux_data[2], yerr=err_data[2], fmt='o', capsize=4, label='pa6', zorder=0)
  plt.ylim([0,1000])
  #plt.scatter(mjd_date, Fs, label='F weighting', c='r', zorder=1)
  plt.xlabel("Time (MJD)")
  plt.ylabel("Flux (mJy)")
  plt.legend(loc = 'best')
  plt.title("Light curves of {name} at {freq}".format(name=name, freq=freq))
  plt.show()


########################################RUN STUFF BELOW##########################################################
#all_lcurves(show = True)
#ratios(show = True)
#lcurves("pa5", "f150", show=True)
#one_lcurve("Ceres", "pa5", "f150", show=True)
#test_weights("Interamnia", "pa5", "f150", show=True)
#phases("Interamnia", 8.727, "f090",show=True)
all_arrays("Pallas", "f150")