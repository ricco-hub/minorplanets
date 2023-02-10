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
sns.set_theme(style='ticks')

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
  
  try:
    indx = int(num_string)
    return indx
  except ValueError:
    print('Object not in current data set')

#get theory fluxes
def get_theory(name, freq):
  with open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/normalized_theory_flux_dict.pk", 'rb') as f:
    '''
    Format: [f090, f150, f220]
    '''
    normalized_theory_flux_dict = pk.load(f)
    
  try:
    if freq == "f090":
      return normalized_theory_flux_dict[name]["run5a"][0]
    elif freq == "f150":
      return normalized_theory_flux_dict[name]["run5a"][1]
    elif freq == "f220":
      return normalized_theory_flux_dict[name]["run5a"][2]
    else:
      print("Not a valid frequency, please try again")
  
  except KeyError:
    print("Object " + name + " not currently in flux theory file")      

def get_measured_flux(name, arr, freq):
  '''
    Inputs
      name, type: string, capitalized name of asteroid we want flux values for 
      arr, type: string, ACT array we want
      freq, type: string, frequency of flux we want
        NOTE: freq takes the format '150' rather than 'f150' as used elsewhere 
    Outputs
      ???flux stack of asteroid name???
  '''
  with open('/scratch/r/rbond/jorlo/actxminorplanets/sigurd/fluxes/' + name + '_flux_dict.pk','rb') as f:
    '''
    Format: {{day: {arr: freq, var}}}, {night: {arr: {freq, var}}}
    '''
    df = pk.load(f)
  try:
    print(df)#['flux'] ['night'][arr][freq]['flux']
  except KeyError:
    if 'f' in freq:
      print('Remove character ''f'' for valid frequency')
    else:
      print('KeyError')


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
  
def fit_sin(p,amp,offset):
  p = np.array(p)
  return amp * np.sin(4*np.pi*p + offset)

#get alpha - astroquery
def get_alpha(name, arr, freq, directory = None, show = False, save=False):
  '''
    Inputs:
      name, type: string, name of object we want
      arr, type: string, ACT array we want
      freq, type: string, frequency band
      directory, type: string, optionally save plots in specified directory
      show, type: boolean, optionally display plot right after calling get_alpha
      save, type: boolean, optionally save alpha and time data as pickle file
    
    Output:
      phase angle (deg) vs time for object 
  '''
    
  #Jack's maps
  path = "/scratch/r/rbond/jorlo/actxminorplanets/sigurd/asteroids/" + name
  
  #get rho, kappa, info files
  rho_files = glob.glob(path + "/*" + arr + "_" + freq + "_" + "rho.fits")
  kap_files = [utils.replace(r, "rho.fits", "kappa.fits") for r in rho_files]
  info_files = [utils.replace(r, "rho.fits", "info.hdf") for r in rho_files]
  
  if len(rho_files) != 0:
    #find time
    str_times = []
    t_start = len(path) + len(name) + 9
    t_end = t_start + 10
    for time in rho_files:
      str_times.append(time[t_start:t_end])
    int_times = [int(t) for t in str_times]
    
    alpha_data = []
    times_data = []
    for count, t in enumerate(int_times):      
      #get info
      info = bunch.read(info_files[count])
      ctime0 = np.mean(info.period)
      
      hour = (ctime0/3600) % 24
      
      #daytime maps
      if (11 < hour < 23):
        continue
      
      #nighttime maps
      else:
        #conver time to iso
        date = Time(t, format='unix')
        start_iso = date.iso
        
        #get ephemerides
        obj_earth = Horizons(id=name, location='W99', epochs={start_iso},id_type='asteroid_name')
        eph = obj_earth.ephemerides()
        
        #get alpha and time
        alpha = eph['alpha']
        alpha_data.append(alpha)
        time = eph['datetime_jd']
        times_data.append(time)
          
    mjd_date = utils.jd2mjd(times_data)
      
    #plot
    plt.scatter(mjd_date, alpha_data)    
    plt.xlabel("Time (MJD)")
    plt.ylabel("alpha (deg)")
    plt.title("alpha for {name} on {arr} at {freq}".format(name=name, arr=arr, freq=freq))    
    
    if show is not False:
      plt.show()
      
    if directory is not None:
      plt.savefig(directory + "{name}_alpha_plot_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))
      
    if save is not False:
      file_name = "alpha_" + name + ".pk"
      f = open(file_name, "wb")
      my_data = [mjd_date, alpha_data]
      pk.dump(my_data, f)
      f.close()
      
  else:
      print("No hits")

def one_lcurve(name, arr, freq, directory = None, show = False, pickle = False):
  '''
    Inputs:
      name, type: string, name of object we want
      arr, type: string, ACT array we want
      freq, type: string, frequency band
      directory, type: string, optionally save plots in specified directory
      show, type: boolean, optionally display plot right after calling one_lcurve()
    
    Output:
      Single daytime light curve for desired object with SPT-like F weighting
  '''
  
  index = get_index(name)
  
  #get ref flux
  ref_flux = get_theory(name, freq)
  print(ref_flux)     
  
  #Jack's maps
  path = "/scratch/r/rbond/jorlo/actxminorplanets/sigurd/asteroids/" + name
  
  #get rho, kappa, info files
  rho_files = glob.glob(path + "/*" + arr + "_" + freq + "_" + "rho.fits")
  kap_files = [utils.replace(r, "rho.fits", "kappa.fits") for r in rho_files]
  info_files = [utils.replace(r, "rho.fits", "info.hdf") for r in rho_files]
  
  if ref_flux is None:
    len(rho_files) == 0
  
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
    
    #get astroquery data to create line of best fit
    obj_earth = Horizons(id=name, location='W99', epochs={'start':start_iso, 'stop':end_iso, 'step':'1d'},id_type='asteroid_name')
    eph = obj_earth.ephemerides()
    
    el = obj_earth.elements()
    print(el['period'])
    
    time = eph['datetime_jd']
    mjd_times = utils.jd2mjd(time)    
    delta_earth_best = eph['delta'] #earth-asteroid distance
    delta_sun_best = eph['r'] #sun-asteroid distance
    alpha_best = eph['alpha'] 
    
    try:
      best_F = (delta_earth_best**(-2) * delta_sun_best**(-1/2)*10**(-0.004*alpha_best)) * ref_flux
    except TypeError:
      pass
      
    flux_data = []
    err_data = []
    times_data = []
        
    night_times = []    
    Fs = []
    
    for count, t in enumerate(int_times):      
      #get info
      info = bunch.read(info_files[count])
      ctime0 = np.mean(info.period)
      
      hour = (ctime0/3600) % 24
      
      #daytime maps
      if (11 < hour < 23):
        continue
      
      #nighttime maps
      else:      
        #convert time
        time_temp = Time(t, format='unix')
        time_iso = time_temp.iso
        eph_table = Horizons(id=name, location='W99', epochs={time_iso}, id_type='asteroid_name')
        tables = eph_table.ephemerides()
                 
        #get instantaneous ephemerides
        r_ast = tables['delta'][0] #earth-asteroid distance
        ra_ast = tables['RA'][0]
        dec_ast = tables['DEC'][0]
        d_earth = tables['delta'][0] #earth-asteroid distance
        d_sun = tables['r'][0] #sun-asteroid distance
          
        #find sun angle using vectors
        cur_time = Time(ctime0/86400. + 40587.0, format = 'mjd')
          
        sun = ephem.Sun()
        sun.compute(cur_time.utc.iso)
                
        #vectors        
        v_ea = utils.ang2rect([ra_ast*utils.degree, dec_ast*utils.degree])*r_ast
        v_es = utils.ang2rect([sun.ra, sun.dec])*sun.earth_distance
        sunang = utils.vec_angdist(-v_ea, -v_ea+v_es) / utils.degree
        
        #F weights
        try:
          F_weight = (d_earth**(-2) * d_sun**(-1/2)*10**(-0.004*sunang)) * ref_flux
        except:
          pass                             
          
        #get data
        kappa = enmap.read_map(kap_files[count])
        rho = enmap.read_map(rho_files[count])
          
        #cut bad maps       
        tol = 1e-2
        r = 5
        mask = kappa > np.max(kappa)*tol
        mask = mask.distance_transform(rmax=r) >= r 
        rho *= mask
        kappa *= mask
        
        if kappa[0,:,:].at([0,0]) <= 1e-9:
          continue         
        else:        
          #get flux, error, and time
          flux = rho / kappa 
          good_flux = flux[0].at([0, 0])
          flux_data.append(good_flux)
            
          err = np.abs(kappa)**(-0.5) 
          err_data.append(err[0].at([0,0]))
          
          night_times.append(t)
          Fs.append(F_weight)          
          
    night_mjd = utils.ctime2mjd(night_times)
    
    plt.clf()
    plt.errorbar(night_mjd, flux_data, yerr=err_data, fmt='o', label='Obs', zorder=0)#capsize=4, 
    plt.scatter(night_mjd, Fs, label='Theory', c='r', zorder=1)
    plt.plot(mjd_times, best_F, label='F weighting', ls='--', color=Blues_9.hex_colors[-2])
    plt.fill_between(mjd_times, 0.95*best_F, 1.05*best_F, label='95% uncertainty', fc=Blues_9.hex_colors[-2], alpha=0.4)
    plt.xlabel("Time (MJD)")
    plt.ylabel("Flux (mJy)")
    plt.legend(loc = 'best')
    plt.title("Light curve of {name} on {arr} at {freq}".format(name=name, arr=arr, freq=freq))         
    
    if show is not False:
      plt.show()
      
    if directory is not None:
      plt.savefig(directory + "{name}_light_curve_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))
      
    if pickle is not False:
      data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Flux': flux_data, 'F': Fs, 'Time': night_mjd, 'Error': err_data, 'Ref Flux': ref_flux, 'astroq F weight': best_F, 'astroq Times': mjd_times}
      filename = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_" + freq +".pk"
      outfile = open(filename, 'wb')
      pk.dump(data_dict, outfile)
      outfile.close()
      
  else:
      print("No hits")
      
def lcurve_freq(name, show=False, directory=None):
  '''
    Input
      name, type: string, name of object we want
      show, type: boolean, optionally display plot after running script
      directory, type: string, optionally save plots in specified directory
      
      Need to run lcurve first, or have access to lcurve data in order to plot across all frequencies
    Output
      Plot of light curves across all frequencies
  '''
  #get f090
  infile_f090 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name +"_f090.pk", 'rb')
  dict_f090 = pk.load(infile_f090)
  infile_f090.close()
  
  #get f150
  infile_f150 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name +"_f150.pk", 'rb')
  dict_f150 = pk.load(infile_f150)
  infile_f150.close()
    
  #get f220
  infile_f220 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name +"_f220.pk", 'rb')
  dict_f220 = pk.load(infile_f220)
  infile_f220.close()
  
  names = dict_f090['Name']  
       
  #get times, fluxs, Fs, errors and fits
  times_f090 = dict_f090['Time']
  flux_f090 = dict_f090['Flux']
  fWeights_f090 = dict_f090['F']
  error_f090 = dict_f090['Error']
  ref_F_weight_f090 = dict_f090['astroq F weight']
  fit_times_f090 = dict_f090['astroq Times']
  
  times_f150 = dict_f150['Time']
  flux_f150 = dict_f150['Flux']
  fWeights_f150 = dict_f150['F']
  error_f150 = dict_f150['Error']
  ref_F_weight_f150 = dict_f150['astroq F weight']
  fit_times_f150 = dict_f150['astroq Times']
    
  times_f220 = dict_f220['Time']
  flux_f220 = dict_f220['Flux']
  fWeights_f220 = dict_f220['F']
  error_f220 = dict_f220['Error']
  ref_F_weight_f220 = dict_f220['astroq F weight']
  fit_times_f220 = dict_f220['astroq Times']  
    
  #plot together
  plt.clf()    
  
  #f090
  plt.errorbar(times_f090, flux_f090, yerr=error_f090, fmt='.', label='Obs at 90 GHz', zorder=0, color='blue')
  plt.plot(fit_times_f090, ref_F_weight_f090, label='Theory', ls='--', color='blue')
  plt.fill_between(fit_times_f090, 0.95*ref_F_weight_f090, 1.05*ref_F_weight_f090, label='95% uncertainty', fc='blue', alpha=0.4)  

  #f150
  plt.errorbar(times_f150, flux_f150, yerr=error_f150, fmt='.', label='Obs at 150 GHz', zorder=0, color='orange')
  plt.plot(fit_times_f150, ref_F_weight_f150, ls='--', color='orange')
  plt.fill_between(fit_times_f150, 0.95*ref_F_weight_f150, 1.05*ref_F_weight_f150, fc='orange', alpha=0.4)  
    
  #f220
  plt.errorbar(times_f220, flux_f220, yerr=error_f220, fmt='.', label='Obs at 220 GHz', zorder=0, color='green')
  plt.plot(fit_times_f220, ref_F_weight_f220, ls='--', color='green')
  plt.fill_between(fit_times_f220, 0.95*ref_F_weight_f220, 1.05*ref_F_weight_f220, fc='green', alpha=0.4)      
        
  plt.xlabel("Time (MJD)")
  plt.ylabel("Flux (mJy)")
  plt.legend(loc='best')      
  plt.title("Light curves of {name} across 90 - 220 GHz".format(name=names))
    
  if show is not False:
    plt.show()  
      
  if directory is not None:
    plt.savefig(directory + "{name}_light_curves.pdf".format(name=names)) 

def lcurves(arr, freq, n, directory = None, show = False, pickle=False):
  '''
    Inputs:
    arr, type: arr, ACT array
    freq, type: freq, frequency we want
    n, type: integer, number of objects to make light curves for
    directory, type: string, optionally save file in directory
    show, type: boolean, if true, display light curve after calling lcurve
    pickle, type: boolean, optionally save data as pickle file
  
    Outputs:
      figure, creates light curve for n objects based on hits after running get_maps
      also plots F weighting
  '''
  
  for i in range(n):    
    #get name
    ignore_desig, name, ignore_semimajor_sun = get_desig(i)
    print(name)
    
    #get ref flux
    ref_flux = get_theory(name, freq)
    
    #Jack's maps
    path = "/scratch/r/rbond/jorlo/actxminorplanets/sigurd/asteroids/" + name 
    
    #get rho and kappa files
    rho_files = glob.glob(path + "/*" + arr + "_" + freq + "_" + "rho.fits")
    kap_files = [utils.replace(r, "rho.fits", "kappa.fits") for r in rho_files]
    info_files = [utils.replace(r, "rho.fits", "info.hdf") for r in rho_files]
    
    if len(rho_files) != 0 and ref_flux is not None:
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
      
      #get astroquery data to create line of best fit
      obj_earth = Horizons(id=name, location='W99',  epochs={'start':start_iso, 'stop':end_iso, 'step':'1d'},id_type='asteroid_name')
      eph = obj_earth.ephemerides()
      
      time = eph['datetime_jd']
      mjd_times = utils.jd2mjd(time)
      delta_earth_best = eph['delta'] #earth-asteroid distance
      delta_sun_best = eph['r'] #sun-asteroid distance
      alpha_best = eph['alpha']
      
      best_F = (delta_earth_best**(-2) * delta_sun_best**(-1/2)*10**(-0.004*alpha_best)) * ref_flux
      
      flux_data = []
      err_data = []
      times_data = []
      
      night_times = []
      Fs = []
      for count, t in enumerate(int_times):        
        #get info
        info = bunch.read(info_files[count])
        ctime0 = np.mean(info.period)
        
        hour = (ctime0/3600) % 24
        
        #daytime maps
        if (11 < hour < 23):
          continue
          
        #nighttime maps
        else:
          #convert time
          time_temp = Time(t, format='unix')
          time_iso = time_temp.iso
          eph_table = Horizons(id=name, location='W99', epochs={time_iso}, id_type='asteroid_name')
          tables = eph_table.ephemerides()
          
          #get instantaneous ephemerides
          r_ast = tables['delta'][0] #earth-asteroid distance
          ra_ast = tables['RA'][0]
          dec_ast = tables['DEC'][0]
          d_earth = tables['delta'][0] #earth-asteroid distance
          d_sun = tables['r'][0] #sun-asteroid distance
          
          #find sun angle using vectors        
          cur_time = Time(ctime0/86400. + 40587.0, format = 'mjd')
          
          sun = ephem.Sun()
          sun.compute(cur_time.utc.iso)
          
          #vectors
          v_ea = utils.ang2rect([ra_ast*utils.degree, dec_ast*utils.degree])*r_ast
          v_es = utils.ang2rect([sun.ra, sun.dec])*sun.earth_distance
          sunang = utils.vec_angdist(-v_ea, -v_ea+v_es) / utils.degree
        
          #F weights
          F_weight = (d_earth**(-2) * d_sun**(-1/2)*10**(-0.004*sunang)) * ref_flux 
                
          #get data
          kappa = enmap.read_map(kap_files[count])
          rho = enmap.read_map(rho_files[count])
          
          #cut bad maps
          tol = 1e-2
          r = 5
          mask = kappa > np.max(kappa)*tol
          mask = mask.distance_transform(rmax=r) >= r
          rho *= mask
          kappa *= mask
          
          if kappa[0,:,:].at([0,0]) <= 1e-9:
            continue
          else:
            #get flux, error, and time
            flux = rho / kappa
            good_flux = flux[0].at([0, 0])
            flux_data.append(good_flux)
            
            err = np.abs(kappa)**(-0.5)
            err_data.append(err[0].at([0,0]))
            
            night_times.append(t)
            Fs.append(F_weight)           
        
      night_mjd = utils.ctime2mjd(night_times)
          
      plt.clf()
      plt.errorbar(night_mjd, flux_data, yerr=err_data, fmt='o', label='Obs', zorder=0)
      plt.scatter(night_mjd, Fs, label='Theory', c='r', zorder=1)
      plt.plot(mjd_times, best_F, label='F weighting', ls='--', color=Blues_9.hex_colors[-2])
      plt.fill_between(mjd_times, 0.95*best_F, 1.05*best_F, label='95% uncertainty', fc=Blues_9.hex_colors[-2], alpha=0.4)
      plt.xlabel("Time (MJD)")
      plt.ylabel("Flux (mJy)")
      plt.legend(loc='best')      
      plt.title("Light curve of {name} on {arr} at {freq}".format(name=name, arr=arr, freq=freq))
        
      if show is not False:
        plt.show()
          
      if directory is not None:
        plt.savefig(directory + "{name}_light_curve_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))
          
      if pickle is not False:
        data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Flux': flux_data, 'F': Fs, 'Time': night_mjd, 'Error': err_data, 'Ref Flux': ref_flux, 'astroq F weight': best_F, 'astroq Times': mjd_times}
        filename = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/lcurve_data_" + name + "_" + freq +".pk"
        outfile = open(filename, 'wb')
        pk.dump(data_dict, outfile)
        outfile.close()
      
    else:
      continue
      
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
  infile_f090 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/lcurve_data_Vesta_f090.pk", 'rb')
  dict_f090 = pk.load(infile_f090)
  infile_f090.close()
  
  #get f150
  infile_f150 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/lcurve_data_Vesta_f150.pk", 'rb')
  dict_f150 = pk.load(infile_f150)
  infile_f150.close()
    
  #get f220
  infile_f220 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/lcurve_data_Vesta_f220.pk", 'rb')
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
  infile_f090 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/lcurve_data_f090.pk", "rb")#open('/scratch/r/rbond/ricco/minorplanets/lcurve_data_f090.pk', 'rb')
  dict_f090 = pk.load(infile_f090)
  infile_f090.close()
  
  #get f150
  infile_f150 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/lcurve_data_f150.pk", "rb")
  dict_f150 = pk.load(infile_f150)
  infile_f150.close()
    
  #get f220
  infile_f220 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/lcurve_data_f220.pk", "rb")
  dict_f220 = pk.load(infile_f220)
  infile_f220.close()
  
  names = dict_f090['Name']  
  print(names)
  print(len(names))
      
  for i in range(1): #len(dict)?
    #get times, fluxs, Fs, errors
    times_f090 = dict_f090['Time']#[i]
    flux_f090 = dict_f090['Flux']#[i]
    fWeights_f090 = dict_f090['F']#[i]
    error_f090 = dict_f090['Error']#[i]
  
    times_f150 = dict_f150['Time']#[i]
    flux_f150 = dict_f150['Flux']#[i]
    fWeights_f150 = dict_f150['F']#[i]
    error_f150 = dict_f150['Error']#[i]
    
    times_f220 = dict_f220['Time']#[i]
    flux_f220 = dict_f220['Flux']#[i]
    fWeights_f220 = dict_f220['F']#[i]
    error_f220 = dict_f220['Error']#[i]  
    
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

def phases(name, period, n, freq, show = False, directory = None, average=False):
  '''
  Inputs:
    name, type: string, name of object we want phase curve for
    period, type: float, rotational period of object in hours
    n, type: integer, number of bins
    freq, type: string, frequency we want to look at
    directory, type: string, optionally save file in directory
    show, type: boolean, if true, display light curve after calling lcurve
  Output:
    Phase curve of object given a period
    also returns residuals for phase
  '''
  
  pos = period / 24
  phases = []
  if freq == "f090":
    infile_f090 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name +"_f090.pk", 'rb')
    dict_f090 = pk.load(infile_f090)
    infile_f090.close()

    obj =  dict_f090['Name']
    
    #get times
    for t in range(len(dict_f090['Time'])):
      num = (dict_f090['Time'][t]) % pos
      phases.append(num/pos)
      
    #get data
    fluxs_f090 = dict_f090['Flux']
    error_f090 = dict_f090['Error']
    fWeights = dict_f090['F']   
    
    res = [(fluxs_f090[f] - fWeights[f]) for f in range(len(fluxs_f090))]
    
    plt.clf()
    if average is not False:
      interval = 1/n
      center = interval * 0.5      
      phase_bins = np.arange(0,1,interval) + center 
            
      #inverse weighting
      err_prop = []
      ave_var = []
      
      for i in np.arange(0,1,interval): 
        err_bin = [error_f090[f] for f in range(len(phases)) if phases[f] >= i and phases[f] <= (i+interval)]
        #find residuals in bin1 
        res_bin = [res[r] for r in range(len(phases)) if phases[r] >= i and phases[r] <= (i+interval)]
        err_data_sqr = [err_bin[e]**2 for e in range(len(err_bin))]
        ave_var_temp, i_var = inv_var(res_bin, err_data_sqr)
        new_err = i_var**0.5
        
        ave_var.append(ave_var_temp)
        err_prop.append(new_err)
       
      res_avg = mean(res)
      
      #fit
      params, params_covariance = optimize.curve_fit(fit_sin, phases, res)      
      
      #plot      
      plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='*', label='Inverse-Weighted Average Flux', ls='--', zorder=1)
      plt.errorbar(phases, res, yerr=error_f090, fmt='o', label='Flux', zorder=0)
      plt.axhline(y=res_avg, linestyle='-', label='Average Flux', zorder=0)
      plt.scatter(phases,fit_sin(phases, params[0], params[1]), label='Fitted Ftn.')
      plt.title("Average phase number")
      plt.xlabel("Phase Number")
      plt.ylabel("Flux Residual (mJy)")
      plt.legend(loc='best')
      
    else:    
      plt.errorbar(phases, res, yerr=error_f090, fmt='o', label='Flux at ' + freq, zorder=0)
        
    
  elif freq == "f150":
    #get f150
    infile_f150 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/lcurve_data_" + name +"_f150.pk", 'rb') 
    dict_f150 = pk.load(infile_f150)
    infile_f150.close()  
  
    obj = dict_f150['Name']
    
    #get times 
    for t in range(len(dict_f150['Time'])):
      num = (dict_f150['Time'][t]) % pos
      phases.append(num/pos)
      
    #get data
    fluxs_f150 = dict_f150['Flux']
    error_f150 = dict_f150['Error']
    fWeights = dict_f150['F']
    
    res = [(fluxs_f150[f]- fWeights[f]) for f in range(len(fluxs_f150))]
    
    plt.clf()
    if average is not False:
      interval = 1/n      
      center = interval * 0.5      
      phase_bins = np.arange(0,1,interval) + center 
            
      #inverse weighting
      err_prop = []
      ave_var = []
      
      for i in np.arange(0,1,interval): 
        err_bin = [error_f150[f] for f in range(len(phases)) if phases[f] >= i and phases[f] <= (i+interval)]
        #find residuals in bin1 
        res_bin = [res[r] for r in range(len(phases)) if phases[r] >= i and phases[r] <= (i+interval)]
        err_data_sqr = [err_bin[e]**2 for e in range(len(err_bin))]
        ave_var_temp, i_var = inv_var(res_bin, err_data_sqr)
        new_err = i_var**0.5
        
        ave_var.append(ave_var_temp)
        err_prop.append(new_err)
       
      res_avg = mean(res)
      
      #fit
      params, params_covariance = optimize.curve_fit(fit_sin, phases, res)      
       
      #plot
      plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='*', label='Inverse-Weighted Average Flux', ls='--', zorder=1)
      plt.errorbar(phases, res, yerr=error_f150, fmt='o', label='Flux', zorder=0)
      plt.axhline(y=res_avg, linestyle='-', label='Average Flux', zorder=0)      
      plt.scatter(phases,fit_sin(phases, params[0], params[1]), label='Fitted Ftn.')      
      plt.title("Average phase number")
      plt.xlabel("Phase Number")
      plt.ylabel("Flux Residual (mJy)")
      plt.legend(loc='best')
      
    
    else:
      plt.errorbar(phases, res, yerr=error_f150, fmt='o', label='Flux at ' + freq, zorder=0)
            
    
  elif freq == "f220":
    #get f220
    infile_f220 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/lcurve_data_" + name +"_f220.pk", 'rb')
    dict_f220 = pk.load(infile_f220)
    infile_f220.close()  

    obj =  dict_f220['Name']
    
    #get times 
    for t in range(len(dict_f220['Time'])):
      num = (dict_f220['Time'][t]) % pos
      phases.append(num/pos)
      
    #get data
    fluxs_f220 = dict_f220['Flux']
    error_f220 = dict_f220['Error']
    fWeights = dict_f220['F']
    
    res = [(fluxs_f220[f] - fWeights[f]) for f in range(len(fluxs_f220))]
    
    plt.clf()
    if average is not False:
      interval = 1/n      
      center = interval * 0.5      
      phase_bins = np.arange(0,1,interval) + center 
            
      #inverse weighting
      err_prop = []
      ave_var = []
      
      for i in np.arange(0,1,interval): 
        err_bin = [error_f220[f] for f in range(len(phases)) if phases[f] >= i and phases[f] <= (i+interval)]
        #find residuals in bin1 
        res_bin = [res[r] for r in range(len(phases)) if phases[r] >= i and phases[r] <= (i+interval)]
        err_data_sqr = [err_bin[e]**2 for e in range(len(err_bin))]
        ave_var_temp, i_var = inv_var(res_bin, err_data_sqr)
        new_err = i_var**0.5
        
        ave_var.append(ave_var_temp)
        err_prop.append(new_err)
       
      res_avg = mean(res)
      
      #fit
      params, params_covariance = optimize.curve_fit(fit_sin, phases, res)      
      
      #plot
      plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='*', label='Inverse-Weighted Average Flux', ls='--', zorder=1)
      plt.errorbar(phases, res, yerr=error_f220, fmt='o', label='Flux', zorder=0)
      plt.axhline(y=res_avg, linestyle='-', label='Average Flux', zorder=0)      
      plt.scatter(phases,fit_sin(phases, params[0], params[1]), label='Fitted Ftn.')      
      plt.title("Average phase number")
      plt.xlabel("Phase Number")
      plt.ylabel("Flux Residual (mJy)")
      plt.legend(loc='best')
      
    else:    
      plt.errorbar(phases, res, yerr=error_f220, fmt='o', label='Flux at ' + freq, zorder=0)

  else:
    print("No data for this frequency. Please select from 'f090, f150, or f220'") 

  #plt.ylim([-100,100])
  plt.xlabel("Phase Number")
  plt.ylabel("Flux Residual (mJy)")
  plt.title("Phase Curve of {name} at {freq}".format(name=obj,freq=freq))
  plt.legend(loc='best')
  
  if show is not False:
    plt.show()
        
  if directory is not None:
    plt.savefig(directory + "{name}_phase_curve_{freq}.pdf".format(name=obj, freq=freq))  
    
  return res
  
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

def period(name, arr, freq, period, directory = None, show = False):
  '''
    Inputs:
      name, type: string, name of object we want
      arr, type: string, ACT array we want
      freq, type: string, frequency band
      directory, type: string, optionally save plots in specified directory
      show, type: boolean, optionally display plot right after calling one_lcurve()
    
    Output:
      Normalized periodogram of object
  '''
  
  #get f090
  infile_f090 = open('/scratch/r/rbond/ricco/minorplanets/lcurve_data_f090.pk', 'rb')
  dict_f090 = pk.load(infile_f090)
  infile_f090.close()
  
  #get f150
  infile_f150 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/lcurve_data_f150.pk", 'rb')
  dict_f150 = pk.load(infile_f150)
  infile_f150.close()
    
  #get f220
  infile_f220 = open('/scratch/r/rbond/ricco/minorplanets/lcurve_data_f220.pk', 'rb')
  dict_f220 = pk.load(infile_f220)
  infile_f220.close()
  
  names = dict_f090['Name']
  
  res = phases(name, period, freq)
  #should I do this across all our data, or just one at a time?
  times_f150 = dict_f150['Time']
  #angular frequencies - 2pi/period???
  w = np.linspace(0.001, 2*np.pi / period, 10000)
  
  
  pgram = signal.lombscargle(times_f150, res, w, normalize=True)
  
  plt.clf()
  plt.plot(w, pgram)
  plt.title('Periodogram for {name} on {arr} at {freq}'.format(name=name, arr=arr, freq=freq))
  plt.xlabel('Angular frequency [rad/hr]')
  plt.ylabel('Normalized amplitude')
  
  if directory is not None:
    plt.savefig(directory + "{name}_periodogram_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))
    
  if show is not False:
    plt.show()
   
  
  
########################################RUN STUFF BELOW##########################################################
#all_lcurves(show = True)
#ratios(show = True)
#lcurves("pa5", "f150", 3, show=True)
#one_lcurve("Bamberga", "pa5", "f150", show=True)
#one_lcurve_fit(name='Vesta', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", show=True)
#ratios(show=True) 
#test_weights("Interamnia", "pa5", "f150", show=True)
phases("Vesta", 5.34212766, 5, "f090", show=True, average=True)
#all_arrays("Pallas", "f150")
#get_alpha("Pallas", "pa4", "f150", show=True)
#period("Vesta", "pa5", "f220",5.34212766, show=True)

#one_lcurve_fit(name="Hebe", show=True)
#get_measured_flux('Vesta', 'pa5', '150')