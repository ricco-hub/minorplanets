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
  
def fit_sin(p,amp,offset,v):
  '''
    Inputs
      p, type: float, rotational phases of object
      amp, type: float, amplitude of fit
      offset, type: float, offset of fit
      v, type: float, vertical shift of fit
      
    Output
      sin fit for phase curve
  '''
  p = np.array(p)
  return amp * np.sin(4*np.pi*p + offset) + v
  
def vesta_fit(p,A,B,C,D,E):
  p = np.array(p)
  return A*np.sin(2*np.pi*p + B) + C*np.sin(4*np.pi*p + D) + E
  
def sin2model(x, A1, phi1, A2, phi2, C):
    #A1, phi1, A2, phi2, C = p
    return A1 * np.sin(4*np.pi*x + phi1) + A2 * np.sin(2*np.pi*x + phi2) + C  
  
def inv_var_weight(n,errs,phases,res):
  '''
    Inputs
      n, type: integer, number of weighted bins
      errs, type: float, errors for binning
      phases, type: float, phases to be binned (x data)
      res, type: float, residuals to be binned (y data)
    Outputs
      ave_var, type: float, inverse variance weighted average
      err_var, type: float, associated errors from inverse variance weighted average
      phase_bins, type: float, bins 
  '''
  interval = 1/n
  center = interval * 0.5      
  phase_bins = np.arange(0,1,interval) + center 
            
  #inverse weighting
  err_prop = []
  ave_var = []
      
  for i in np.arange(0,1,interval):
    try: 
      err_bin = [errs[f] for f in range(len(phases)) if phases[f] >= i and phases[f] <= (i+interval)]
      res_bin = [res[r] for r in range(len(phases)) if phases[r] >= i and phases[r] <= (i+interval)]
      err_data_sqr = [err_bin[e]**2 for e in range(len(err_bin))]
      ave_var_temp, i_var = inv_var(res_bin, err_data_sqr)
      new_err = i_var**0.5
          
      ave_var.append(ave_var_temp)
      err_prop.append(new_err)
    except ZeroDivisionError:
      print('Zero Division Error')      
      ave_var.append(np.nan)
      err_prop.append(np.nan)
      
  return ave_var, err_prop, phase_bins

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
    
      use asteroid times, not time of depth-1 maps
    Output:
      Single daytime light curve for desired object with SPT-like F weighting
  '''
  
  index = get_index(name)
  
  #get ref flux
  ref_flux = get_theory(name, freq)
  
  #Jack's maps
  path = "/scratch/r/rbond/jorlo/actxminorplanets/sigurd/asteroids/" + name
  
  #get rho, kappa, info files
  rho_files = glob.glob(path + "/*" + arr + "_" + freq + "_" + "rho.fits")
  kap_files = [utils.replace(r, "rho.fits", "kappa.fits") for r in rho_files]
  info_files = [utils.replace(r, "rho.fits", "info.hdf") for r in rho_files]
  
  if ref_flux is None:
    len(rho_files) == 0
  
  astroQ_times = []
  
  flux_data = []
  err_data = []
  times_data = []
       
  night_times = []    
  Fs = []
  if len(rho_files) != 0:
    #find max and min time
    for t in range(len(info_files)):
      ifile = info_files[t]
      info = bunch.read(ifile)
      ctime0 = info.ctime_ast
      astroQ_times.append(ctime0)      
      
      hour = (ctime0/3600) % 24
      
      #daytime maps
      if (11 < hour < 23):
        continue
        
      #nighttime maps
      else:
        time_temp = Time(ctime0, format='unix')
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
        kappa = enmap.read_map(kap_files[t])
        rho = enmap.read_map(rho_files[t])
          
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
          
          night_times.append(ctime0)
          Fs.append(F_weight)              
        
    night_mjd = utils.ctime2mjd(night_times)    
    
    #find min and max times for astroquery
    start = Time(min(astroQ_times), format='unix')
    end = Time(max(astroQ_times), format='unix')    
    start_iso = start.iso
    end_iso = end.iso
    
    #get astroquery data to create line of best fit
    obj_earth = Horizons(id=name, epochs={'start':start_iso, 'stop':end_iso, 'step':'1d'},id_type='asteroid_name')#location='W99', 
    eph = obj_earth.ephemerides()
    
    el = obj_earth.elements()
    
    time = eph['datetime_jd']
    mjd_times = utils.jd2mjd(time)    
    delta_earth_best = eph['delta'] #earth-asteroid distance
    delta_sun_best = eph['r'] #sun-asteroid distance
    alpha_best = eph['alpha'] 
    
    try:
      best_F = (delta_earth_best**(-2) * delta_sun_best**(-1/2)*10**(-0.004*alpha_best)) * ref_flux
    except TypeError:
      pass
    
    #plot
    plt.clf()
    plt.errorbar(night_mjd, flux_data, yerr=err_data, fmt='o', label='Obs', zorder=0)#capsize=4,  
    plt.scatter(night_mjd, Fs, label='Theory', c='r', zorder=1) 
    plt.plot(mjd_times, best_F, label='F Weighting', ls='--', color=Blues_9.hex_colors[-2])
    plt.fill_between(mjd_times, 0.95*best_F, 1.05*best_F, label='95% uncertainty', fc=Blues_9.hex_colors[-2], alpha=0.4)
    plt.xlabel("Time (yr)")
    plt.ylabel("Flux (mJy)")
    plt.legend(loc = 'best')
    plt.title("Light curve of {name} on {arr} at {freq}".format(name=name, arr=arr, freq=freq))          
    
    if show is not False:
      plt.show()
      
    if directory is not None:
      plt.savefig(directory + "{name}_light_curve_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))
      
    if pickle is not False:
      data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Flux': flux_data, 'F': Fs, 'Time': night_mjd, 'Error': err_data, 'Ref Flux': ref_flux, 'astroq F weight': best_F, 'astroq Times': mjd_times}
      filename = directory + "lcurve_data_"+ name + "_" + arr + "_" + freq +".pk"
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
        filename = directory + "lcurve_data_" + name + "_" + arr + "_" + freq +".pk"
        outfile = open(filename, 'wb')
        pk.dump(data_dict, outfile)
        outfile.close()
      
    else:
      continue
      
def get_lcurves(arr, freq, n, directory = None):
  '''
    Inputs:
    arr, type: arr, ACT array
    freq, type: freq, frequency we want
    n, type: integer, number of objects to make light curves for
    directory, type: string, optionally save file in directory
    show, type: boolean, if true, display light curve after calling lcurve
    pickle, type: boolean, optionally save data as pickle file
  
    Outputs:
      pickle file of light curve data without F weights 
  '''
  
  for i in range(n):    
    #get name
    ignore_desig, name, ignore_semimajor_sun = get_desig(i)
    print(name, i)
    
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
          
      data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Flux': flux_data, 'Time': night_mjd, 'Error': err_data, 'Ref Flux': ref_flux, 'Weighting': Fs}
      filename = directory + "lcurve_data_" + name + "_" + arr + "_" + freq +".pk"
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

def phases(name, n, arr, freq, show = False, directory = None):
  '''
  Inputs:
    name, type: string, name of object we want phase curve for
    n, type: integer, number of bins for inverse-weighted average
    freq, type: string, frequency we want to look at
    show, type: boolean, if true, display light curve after calling lcurve    
    directory, type: string, optionally save file in directory
  Output:
    Phase curve of object given a period
    also returns residuals for phase
  '''
  period = get_period(name)
  
  pos = period / 24
  phases = []
  #print("HEADER  CHI_SQR  RED_CHI_SQR  AMP  ERROR  S/N  PTE")
  if freq == "f090":
    infile_f090 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_" + arr + "_f090.pk", 'rb')
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
    fWeights = dict_f090['Weighting']   
    
    res = [(fluxs_f090[f] - fWeights[f]) for f in range(len(fluxs_f090))]
    
    plt.clf()
    
    interval = 1/n
    center = interval * 0.5      
    phase_bins = np.arange(0,1,interval) + center 
            
    #inverse weighting
    err_prop = []
    ave_var = []
      
    for i in np.arange(0,1,interval):
      try: 
        err_bin = [error_f090[f] for f in range(len(phases)) if phases[f] >= i and phases[f] <= (i+interval)]
        #find residuals in bin1 
        res_bin = [res[r] for r in range(len(phases)) if phases[r] >= i and phases[r] <= (i+interval)]
        err_data_sqr = [err_bin[e]**2 for e in range(len(err_bin))]
        ave_var_temp, i_var = inv_var(res_bin, err_data_sqr)
        new_err = i_var**0.5
          
        ave_var.append(ave_var_temp)
        err_prop.append(new_err)
      except ZeroDivisionError:
        print('Zero Division Error')      
        ave_var.append(np.nan)
        err_prop.append(np.nan)
       
    res_avg = mean(res)
      
    #sin fit
    params, params_covariance = optimize.curve_fit(fit_sin, phases, res, sigma=error_f090)   
    perr = np.sqrt(np.diag(params_covariance))   
    x = np.arange(0,1,0.01)
    y_fit = fit_sin(x, params[0], params[1], params[2])
      
    #chi-square sin
    diff = (fit_sin(phases, params[0], params[1], params[2]) - res)**2
    error_f090 = np.array(error_f090)
    chi_sqr = np.sum(diff / (error_f090**2))
    dof = len(res) - 3
    red_chi_sqr = chi_sqr / dof
    if abs(params[0]/perr[0]) > 5:
      print('POSSIBLE DETECTION')
    print(name + "_" + arr + "_" + freq, chi_sqr, red_chi_sqr, params[0], perr[0], abs(params[0]/perr[0]), chi2.sf(chi_sqr, dof))        
    
    #print("Chi-squared for fit: ", chi_sqr)
    #print("DOF: ", dof) 
    #print("Reduced chi-squared for fit: ", red_chi_sqr)
    #print("PTE for fit: ", chi2.sf(chi_sqr, dof))
    #print("Amplitude: ", params[0], "with uncertainty: ", perr[0])
      
    #chi-square average     
    diff_avg = (res_avg - res)**2
    chi_sqr_avg = np.sum(diff_avg / (error_f090**2))
    #print("Chi-squared for average: ", chi_sqr_avg)            
      
    #plot      
    plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='s', label='Bin Flux', zorder=1)
    plt.errorbar(phases, res, yerr=error_f090, fmt='o', label='Flux', zorder=0,alpha=0.5)
    plt.axhline(y=res_avg, linestyle='--', label='Average Flux', zorder=0)
    plt.plot(x,y_fit, label='Fitted Ftn.')
    
    #data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Phase Number': phases, 'Residuals': res, 'Error': error_f090, 'Measured':fluxs_f090,'Model (F Weights)':fWeights}
    #filename = directory + "phase_curve_data_" + name + "_" + arr + "_" + freq +".pk"
    #outfile = open(filename, 'wb')
    #pk.dump(data_dict, outfile)
    #outfile.close()        
    
  elif freq == "f150":
    plt.rcParams.update({'font.size': 16})
    #get f150
    infile_f150 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_" + arr + "_f150.pk", 'rb') 
    dict_f150 = pk.load(infile_f150)
    infile_f150.close()  
  
    obj = dict_f150['Name']
    
    times = dict_f150['Time']
    
    #get times 
    for t in range(len(dict_f150['Time'])):
      num = (dict_f150['Time'][t]) % pos
      phases.append(num/pos)
      
    #get data
    fluxs_f150 = dict_f150['Flux']
    error_f150 = dict_f150['Error']
    fWeights = dict_f150['Weighting']
    
    res = [(fluxs_f150[f]- fWeights[f]) for f in range(len(fluxs_f150))]
    
    plt.clf()

    interval = 1/n      
    center = interval * 0.5      
    phase_bins = np.arange(0,1,interval) + center 
            
    #inverse weighting
    err_prop = []
    ave_var = []
      
    for i in np.arange(0,1,interval):
      try: 
        err_bin = [error_f150[f] for f in range(len(phases)) if phases[f] >= i and phases[f] <= (i+interval)]
        #print(err_bin)
        #find residuals in bin1 
        res_bin = [res[r] for r in range(len(phases)) if phases[r] >= i and phases[r] <= (i+interval)]
        err_data_sqr = [err_bin[e]**2 for e in range(len(err_bin))]
        #print(err_data_sqr)
        ave_var_temp, i_var = inv_var(res_bin, err_data_sqr)
        new_err = i_var**0.5
          
        ave_var.append(ave_var_temp)
        err_prop.append(new_err)
      except ZeroDivisionError:
        print('Zero Division Error')
        ave_var.append(np.nan)
        err_prop.append(np.nan)
        continue
       
    res_avg = mean(res)
      
    #sin fit
    params, params_covariance = optimize.curve_fit(fit_sin, phases, res, sigma=error_f150)   
    perr = np.sqrt(np.diag(params_covariance))   
    x = np.arange(0,1,0.01)
    y_fit = fit_sin(x, params[0], params[1], params[2])      
      
    #chi-square sin
    diff = (fit_sin(phases, params[0], params[1], params[2]) - res)**2
    error_f150 = np.array(error_f150)
    chi_sqr = np.sum(diff / (error_f150**2))
    dof = len(res) - 3
    red_chi_sqr = chi_sqr / dof
    diff = (fit_sin(phases, params[0], params[1], params[2]) - res)**2
    error_f150 = np.array(error_f150)
    chi_sqr = np.sum(diff / (error_f150**2))
    dof = len(res) - 3
    red_chi_sqr = chi_sqr / dof
    if abs(params[0]/perr[0]) > 5:
      print('POSSIBLE DETECTION')    
    print(name + "_" + arr + "_" + freq, chi_sqr, red_chi_sqr, params[0], perr[0], abs(params[0]/perr[0]), chi2.sf(chi_sqr, dof))
            
    #print(name, freq)         
    #print("Chi-squared for fit: ", chi_sqr)
    #print("DOF: ", dof) 
    #print("Reduced chi-squared for fit: ", red_chi_sqr)
    #print("PTE for fit: ", chi2.sf(chi_sqr, dof))
    #print("Amplitude: ", params[0], "with uncertainty: ", perr[0])
      
    #chi-square average     
    diff_avg = (res_avg - res)**2
    chi_sqr_avg = np.sum(diff_avg / (error_f150**2))
    #print("Chi-squared for average: ", chi_sqr_avg)      
         
    #plot
    plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='s', label='Bin Flux', c='r', zorder=1)
    plt.errorbar(phases, res, yerr=error_f150, fmt='o', label='Flux', zorder=0,alpha=0.5)
    plt.axhline(y=res_avg, linestyle='--', label='Average Flux', zorder=0)      
    plt.plot(x,y_fit, label='Fitted Ftn.')     
    
    data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Phase Number': phases, 'Residuals': res, 'Error': error_f150, 'Measured':fluxs_f150,'Model (F Weights)':fWeights}
    filename = directory + "phase_curve_data_" + name + "_" + arr + "_" + freq +".pk"
    outfile = open(filename, 'wb')
    pk.dump(data_dict, outfile)
    outfile.close()             
    
  elif freq == "f220":
    #get f220
    infile_f220 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_" + arr + "_f220.pk", 'rb')
    dict_f220 = pk.load(infile_f220)
    infile_f220.close()  

    obj = dict_f220['Name']
    
    #get times 
    for t in range(len(dict_f220['Time'])):
      num = (dict_f220['Time'][t]) % pos
      phases.append(num/pos)
      
    #get data
    fluxs_f220 = dict_f220['Flux']
    error_f220 = dict_f220['Error']
    fWeights = dict_f220['Weighting']
    
    res = [(fluxs_f220[f] - fWeights[f]) for f in range(len(fluxs_f220))]
    
    plt.clf()

    interval = 1/n      
    center = interval * 0.5      
    phase_bins = np.arange(0,1,interval) + center 
            
    #inverse weighting
    err_prop = []
    ave_var = []
      
    for i in np.arange(0,1,interval): 
      try:
        err_bin = [error_f220[f] for f in range(len(phases)) if phases[f] >= i and phases[f] <= (i+interval)]
        #find residuals in bin1 
        res_bin = [res[r] for r in range(len(phases)) if phases[r] >= i and phases[r] <= (i+interval)]
        err_data_sqr = [err_bin[e]**2 for e in range(len(err_bin))]
        ave_var_temp, i_var = inv_var(res_bin, err_data_sqr)
        new_err = i_var**0.5
          
        ave_var.append(ave_var_temp)
        err_prop.append(new_err)
      except ZeroDivisionError:
        print('Zero Division Error')
        ave_var.append(np.nan)
        err_prop.append(np.nan)
       
    res_avg = mean(res)
      
    #sin fit
    params, params_covariance = optimize.curve_fit(fit_sin, phases, res, sigma=error_f220)   
    perr = np.sqrt(np.diag(params_covariance))   
    x = np.arange(0,1,0.01)
    y_fit = fit_sin(x, params[0], params[1], params[2])      
      
    #chi-square sin
    diff = (fit_sin(phases, params[0], params[1], params[2]) - res)**2
    error_f220 = np.array(error_f220)
    chi_sqr = np.sum(diff / (error_f220**2))
    dof = len(res) - 3
    red_chi_sqr = chi_sqr / dof
    diff = (fit_sin(phases, params[0], params[1], params[2]) - res)**2
    error_f220 = np.array(error_f220)
    chi_sqr = np.sum(diff / (error_f220**2))
    dof = len(res) - 3
    red_chi_sqr = chi_sqr / dof
    if abs(params[0]/perr[0]) > 5:
      print('POSSIBLE DETECTION')    
    print(name + "_" + arr + "_" + freq, chi_sqr, red_chi_sqr, params[0], perr[0], abs(params[0]/perr[0]), chi2.sf(chi_sqr, dof))  
       
    #print(name, freq)              
    #print("Chi-squared for fit: ", chi_sqr)
    #print("DOF: ", dof) 
    #print("Reduced chi-squared for fit: ", red_chi_sqr)
    #print("PTE for fit: ", chi2.sf(chi_sqr, dof))
    #print("Amplitude: ", params[0], "with uncertainty: ", perr[0])
      
    #chi-square average     
    diff_avg = (res_avg - res)**2
    chi_sqr_avg = np.sum(diff_avg / (error_f220**2))
    #print("Chi-squared for average: ", chi_sqr_avg)      
      
    #plot
    plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='s', label='Bin Flux', zorder=1)
    plt.errorbar(phases, res, yerr=error_f220, fmt='o', label='Flux', zorder=0,alpha=0.5)
    plt.axhline(y=res_avg, linestyle='--', label='Average Flux', zorder=0)      
    plt.plot(x,y_fit, label='Fitted Ftn.')      
    
    data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Phase Number': phases, 'Residuals': res, 'Error': error_f220, 'Measured':fluxs_f220,'Model (F Weights)':fWeights}
    filename = directory + "phase_curve_data_" + name + "_" + arr + "_" + freq +".pk"
    outfile = open(filename, 'wb')
    pk.dump(data_dict, outfile)
    outfile.close()            
      
  else:
    print("No data for this frequency. Please select from 'f090, f150, f220'") 

  plt.xlabel("Phase Number")
  plt.ylabel("Flux Residual (mJy)")
  plt.legend(loc='best')
  
  if show is not False:
    plt.show()
        
  if directory is not None:
    plt.savefig(directory + "{name}_phase_curve_{freq}.pdf".format(name=obj, freq=freq))  
    
  return res
  
def all_arrays(name, freq, directory = None, show = False, pickle=False):
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
  
  #Jack's maps
  path = "/scratch/r/rbond/jorlo/actxminorplanets/sigurd/asteroids/" + name
  
  arrays = ["pa4","pa5","pa6"]
  
  flux_data = []
  err_data = []
  times_data = []
  arrs_used = [] 
  night_times = []    
  Fs = []
  Night_mjd = []
  
  times = []
  fit = []     
  
  astroQ_times = []
  
  for i, arr in enumerate(arrays):  
    #get rho, kappa, info files
    rho_files = glob.glob(path + "/*" + arr + "_" + freq + "_" + "rho.fits")
    kap_files = [utils.replace(r, "rho.fits", "kappa.fits") for r in rho_files]
    info_files = [utils.replace(r, "rho.fits", "info.hdf") for r in rho_files]
    
    if ref_flux is None:
      len(rho_files) == 0
    
    if len(rho_files) != 0:
      arrs_used.append(arr)
      #find time
      for t in range(len(info_files)):
        ifile = info_files[t]
        info = bunch.read(ifile)
        ctime0 = info.ctime_ast
        astroQ_times.append(ctime0)
        
        hour = (ctime0/3600) % 24
        
        #daytime maps
        if (11 < hour < 23):
          continue
          
        #nighttime maps
        else:
          time_temp = Time(ctime0,format='unix')
          time_iso = time_temp.iso
          eph_table = Horizons(id=name, location='W99', epochs={time_iso},id_type='asteroid_name')
          tables = eph_table.ephemerides()
          
          #get instantaneous ephemerides
          r_ast = tables['delta'][0] #earth-asteroid distance
          ra_ast = tables['RA'][0]
          dec_ast = tables['DEC'][0]
          d_earth = tables['delta'][0] #earth_asteroid distance
          d_sun = tables['r'][0] #sun-asteroid distance
          
          #find sun angle using vectors
          cur_time = Time(ctime0/86400. + 40587.0, format='mjd')
          
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
          kappa = enmap.read_map(kap_files[t])
          rho = enmap.read_map(rho_files[t])
          
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
            flux = rho/kappa
            good_flux = flux[0].at([0, 0])
            flux_data.append(good_flux)
            
            err = np.abs(kappa)**(-0.5)
            err_data.append(err[0].at([0,0]))
            
            night_times.append(ctime0)
            Fs.append(F_weight)
            
            times.append(ctime0)
            
        night_mjd = utils.ctime2mjd(night_times)
        
    else:
      times.append(np.NaN)
      print("No hits")
  
  #find min(time) from times, max(time) from times --> use astroquery to plot between these values
  min_time = np.nanmin(times)
  max_time = np.nanmax(times)
  start = Time(min_time,format='unix')
  end = Time(max_time,format='unix')    
  start_iso = start.iso
  end_iso = end.iso
      
  #get astroquery data to create line of best fit
  obj_earth = Horizons(id=name, epochs={'start':start_iso, 'stop':end_iso, 'step':'1d'},id_type='asteroid_name')#location='W99', 
  eph = obj_earth.ephemerides()
     
  el = obj_earth.elements()
      
  time = eph['datetime_jd']
  mjd_times = utils.jd2mjd(time)  
  delta_earth_best = eph['delta'] #earth-asteroid distance
  delta_sun_best = eph['r'] #sun-asteroid distance
  alpha_best = eph['alpha'] 
      
  try:
    best_F = (delta_earth_best**(-2) * delta_sun_best**(-1/2)*10**(-0.004*alpha_best)) * ref_flux
    #best_F = np.array(best_F)
    fit.append(best_F)
  except TypeError:
    pass  
  
  plt.errorbar(night_mjd, flux_data, yerr=err_data, fmt='o', label='Obs', zorder=0)#capsize=4,  
  plt.scatter(night_mjd, Fs, label='Theory', c='r', zorder=1)
  plt.plot(mjd_times, best_F, label='F Weighting', ls='--', color=Blues_9.hex_colors[-2])
  plt.fill_between(mjd_times, 0.95*best_F, 1.05*best_F, label='95% uncertainty', fc=Blues_9.hex_colors[-2], alpha=0.4)   
  plt.xlabel("Time (MJD)")
  plt.ylabel("Flux (mJy)")
  plt.legend(loc = 'best')
  plt.title("Light curve of {name} at {freq}".format(name=name, freq=freq))          
      
  if show is not False:
    plt.show()
        
  if directory is not None:
    plt.savefig(directory + "{name}_lcurve_data_all_{freq}.pdf".format(name=name, freq=freq))
        
  if pickle is not False:
    data_dict = {'Name': name, 'Array': arrs_used, 'Frequency': freq, 'Flux': flux_data, 'F': Fs, 'Time': night_mjd, 'Error': err_data, 'Ref Flux': ref_flux, 'astroq F weight': best_F, 'astroq Times': mjd_times}
    filename = directory + "lcurve_data_" + name + "_all_" + freq +".pk"
    outfile = open(filename, 'wb')
    pk.dump(data_dict, outfile)
    outfile.close()

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
   
def get_API(n,freq):
  '''
  Input
  Output
    Return pickle file of objects with their rotation period
  '''
  names = []
  pers = []
  for i in range(n):    
    #get name
    ignore_desig, name, ignore_semimajor_sun = get_desig(i)
    print(name)
    
    #get ref flux
    ref_flux = get_theory(name, freq)
    per = get_period(name)
    
    names.append(name)
    pers.append(per)

  api_per = {'Name': names, 'Rotation Period': pers}
  filename = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/phase_curves/rot_period_data_" + freq + ".pk"
  outfile = open(filename, 'wb')
  pk.dump(api_per, outfile)
  outfile.close()
          
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
  
def better_phases(name, n, arr, freq, show = False, directory = None, pickle=False):
  '''
  Inputs:
    name, type: string, name of object we want phase curve for
    n, type: integer, number of bins for inverse-weighted average
    freq, type: string, frequency we want to look at
    show, type: boolean, if true, display light curve after calling lcurve    
    directory, type: string, optionally save file in directory
    pickle, type: boolean, optionally pickle phase data
  Output:
    Phase curve of object given a period
    also returns residuals for phase
    Incorporates rotational, orbital period
    solar phase angle in binning
  '''
  if freq == "f090":
    #infile_f090 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_" + arr + "_f090.pk", 'rb')
    infile_f090 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_all" + "_f090.pk", 'rb')
    dict_f090 = pk.load(infile_f090)
    infile_f090.close()

    #get data
    fluxs_f090 = dict_f090['Flux']
    error_f090 = dict_f090['Error']
    fWeights = dict_f090['F'] #Weighting
    obj = dict_f090['Name']
    time_f090 = dict_f090['Time']
    
    #get phase numbers
    phi = rot_phase(obj, time_f090) #rotation phase numbers
    phi = np.array(phi)    
    theta, ignore_T = orb_phase(obj, time_f090) #orbital phase numbers
    theta = np.array(theta)    
    alpha = sunang_phase(obj, time_f090) #solar phase angle numbers
    alpha = np.array(alpha)          
    
    res = [(fluxs_f090[f] - fWeights[f]) for f in range(len(fluxs_f090))]
    res_avg = mean(res)    
    #phi = -1*phi
    
    eta = phi + theta + alpha
    eta_phase = eta % 1
    
    #binning
    ave_var, err_prop, phase_bins = inv_var_weight(n,error_f090,eta_phase,res)
      
    #sin fit
    eta_phase, res, error_f090 = zip(*sorted(zip(eta_phase,res,error_f090)))    
    params, params_covariance = optimize.curve_fit(fit_sin, eta_phase, res, sigma=error_f090)   
    perr = np.sqrt(np.diag(params_covariance))   
    x = np.arange(0,1,0.01)
    y_fit = fit_sin(x, params[0], params[1], params[2])
      
    #chi-square sin
    diff = (fit_sin(eta_phase, params[0], params[1], params[2]) - res)**2
    error_f090 = np.array(error_f090)
    chi_sqr = np.sum(diff / (error_f090**2))
    dof = len(res) - 3
    red_chi_sqr = chi_sqr / dof
    if abs(params[0]/perr[0]) > 5:
      print('POSSIBLE DETECTION')
    print(name + "_" + arr + "_" + freq, chi_sqr, red_chi_sqr, params[0], perr[0], abs(params[0]/perr[0]), chi2.sf(chi_sqr, dof))        
    
    #print("Chi-squared for fit: ", chi_sqr)
    #print("DOF: ", dof) 
    #print("Reduced chi-squared for fit: ", red_chi_sqr)
    #print("PTE for fit: ", chi2.sf(chi_sqr, dof))
    #print("Amplitude: ", params[0], "with uncertainty: ", perr[0])
      
    #chi-square average     
    diff_avg = (res_avg - res)**2
    chi_sqr_avg = np.sum(diff_avg / (error_f090**2))
    #print("Chi-squared for average: ", chi_sqr_avg)            
      
    #plot      
    plt.clf()    
    plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='s', label='Bin Flux', zorder=1)
    plt.errorbar(eta_phase, res, yerr=error_f090, fmt='o', label='Flux', zorder=0,alpha=0.5)
    plt.axhline(y=res_avg, linestyle='--', label='Average Flux', zorder=0)
    plt.plot(x,y_fit, label='Fitted Ftn.')
    
    if pickle is not False:
      data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Phase Number': eta_phase, 'Residuals': res, 'Error': error_f090, 'Measured':fluxs_f090,'Model (F Weights)':fWeights}
      filename = directory + "phase_curve_data_" + name + "_" + arr + "_" + freq +".pk"
      outfile = open(filename, 'wb')
      pk.dump(data_dict, outfile)
      outfile.close()        
    
  elif freq == "f150":
    plt.rcParams.update({'font.size': 16})
    #get f150
    infile_f150 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_" + arr + "_f150.pk", 'rb') 
    #infile_f150 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_all" + "_f150.pk", 'rb')
    dict_f150 = pk.load(infile_f150)
    infile_f150.close()  
  
    #get data
    fluxs_f150 = dict_f150['Flux']
    error_f150 = dict_f150['Error']
    fWeights = dict_f150['F']
    obj = dict_f150['Name']
    time_f150 = dict_f150['Time']
    
    #get phase numbers
    phi = rot_phase(obj, time_f150) #rotation phase numbers
    phi = np.array(phi)    
    theta, ignore_T = orb_phase(obj, time_f150) #orbital phase numbers
    theta = np.array(theta)    
    alpha = sunang_phase(obj, time_f150) #solar phase angle numbers     
    
    res = [(fluxs_f150[f] - fWeights[f]) for f in range(len(fluxs_f150))]
    res_avg = mean(res)    
    
    eta = phi + theta + alpha
    eta_phase = eta % 1    
    
    #binning
    ave_var, err_prop, phase_bins = inv_var_weight(n,error_f150,eta_phase,res)    
      
    #sin fit
    eta_phase, res, error_f150 = zip(*sorted(zip(eta_phase,res,error_f150)))
    params, params_covariance = optimize.curve_fit(fit_sin, eta_phase, res, sigma=error_f150)   
    perr = np.sqrt(np.diag(params_covariance))   
    x = np.arange(0,1,0.01)
    y_fit = fit_sin(x, params[0], params[1], params[2])      
      
    #chi-square sin
    diff = (fit_sin(eta_phase, params[0], params[1], params[2]) - res)**2
    error_f150 = np.array(error_f150)
    chi_sqr = np.sum(diff / (error_f150**2))
    dof = len(res) - 3
    red_chi_sqr = chi_sqr / dof
    diff = (fit_sin(eta_phase, params[0], params[1], params[2]) - res)**2
    error_f150 = np.array(error_f150)
    chi_sqr = np.sum(diff / (error_f150**2))
    dof = len(res) - 3
    red_chi_sqr = chi_sqr / dof
    if abs(params[0]/perr[0]) > 5:
      print('POSSIBLE DETECTION')    
    print(name + "_" + arr + "_" + freq, chi_sqr, red_chi_sqr, params[0], perr[0], abs(params[0]/perr[0]), chi2.sf(chi_sqr, dof))
            
    #print(name, freq)         
    #print("Chi-squared for fit: ", chi_sqr)
    #print("DOF: ", dof) 
    #print("Reduced chi-squared for fit: ", red_chi_sqr)
    #print("PTE for fit: ", chi2.sf(chi_sqr, dof))
    #print("Amplitude: ", params[0], "with uncertainty: ", perr[0])
      
    #chi-square average     
    diff_avg = (res_avg - res)**2
    chi_sqr_avg = np.sum(diff_avg / (error_f150**2))
    #print("Chi-squared for average: ", chi_sqr_avg)      
         
    #plot
    plt.clf()
    plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='s', label='Bin Flux', c='r', zorder=1)
    plt.errorbar(eta_phase, res, yerr=error_f150, fmt='o', label='Flux', zorder=0,alpha=0.5)
    plt.axhline(y=res_avg, linestyle='--', label='Average Flux', zorder=0)      
    plt.plot(x,y_fit, label='Fitted Ftn.')     
    
    if pickle is not False:
      data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Phase Number': phases, 'Residuals': res, 'Error': error_f150, 'Measured':fluxs_f150,'Model (F Weights)':fWeights}
      filename = directory + "phase_curve_data_" + name + "_" + arr + "_" + freq +".pk"
      outfile = open(filename, 'wb')
      pk.dump(data_dict, outfile)
      outfile.close()             
    
  elif freq == "f220":
    #get f220
    infile_f220 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_" + arr + "_f220.pk", 'rb')
    dict_f220 = pk.load(infile_f220)
    infile_f220.close()  

    #get data
    fluxs_f220 = dict_f220['Flux']
    error_f220 = dict_f220['Error']
    fWeights = dict_f220['F']
    obj = dict_f220['Name']
    time_f220 = dict_f220['Time']
    
    #get phase numbers
    phi = rot_phase(obj, time_f220) #rotation phase numbers
    phi = np.array(phi)    
    theta, ignore_T = orb_phase(obj, time_f220) #orbital phase numbers
    theta = np.array(theta)    
    alpha = sunang_phase(obj, time_f220) #solar phase angle numbers
    alpha = np.array(alpha)          
    
    res = [(fluxs_f220[f] - fWeights[f]) for f in range(len(fluxs_f220))]
    low_times = [time_f220[r] for r in range(len(time_f220)) if res[r] < -700]
    t = Time(low_times,format='mjd')
    print(t.jyear)
    res_avg = mean(res)    
    
    eta = phi + theta + alpha
    eta_phase = eta % 1
    
    plt.clf()

    #binning
    ave_var, err_prop, phase_bins = inv_var_weight(n,error_f220,eta_phase,res)
      
    #sin fit
    eta_phase, res, error_f220 = zip(*sorted(zip(eta_phase,res,error_f220)))    
    params, params_covariance = optimize.curve_fit(fit_sin, eta_phase, res, sigma=error_f220)   
    perr = np.sqrt(np.diag(params_covariance))   
    x = np.arange(0,1,0.01)
    y_fit = fit_sin(x, params[0], params[1], params[2])      
      
    #chi-square sin
    diff = (fit_sin(eta_phase, params[0], params[1], params[2]) - res)**2
    error_f220 = np.array(error_f220)
    chi_sqr = np.sum(diff / (error_f220**2))
    dof = len(res) - 3
    red_chi_sqr = chi_sqr / dof
    diff = (fit_sin(eta_phase, params[0], params[1], params[2]) - res)**2
    error_f220 = np.array(error_f220)
    chi_sqr = np.sum(diff / (error_f220**2))
    dof = len(res) - 3
    red_chi_sqr = chi_sqr / dof
    if abs(params[0]/perr[0]) > 5:
      print('POSSIBLE DETECTION')    
    print(name + "_" + arr + "_" + freq, chi_sqr, red_chi_sqr, params[0], perr[0], abs(params[0]/perr[0]), chi2.sf(chi_sqr, dof))  
       
    #print(name, freq)              
    #print("Chi-squared for fit: ", chi_sqr)
    #print("DOF: ", dof) 
    #print("Reduced chi-squared for fit: ", red_chi_sqr)
    #print("PTE for fit: ", chi2.sf(chi_sqr, dof))
    #print("Amplitude: ", params[0], "with uncertainty: ", perr[0])
      
    #chi-square average     
    diff_avg = (res_avg - res)**2
    chi_sqr_avg = np.sum(diff_avg / (error_f220**2))
    #print("Chi-squared for average: ", chi_sqr_avg)      
      
    #plot
    plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='s', label='Bin Flux', zorder=1)
    plt.errorbar(eta_phase, res, yerr=error_f220, fmt='o', label='Flux', zorder=0,alpha=0.5)
    plt.axhline(y=res_avg, linestyle='--', label='Average Flux', zorder=0)      
    plt.plot(x,y_fit, label='Fitted Ftn.')      
    
    if pickle is not False:
      data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Phase Number': phases, 'Residuals': res, 'Error': error_f220, 'Measured':fluxs_f220,'Model (F Weights)':fWeights}
      filename = directory + "phase_curve_data_" + name + "_" + arr + "_" + freq +".pk"
      outfile = open(filename, 'wb')
      pk.dump(data_dict, outfile)
      outfile.close()            
      
  else:
    print("No data for this frequency. Please select from 'f090, f150, f220'") 
  
  plt.tick_params(direction='in')
  plt.xlabel("Phase Number")
  plt.ylabel("Flux Residual (mJy)")
  plt.legend(loc='best')
  
  if show is not False:
    plt.show()
        
  if directory is not None:
    plt.savefig(directory + "{name}_phase_curve_{freq}.pdf".format(name=obj, freq=freq))  
    
  return res
  
def vesta_phases(name, n, arr, freq, show = False, directory = None, pickle=False):
  '''
  Inputs:
    name, type: string, name of object we want phase curve for
    n, type: integer, number of bins for inverse-weighted average
    freq, type: string, frequency we want to look at
    show, type: boolean, if true, display light curve after calling lcurve    
    directory, type: string, optionally save file in directory
    pickle, type: boolean, optionally pickle phase data
  Output:
    Phase curve of object given a period
    also returns residuals for phase
    Incorporates rotational, orbital period
    solar phase angle in binning
  '''
  if freq == "f090":
    #infile_f090 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_" + arr + "_f090.pk", 'rb')
    infile_f090 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_all" + "_f090.pk", 'rb')
    dict_f090 = pk.load(infile_f090)
    infile_f090.close()

    #get data
    fluxs_f090 = dict_f090['Flux']
    error_f090 = dict_f090['Error']
    fWeights = dict_f090['F'] #Weighting
    obj = dict_f090['Name']
    time_f090 = dict_f090['Time']
    
    #get phase numbers
    phi = rot_phase(obj, -time_f090) #rotation phase numbers
    phi = np.array(phi)    
    theta, ignore_T = orb_phase(obj, time_f090) #orbital phase numbers
    theta = np.array(theta)    
    alpha = sunang_phase(obj, time_f090) #solar phase angle numbers
    #alpha = np.array(alpha)          
    
    res = [(fluxs_f090[f] - fWeights[f]) for f in range(len(fluxs_f090))]
    res_avg = mean(res)    
    #phi = -1*phi
    
    #eta = np.add(phi,theta,alpha)
    #eta_phase = eta % 1
    
    #binning
    ave_var, err_prop, phase_bins = inv_var_weight(n,error_f090,phi,res)
      
    #sin fit
    phi, res, error_f090 = zip(*sorted(zip(phi,res,error_f090)))
    params, params_covariance = optimize.curve_fit(vesta_fit, phi, res, sigma=error_f090)   
    perr = np.sqrt(np.diag(params_covariance))   
    x = np.arange(0,1,0.01)
    y_fit = vesta_fit(x, params[0], params[1], params[2], params[3], params[4])
      
    #chi-square sin
    diff = (vesta_fit(phi, params[0], params[1], params[2], params[3], params[4]) - res)**2
    error_f090 = np.array(error_f090)
    chi_sqr = np.sum(diff / (error_f090**2))
    dof = len(res) - 5
    red_chi_sqr = chi_sqr / dof
    print(name + "_" + arr + "_" + freq, chi_sqr, red_chi_sqr, chi2.sf(chi_sqr, dof))        
    #name arr freq chi_square reduced_chi_square pte
      
    #chi-square average     
    diff_avg = (res_avg - res)**2
    chi_sqr_avg = np.sum(diff_avg / (error_f090**2))
    #print("Chi-squared for average: ", chi_sqr_avg)            
      
    #plot      
    plt.clf()    
    plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='s', label='Bin Flux', zorder=1)
    plt.errorbar(phi, res, yerr=error_f090, fmt='o', label='Flux', zorder=0,alpha=0.5)
    plt.axhline(y=res_avg, linestyle='--', label='Average Flux', zorder=0)
    plt.plot(x,y_fit, label='Fitted Ftn.')
    
    if pickle is not False:
      data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Phase Number': phi, 'Residuals': res, 'Error': error_f090, 'Measured':fluxs_f090,'Model (F Weights)':fWeights}
      filename = directory + "phase_curve_data_" + name + "_" + arr + "_" + freq +".pk"
      outfile = open(filename, 'wb')
      pk.dump(data_dict, outfile)
      outfile.close()        
    
  elif freq == "f150":
    plt.rcParams.update({'font.size': 16})
    #get f150
    infile_f150 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_" + arr + "_f150.pk", 'rb') 
    #infile_f150 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_all" + "_f150.pk", 'rb')
    dict_f150 = pk.load(infile_f150)
    infile_f150.close()  
  
    #get data
    fluxs_f150 = dict_f150['Flux']
    error_f150 = dict_f150['Error']
    fWeights = dict_f150['F']
    obj = dict_f150['Name']
    time_f150 = dict_f150['Time']
    
    #get phase numbers
    phi = rot_phase(obj, time_f150) #rotation phase numbers
    phi = np.array(phi)    
    #theta, ignore_T = orb_phase(obj, time_f150) #orbital phase numbers
    #theta = np.array(theta)    
    #alpha = sunang_phase(obj, time_f150) #solar phase angle numbers     
    
    res = [(fluxs_f150[f] - fWeights[f]) for f in range(len(fluxs_f150))]
    res_avg = mean(res)    
    
    #eta = np.add(phi,theta,alpha)
    #eta_phase = eta % 1    
    
    #binning
    ave_var, err_prop, phase_bins = inv_var_weight(n,error_f150,phi,res)    
      
    #sin fit
    phi, res, error_f150 = zip(*sorted(zip(phi,res,error_f150)))
    params, params_covariance = optimize.curve_fit(vesta_fit, phi, res, sigma=error_f150)   
    perr = np.sqrt(np.diag(params_covariance))   
    x = np.arange(0,1,0.01)
    y_fit = vesta_fit(x, params[0], params[1], params[2], params[3], params[4])      
      
    #chi-square sin
    diff = (vesta_fit(phi, params[0], params[1], params[2], params[3], params[4]) - res)**2
    error_f150 = np.array(error_f150)
    chi_sqr = np.sum(diff / (error_f150**2))
    dof = len(res) - 5
    red_chi_sqr = chi_sqr / dof
  
    print(name + "_" + arr + "_" + freq, chi_sqr, red_chi_sqr, chi2.sf(chi_sqr, dof))
      
    #chi-square average     
    diff_avg = (res_avg - res)**2
    chi_sqr_avg = np.sum(diff_avg / (error_f150**2))
    #print("Chi-squared for average: ", chi_sqr_avg)      
         
    #plot
    plt.clf()
    plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='s', label='Bin Flux', c='r', zorder=1)
    plt.errorbar(phi, res, yerr=error_f150, fmt='o', label='Flux', zorder=0,alpha=0.5)
    plt.axhline(y=res_avg, linestyle='--', label='Average Flux', zorder=0)      
    plt.plot(x,y_fit, label='Fitted Ftn.')     
    
    if pickle is not False:
      data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Phase Number': phi, 'Residuals': res, 'Error': error_f150, 'Measured':fluxs_f150,'Model (F Weights)':fWeights}
      filename = directory + "phase_curve_data_" + name + "_" + arr + "_" + freq +".pk"
      outfile = open(filename, 'wb')
      pk.dump(data_dict, outfile)
      outfile.close()             
    
  elif freq == "f220":
    #get f220
    infile_f220 = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_" + arr + "_f220.pk", 'rb')
    dict_f220 = pk.load(infile_f220)
    infile_f220.close()  

    #get data
    fluxs_f220 = dict_f220['Flux']
    error_f220 = dict_f220['Error']
    fWeights = dict_f220['F']
    obj = dict_f220['Name']
    time_f220 = dict_f220['Time']
    
    #get phase numbers
    phi = rot_phase(obj, time_f220) #rotation phase numbers
    phi = np.array(phi)    
    #theta, ignore_T = orb_phase(obj, time_f220) #orbital phase numbers
    #theta = np.array(theta)    
    #alpha = sunang_phase(obj, time_f220) #solar phase angle numbers
    #alpha = np.array(alpha)          
    
    res = [(fluxs_f220[f] - fWeights[f]) for f in range(len(fluxs_f220))]
    res_avg = mean(res)    
    
    #eta = np.add(phi,theta,alpha)
    #eta_phase = eta % 1
    
    plt.clf()

    #binning
    ave_var, err_prop, phase_bins = inv_var_weight(n,error_f220,phi,res)
      
    #sin fit
    phi, res, error_f220 = zip(*sorted(zip(phi,res,error_f220)))    
    params, params_covariance = optimize.curve_fit(vesta_fit, phi, res, sigma=error_f220)   
    perr = np.sqrt(np.diag(params_covariance))   
    x = np.arange(0,1,0.01)
    y_fit = vesta_fit(x, params[0], params[1], params[2], params[3], params[4])      
      
    #chi-square sin
    diff = (vesta_fit(phi, params[0], params[1], params[2], params[3], params[4]) - res)**2
    error_f220 = np.array(error_f220)
    chi_sqr = np.sum(diff / (error_f220**2))
    dof = len(res) - 5
    red_chi_sqr = chi_sqr / dof
    print(name + "_" + arr + "_" + freq, chi_sqr, red_chi_sqr, chi2.sf(chi_sqr, dof))
    print('dof: ', dof)  
      
    #chi-square average     
    diff_avg = (res_avg - res)**2
    chi_sqr_avg = np.sum(diff_avg / (error_f220**2))
    #print("Chi-squared for average: ", chi_sqr_avg)      
      
    #plot
    plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='s', label='Bin Flux', zorder=1)
    plt.errorbar(phi, res, yerr=error_f220, fmt='o', label='Flux', zorder=0,alpha=0.5)
    plt.axhline(y=res_avg, linestyle='--', label='Average Flux', zorder=0)      
    plt.plot(x,y_fit, label='Fitted Ftn.')      
    
    if pickle is not False:
      data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Phase Number': phi, 'Residuals': res, 'Error': error_f220, 'Measured':fluxs_f220,'Model (F Weights)':fWeights}
      filename = directory + "phase_curve_data_" + name + "_" + arr + "_" + freq +".pk"
      outfile = open(filename, 'wb')
      pk.dump(data_dict, outfile)
      outfile.close()            
      
  else:
    print("No data for this frequency. Please select from 'f090, f150, f220'") 
  
  plt.tick_params(direction='in')
  plt.xlabel("Phase Number")
  plt.ylabel("Flux Residual (mJy)")
  plt.legend(loc='best')
  
  if show is not False:
    plt.show()
        
  if directory is not None:
    plt.savefig(directory + "{name}_phase_curve_{freq}.pdf".format(name=obj, freq=freq))  
    
  return res      
  
def jack_vesta_phases():
  '''
  Inputs:
    
  Output:
    Phase curve given Jack's light curves
    
  '''
  pa_dict = {'090':['pa5', 'pa6'], '150':['pa4', 'pa5', 'pa6'], '220':['pa4']}
  name = 'Vesta'
  pas = [ 'pa5', 'pa4']
  freq = '150'

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
  
      cur_phi = rot_phase(name, cur_times) #rotation phase numbers
      phi = np.hstack([phi, np.array(cur_phi)])
      cur_theta, ignore_T = orb_phase(name, cur_times) #orbital phase numbers
      theta = np.hstack([theta, np.array(cur_theta)])
      cur_alpha = sunang_phase(name, cur_times) #solar phase angle numbers
      alpha = np.hstack([alpha, np.array(cur_alpha)])
  
  eta = phi + theta 
  eta = eta % 1
  
  norm = np.mean(flux*F)
  flux = flux*F/norm
  err = err*F/norm  
  
  #fit
  x = np.arange(0,1,0.01)
  #y_fit = sin2model(x, 0.031, 0.203, 0.012, 0.492, 0.991)
  y_fit = fit_sin(x, 0.03, 0.181, 0.991)
  
  #binning
  ave_var, err_prop, phase_bins = inv_var_weight(20,err,eta,flux)
  
  
  #plot
  plt.errorbar(eta, flux, yerr=err, fmt='o', label='Flux', zorder=0,alpha=0.3)
  plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='.', label='Bin Flux', zorder=1, capsize=5, alpha=1)
  plt.plot(x,y_fit, label='Fitted Ftn.', alpha=1)   
  plt.tick_params(direction='in')
  plt.xlabel("Phase Number")
  plt.ylabel("Normalized Flux")
  #plt.title('Phase Curve of {name} at {freq}'.format(name=name,freq=freq))
  plt.legend(loc='best')       
  plt.show()
  plt.savefig("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/{name}_light_curve_all_{freq}.pdf".format(name=name, freq=freq))
  
########################################RUN STUFF BELOW##########################################################
#all_lcurves(show = True)
#ratios(show = True)
#lcurves("pa5", "f150", 3, show=True)
#one_lcurve("Vesta", "pa6", "f090", show=True)
#one_lcurve_fit(name='Vesta', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", show=True)
#ratios(show=True) 
#test_weights("Interamnia", "pa5", "f150", show=True)
#objs = ['Amphitrite','Antigone','Ariadne','Aspasia', 'Astraea','Athamantis','Ausonia','Bamberga','Davida','Dembowska','Egeria','Eleonora','Eunomia','Euphrosyne', 'Europa', 'Eurynome','Flora','Fortuna','Harmonia','Hebe','Herculina','Hermentaria','Hertha','Hesperia','Hygiea','Interamnia','Io','Irene','Julia','Kalliope','Kleopatra','Klotho','Laetitia','Leto','Lutetia','Melpomene','Mnemosyne','Nausikaa','Nemausa','Niobe','Pallas','Papagena','Parthenope','Patientia','Philomela','Pomona','Prokne','Proserpina','Rachele','Sappho','Tanete','Thetis','Undina','Urania','Vesta','Victoria','Zelinda']#fix 'Ganymed'
#freq = ['f090','f150','f220']
#arr = ['pa4','pa5','pa6']
#for i in objs:
#  for f in freq:
#    for a in arr:
#      try:
#        phases(i,5,a,f)
#      except FileNotFoundError:
#        continue
        

#phases('Vesta',5,'f090',show=True, directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/phase_curves/")
#pa{5,6}_f090, pa{4,5,6}_f150 and pa4_f220
#get_lcurves('pa5', 'f090', 200, directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/") 

#phases('Vesta',20,'pa6','f090',show=True)
#better_phases('Prokne', 20, 'pa4', 'f150', show = True, directory = None)
#better_phases('Vesta', 20, 'pa6', 'f090', show = True, directory = None)
#better_phases('Vesta', 20, 'pa4', 'f150', show = True, directory = None)
#better_phases('Vesta', 20, 'pa5', 'f150', show = True, directory = None)
#better_phases('Vesta', 20, 'pa6', 'f150', show = True, directory = None)
#better_phases('Vesta', 20, 'pa4', 'f220', show = True, directory = None)
#phases('Prokne',20,'pa5','f150', directory="/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/phase_curves/")
#phases('Prokne',20,'pa6','f150', directory="/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/phase_curves/")
#phases('Prokne',20,'pa4','f220', directory="/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/phase_curves/")

#one_lcurve('Prokne', 'pa4', 'f150', show = True)
#one_lcurve('Vesta', 'pa5', 'f090', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", pickle = True)
#one_lcurve('Vesta', 'pa6', 'f090', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", pickle = True)
#one_lcurve('Vesta', 'pa4', 'f150', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", pickle = True)
#one_lcurve('Vesta', 'pa5', 'f150', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", pickle = True)
#one_lcurve('Vesta', 'pa6', 'f150', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", pickle = True)
#one_lcurve('Vesta', 'pa4', 'f220', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", pickle = True)

#get_API(100,'f220')

#all_arrays("Pallas", "f150")
#get_alpha("Pallas", "pa4", "f150", show=True)
#period("Vesta", "pa5", "f220",5.34212766, show=True)

#one_lcurve_fit(name="Hebe", show=True)
#get_measured_flux('Vesta', 'pa5', '150')

#all_arrays('Prokne', 'f150', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", show = True, pickle=True)
#one_lcurve('Prokne', 'pa5', 'f150', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", show = True, pickle=True)

#vesta_phases('Vesta', 20, 'pa5', 'f090', show = True, directory = None)
#vesta_phases('Vesta', 20, 'pa6', 'f090', show = True, directory = None)
#vesta_phases('Vesta', 20, 'pa4', 'f150', show = True, directory = None)
#vesta_phases('Vesta', 20, 'pa5', 'f150', show = True, directory = None)
#vesta_phases('Vesta', 20, 'pa6', 'f150', show = True, directory = None)
#vesta_phases('Vesta', 20, 'pa4', 'f220', show = True, directory = None)
#better_phases('Prokne', 20, 'pa5', 'f150', show = True, directory = None)
#all_arrays('Vesta', 'f150', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", show = False,pickle=True)
#all_arrays('Pallas', 'f090', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", pickle=True)
#all_arrays('Pallas', 'f150', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", pickle=True)
#all_arrays('Pallas', 'f220', directory = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/", pickle=True)

jack_vesta_phases()