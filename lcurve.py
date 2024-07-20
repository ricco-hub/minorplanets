import numpy as np, ephem
import glob
import matplotlib.pyplot as plt
import pickle as pk
import ephem
import seaborn as sns
import requests
import fsspec # optional depending on the astropy dependencies installed

from pixell import utils, enmap, bunch
from astropy.visualization import astropy_mpl_style
from astroquery.jplhorizons import Horizons
from astropy.time import Time
from astropy.io import fits
from palettable.colorbrewer.sequential import Blues_9
from astro_utils import *

plt.style.use(astropy_mpl_style)
sns.set_theme(style='ticks')


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

        except:
          continue                   
        
    night_mjd = utils.ctime2mjd(night_times)    
    
    #find min and max times for astroquery
    start = Time(min(astroQ_times), format='unix')
    end = Time(max(astroQ_times), format='unix')    
    start_iso = start.iso
    end_iso = end.iso
    
    #get astroquery data to create line of best fit
    obj_earth = Horizons(id=name, epochs={'start':start_iso, 'stop':end_iso, 'step':'1d'},id_type='asteroid_name')#location='W99', 
    eph = obj_earth.ephemerides()
    
    time = eph['datetime_jd']
    mjd_times = utils.jd2mjd(time)    
    delta_earth_best = eph['delta'] #earth-asteroid distance
    delta_sun_best = eph['r'] #sun-asteroid distance
    alpha_best = eph['alpha'] 
    
    try:
      best_F = (delta_earth_best**(-2) * delta_sun_best**(-1/2)*10**(-0.004*alpha_best)) * ref_flux
    
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
    
    except (TypeError, UnboundLocalError):
      pass      
    
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

def lcurves(arr:str, freq:str, n:int, directory:str = None, show:bool = False, pickle:bool = False):
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
    one_lcurve(name, arr, freq, directory, show, pickle)
  
def lcurve_data(name:str, arr:str, freq:str):
  '''
    Inputs:
      name, name of object
      arr, ACT array
      freq, frequency
    Outputs:
      times, float array of MJD observation times
      flux, asteroid thermal emission in mJy
      error, error of asteroid thermal emission in mJy
  '''

  # remove f in f090, f150, f220
  freq.replace('f', '')
  if freq == '090':
    freq = '90'

  name = '{}_lc_{}_{}'.format(name, arr, freq) 
  s3_path = 's3://cornell-acteroids/' + name + '.fits'
  with fits.open(s3_path, fsspec_kwargs={"anon": True}) as hdul:  
    data = hdul[1].data  

  times = data['Time']
  flux = data['Flux']
  error = data['FluxUncertainty']    

  return times, flux, error