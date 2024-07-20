import numpy as np, ephem
import requests
import glob
import matplotlib.pyplot as plt
import pickle as pk

from pixell import utils, enmap, bunch
from scipy import optimize
from astropy.visualization import astropy_mpl_style
from astroquery.jplhorizons import Horizons
from astropy.time import Time
from lcurve import *

plt.style.use(astropy_mpl_style)


def in_box(box, point): #checks if points are inside box or not
	box   = np.asarray(box)
	point = np.asarray(point)
	point[1] = utils.rewind(point[1], box[0,1])
	# This assumes reverse RA axis in box
	return point[0] > box[0,0] and point[0] < box[1,0] and point[1] > box[1,1] and point[1] < box[0,1]

def make_box(point, rad): #making box
	box     = np.array([point - rad, point + rad])
	box[:,1]= box[::-1,1] # reverse ra
	return box

def filter_map(map, lknee=3000, alpha=-3, beam=0): #filtering map somehow (FFT)
	fmap  = enmap.fft(map)
	l     = np.maximum(0.5, map.modlmap())
	filter= (1+(l/lknee)**alpha)**-1
	if beam:
		filter *= np.exp(-0.5*l**2*beam**2)
	fmap *= filter
	omap  = enmap.ifft(fmap).real
	return omap

def calc_obs_ctime(orbit, tmap, ctime0):
	def calc_chisq(x):
		ctime = ctime0+x
		try:
			adata = orbit(ctime)
			mtime = tmap.at(adata[1::-1], order=1)
		except ValueError:
			mtime = 0
		return (mtime-ctime)**2
	ctime = optimize.fmin_powell(calc_chisq, 0, disp=False)+ctime0 
	err   = calc_chisq(ctime-ctime0)**0.5
	return ctime, err

def calc_sidereal_time(lon, ctime):
	obs      = ephem.Observer()
	obs.lon  = lon
	obs.date = utils.ctime2djd(ctime)
	return obs.sidereal_time()

def geocentric_to_site(pos, dist, site_pos, site_alt, ctime):
	"""Given a geocentric position pos[{ra,dec},...] and distance dist [...]
	in m, transform it to coordinates relative to the given site, with
	position pos[{lon,lat}] and altitude alt_site in m, returns the
	position observed from the site, as well as the distance from the site in m"""
	# This function isn't properly debugged. I should check the sign of
	# the sidereal time shift. But anyway, this function is a 0.2 arcmin
	# effect in the asteroid belt.
	sidtime    = calc_sidereal_time(site_pos[0], ctime)
	site_radec = np.array([site_pos[0]+sidtime*15*utils.degree,site_pos[1]])
	vec_site   = utils.ang2rect(site_radec)*(utils.R_earth + site_alt)
	vec_obj    = utils.ang2rect(pos)*dist
	vec_rel  = vec_obj-vec_site
	dist_rel = np.sum(vec_rel**2,0)**0.5
	pos_rel  = utils.rect2ang(vec_rel)
	return pos_rel, dist_rel

def get_desig(id_num:int) -> {int, str, float}:
  '''
  Input:
    id_num, designation number of object from Small-Body Database Lookup
  Output:
    desig, designation number of object
    name, name of object
    semimajor, semimajor axis of object

    Gets semimajor axis of object
  '''

  with open('/home/r/rbond/ricco/minorplanets/asteroids.pk', 'rb') as f:
    df = pk.load(f)
    name = df['name'][id_num]
    desig = df['designation'][id_num]
    semimajor = df['semimajor'][id_num]
  return desig, name, semimajor

def get_index(name:str) -> int:
  '''
    Inputs:
      name, name of object    
    Output:
      desig, index of object in asteroids.pk file
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

def get_theory(name:str, freq:str) -> dict:
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

def get_period(name:str) -> float:
  '''
    Input
      name, capitalized name of object we want rotation period of
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

def inv_var(data, variances) -> {float, float}:
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

def fit_sin(p:float, amp:float, offset:float, v:float):
  '''
    Inputs
      p, rotational phases of object
      amp, amplitude of fit
      offset, offset of fit
      v, vertical shift of fit      
    Output
      sin fit for phase curve
  '''

  p = np.array(p)
  return amp * np.sin(4*np.pi*p + offset) + v

def vesta_fit(p:float, A:float, B:float, C:float, D:float, E:float):
  p = np.array(p)
  return A*np.sin(2*np.pi*p + B) + C*np.sin(4*np.pi*p + D) + E

def sin2model(x:float, A1:float, phi1:float, A2:float, phi2:float, C:float):
    #A1, phi1, A2, phi2, C = p
    return A1 * np.sin(4*np.pi*x + phi1) + A2 * np.sin(2*np.pi*x + phi2) + C  

def inv_var_weight(n:int, errs, phases, res):
  '''
    Inputs
      n, number of weighted bins
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
      err_bin = []
      res_bin = []
      err_data_sqr = []
      
      err_bin = [errs[f] for f in range(len(phases)) if phases[f] > i and phases[f] <= (i+interval)]
      res_bin = [res[r] for r in range(len(phases)) if phases[r] > i and phases[r] <= (i+interval)]
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

def get_alpha(path:str, name:str, arr:str, freq:str, directory:str = None, show:bool = False, save:bool = False) -> bytes:
  '''
    Inputs:
   	  path, path to depth1 maps
      name, name of object we want
      arr, ACT array we want
      freq, frequency band
      directory, optionally save plots in specified directory
      show, optionally display plot right after calling get_alpha
      save, optionally save alpha and time data as pickle file
    
    Output:
      phase angle (deg) vs time for object 
  '''
    
  #Jack's maps
  path += name
  
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
      
def get_API(n:int, freq:str):
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

def orb_phase(name:str, times):
  '''
    Inputs
      name, name of asteroid
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

def rot_phase(name:str, times):
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

def sunang_phase(name:str, times):
  '''
    Inputs
      name, name of asteroid
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

def test_weighting():
  x = [1/4, 0.5, 3/4, 4/4]
  y = [1, 1, 1, 1]
  error = [0.25, 0.5, 0.5, 0.25]
  
  ave_var, err_prop, phase_bins = inv_var_weight(2,error,x,y)
  plt.errorbar(x,y, yerr=error, fmt='o', label='Data')
  plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='.', label='Binned Data')
  plt.legend(loc='best')
  plt.show()