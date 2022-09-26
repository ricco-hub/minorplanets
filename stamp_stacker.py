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

def get_desig(id_num):
  '''
  Input:
    id_num, type: integer, designation number of object from Small-Body Database Lookup
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
  
def flux_stack(arr, freq, directory = None, show = False):
  '''
  Inputs:
    arr, type: string, ACT array
    freq, type: string, frequency we want
    directory, type: string, directory to save the output file
    show, type: boolean, if true, display plot after calling flux
    
  Output:
    stack.fits, type: file, stack of snr or flux of object
    Rho and kappa maps with F weighting
  '''
  for i in range(3):
    #get semimajor axis and name
    ignore_desig, name, semimajor_sun = get_desig(i)
    ignore_desig, ignore_name, semimajor_earth = get_desig(i)
  
    #Jack's maps  
    path = "/scratch/r/rbond/jorlo/actxminorplanets/sigurd/asteroids/" + name  
    
    #get rho and kappa files
    rho_files = glob.glob(path + "/*" + arr + "_" + freq + "_" + "rho.fits")
    kap_files = [utils.replace(r, "rho.fits", "kappa.fits") for r in rho_files]
      
    rho_tot = 0
    kap_tot = 0
    
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
      
      for count, t in enumerate(int_times):
        #get distances
        pos = orbit(t)
      
        d_sun_0, d_earth_0 = semimajor_sun, semimajor_earth 
      
        ignore_ra, ignore_dec, delta_earth, delta_sun, ignore_ang = pos      
      
        #F weighting
        F = (d_sun_0)**2 * (d_earth_0)**2 / ((delta_earth)**2*(delta_sun)**2) * 791.92334
      
        #open files
        hdu_rho = fits.open(rho_files[count])
        hdu_kap = fits.open(kap_files[count])
      
        #get data from files
        data_rho = hdu_rho[0].data
        data_kap = hdu_kap[0].data      
      
        #add and F weight
        rho_tot += data_rho * F
        kap_tot += data_kap * (F**2)
      
      if len(rho_files) != 0:
        flux = rho_tot / kap_tot    
        flux_unt = kap_tot**(-0.5)
        snr = flux / flux_unt
        #print("flux for {name}_{arr}_{freq}: ".format(name=name, arr=arr, freq=freq), flux[0, 40, 40])
        #print("flux uncertainty for {name}_{arr}_{freq}: ".format(name=name, arr=arr, freq=freq), flux_unt[0,40,40])    
    
        hdu = fits.PrimaryHDU(snr)
        hdu.writeto('snr.fits', overwrite = True)
      
        image_file = get_pkg_data_filename('snr.fits')
        image_data = fits.getdata(image_file, ext = 0)
    
        Name = name.capitalize()
    
        plt.figure()
        plt.title("snr of {name} on array {arr} at {freq}".format(name=Name, arr=arr, freq=freq))
        plt.imshow(image_data[0,:,:])
        plt.colorbar()
    
      else:
        print("No hits on {name}_{arr}_{freq}".format(name=name, arr=arr, freq=freq))
    
      if show is not False:
        plt.show()
      
      if directory is not None:
        plt.savefig(directory + "{name}_snr_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))
      
    else:
      print("No hits on {name}_{arr}_{freq}".format(name=name, arr=arr, freq=freq))

flux_stack("pa5", "f150", show=True)