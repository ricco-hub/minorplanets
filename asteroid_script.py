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
 
 
 
def get_maps(astinfo, name, arr, freq, directory = None, show = False, rad=10.0, pad=30.0, tol=0.0, lknee=1500, alpha=3.5, beam=0, verbose=2, quiet=0):
  '''
  Inputs: 
    astinfo, type: string, specifies path to ephemerides files
    name, type: string, name of asteroid
    arr, type: string, array ACT is on
    freq, type: string, frequency we want
    make sure array and frequency correspond to each other
    
    rad, type: float, ???
    pad, type: float, ???
    tol, type: float, ???
    lknee, type: float, ???
    alpha, type: float, ???
    beam, type: float, ???
    verbose, type: integer, ???
    quiet, type: integer, ????  
  
  Outputs:
    map.fits, rho.fits, kappa.fits, ivar.fits, type: fits file, file of object name on array arr at frequency freq saved in directory odir
  '''
  
  #path of depth1 maps
  depth_path = "/home/r/rbond/sigurdkn/project/actpol/maps/depth1/release/*"
  
  #make map.fits for each file in depth_path
  some_ifiles = [depth_path + "/*" + arr + "_" + freq + "*" + "map.fits"]      
      
  comm    = mpi.COMM_WORLD
  verbose = verbose - quiet

  r_thumb = rad*utils.arcmin
  r_full  = r_thumb + pad*utils.arcmin
      
  name = utils.replace(os.path.basename(astinfo), ".npy", "").lower()  
  alpha = -alpha
  lknee = lknee      
  beam     = beam*utils.fwhm*utils.arcmin
  site     = coordinates.default_site
  site_pos = np.array([site.lon,site.lat])*utils.degree
  time_tol  = 60
  time_sane = 3600*24
      
  ifiles  = sum([sorted(utils.glob(ifile)) for ifile in some_ifiles],[])
  info    = np.load(astinfo).view(np.recarray)
  orbit   = interpolate.interp1d(info.ctime, [utils.unwind(info.ra*utils.degree), info.dec*utils.degree, info.r, info.ang*utils.arcsec], kind=3)
  
  Name = name.capitalize()  
  
  ra_unhit = []
  dec_unhit = []
  
  ra_hit = []
  dec_hit = []
  t_hit = []
      
  #output directory
  #odir = "asteroids/" + name + "/" + arr + "/" + freq
  odir = "test/" + name + "/" + arr + "/" + freq      
  utils.mkdir(odir)

  for fi in range(comm.rank, len(ifiles), comm.size):		
    ifile    = ifiles[fi]  
    infofile = utils.replace(ifile, "map.fits", "info.hdf")
    tfile    = utils.replace(ifile, "map.fits", "time.fits")
    varfile    = utils.replace(ifile, "map.fits", "ivar.fits")    
    rhofile    = utils.replace(ifile, "map.fits", "rho.fits")
    kfile    = utils.replace(ifile, "map.fits", "kappa.fits")
            
    ofname   = "%s/%s_%s" % (odir, name, os.path.basename(ifile))
    varname   = "%s/%s_%s" % (odir, name, os.path.basename(varfile))    
    rhoname   = "%s/%s_%s" % (odir, name, os.path.basename(rhofile))
    kname   = "%s/%s_%s" % (odir, name, os.path.basename(kfile))        
        
    
    info     = bunch.read(infofile)    
      	# Get the asteroid coordinates
    ctime0   = np.mean(info.period)
    adata0   = orbit(ctime0)
    ast_pos0 = utils.rewind(adata0[1::-1])
    message  = "%.0f  %8.3f %8.3f  %8.3f %8.3f %8.3f %8.3f" % (info.t, ast_pos0[1]/utils.degree, ast_pos0[0]/utils.degree, info.box[0,1]/utils.degree, info.box[1,1]/utils.degree, info.box[0,0]/utils.degree, info.box[1,0]/utils.degree)
      	# Check if we're in bounds
    if not in_box(info.box, ast_pos0):
      if verbose >= 3:
        print(colors.lgray + message + " outside" + colors.reset)
        continue
      	#Ok, should try to read in this map. Decide on
      	# bounding box to read in
    full_box  = make_box(ast_pos0, r_full)
      	# Read it and check if we have enough hits in the area we want to use
    try:
      tmap = enmap.read_map(tfile, box=full_box)
      tmap[tmap!=0] += info.t
    except (TypeError, FileNotFoundError):
      print("Error reading %s. Skipping" % ifile)
      continue
      	# Break out early if nothing is hit
    if np.all(tmap == 0):
      if verbose >= 2: 
        print(colors.white + message + " unhit" + colors.reset)
        RA = ast_pos0[1]/utils.degree
        if RA > 360:
          while RA >= 360:
            RA -= 360
          ra_unhit.append(RA)
        elif RA < 0:
          while RA <= 0:
            RA += 360
          ra_unhit.append(RA)
        else:
          ra_unhit.append(RA)
        
        dec_unhit.append(ast_pos0[0]/utils.degree)
        continue
      	# Figure out what time the asteroid was actually observed 
    ctime, err = calc_obs_ctime(orbit, tmap, ctime0) 
    if err > time_tol or abs(ctime-ctime0) > time_sane:
      if verbose >= 2:
        print(colors.white + message + " time" + colors.reset) 
      continue
      	# Now that we have the proper time, get the asteroids actual position
    adata    = orbit(ctime) 
    ast_pos  = utils.rewind(adata[1::-1])
      	# optionally transform to topocentric here. ~0.1 arcmin effect
    thumb_box = make_box(ast_pos, r_thumb)
      	# Read the actual data
    try:
      imap = enmap.read_map(ifile, box=full_box)
      var = enmap.read_map(varfile, box=full_box)      
      rho = enmap.read_map(rhofile, box=full_box)
      kap = enmap.read_map(kfile, box=full_box)            
    except Exception as e: 
      print(colors.red + str(e) + colors.reset)
      continue
    if np.mean(imap.submap(thumb_box) == 0) > tol:
      if verbose >= 2: 
        print(colors.white + message + " unhit" + colors.reset)
        continue
      	# Filter the map
    wmap     = filter_map(imap, lknee=lknee, alpha=alpha, beam=beam)
      	# And reproject it
    omap = reproject.thumbnails(wmap, ast_pos, r=r_thumb)
    enmap.write_map(ofname, omap)
    
    vmap = reproject.thumbnails(var, ast_pos, r=r_thumb)
    enmap.write_map(varname, vmap)
    
    rmap = reproject.thumbnails(rho, ast_pos, r=r_thumb)
    enmap.write_map(rhoname, rmap)
    
    kmap = reproject.thumbnails(kap, ast_pos, r=r_thumb)
    enmap.write_map(kname, kmap)
    
    RA = ast_pos[1]/utils.degree
    if RA > 360:
      while RA >= 360:
        RA -= 360
      ra_hit.append(RA)
    elif RA < 0:
      while RA <= 0:
        RA += 360
      ra_hit.append(RA)
    else:
      ra_hit.append(RA)
    
    dec_hit.append(ast_pos[0]/utils.degree)
    t_hit.append(info.t)
                            
    if verbose >= 1: 
      print(colors.lgreen + message + " ok" + colors.reset)
      
  if show is not False:
  
    plt.figure()
    unhit = plt.scatter(ra_unhit, dec_unhit, marker='.', c="grey", label="Unhit")
    
    dates = [datetime.utcfromtimestamp(t).strftime('%Y-%m-%d') for t in t_hit]
    hit = plt.scatter(ra_hit, dec_hit, c=mdates.date2num(dates), marker="o", label = "Hit")
    
    cb = plt.colorbar(label="Date")
    loc = mdates.AutoDateLocator()
    cb.ax.yaxis.set_major_locator(loc)
    cb.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.title("Path of {name}".format(name=Name))
    plt.legend()
    #plt.axis([180,190, 4,7])
    plt.show()
    plt.close()
    
  if directory is not None:
    plt.savefig(directory + "{name}_plot_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))                     
                   
def snr(name, arr, freq, directory = None, show = False):
  '''
  Inputs:
    name, type: string, name of object we're interested in
    arr, type: string, ACT array
    freq, type: string, frequency we want
    directory, type: string, directory to save the output file
    show, type: boolean, if true, display plot after calling flux
    
  Output:
    stack.fits, type: file, stack of snr_tot of object
    F weighted maps
  '''
  
  #path after running get_maps on depth1 maps  
  #path = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/" + name + "/" + arr + "/" + freq
  #temp 
  path = "/scratch/r/rbond/ricco/minorplanets/test/" + name + "/" + arr + "/" + freq  
  
  #get rho and kappa files
  rho_files = glob.glob(path + "/*rho.fits")
  kap_files = [utils.replace(r, "rho.fits", "kappa.fits") for r in rho_files]
  
  if len(rho_files) != 0:  
    #find time
    str_times = []
    t_start = len(path) + len(name) + 9
    t_end = t_start + 10
    for time in rho_files:
      str_times.append(time[t_start:t_end])  
    int_times = [int(t) for t in str_times]
    
    #get data we want: ra, dec, geocentric dist, solar dist
    eph = np.load("/gpfs/fs0/project/r/rbond/sigurdkn/actpol/ephemerides/objects/" + name.capitalize() + ".npy").view(np.recarray)
    orbit = interpolate.interp1d(eph.ctime, [utils.unwind(eph.ra*utils.degree), eph.dec*utils.degree, eph.r, eph.rsun], kind=3)
    
    rho_tot = 0
    kap_tot = 0
    for count, time in enumerate(int_times):
      pos = orbit(time)
      ra = utils.rewind(pos[0])/utils.degree
      dec = pos[1]/utils.degree
      r_us = pos[2]
      r_sun = pos[3]
      
      #F weighting
      F = (r_us**-2) * (r_sun**-0.5)
      
      
      #open files
      hdu_rho = fits.open(rho_files[count])
      hdu_kap = fits.open(kap_files[count])
      
      #get data from files
      data_rho = hdu_rho[0].data
      data_kap = hdu_kap[0].data
      
      #do math
      rho_tot += data_rho * F
      kap_tot += data_kap * (F**2)
      
    m_tot = rho_tot / kap_tot
    snr_tot = m_tot / kap_tot**0.5
      
    hdu = fits.PrimaryHDU(snr_tot)
    hdu.writeto("snr_tot.fits", overwrite = True)
      
    image_file = get_pkg_data_filename("snr_tot.fits")
    image_data = fits.getdata(image_file, ext = 0)
    
    Name = name.capitalize()
      
    plt.figure()
    plt.title("snr tot of {name} on array {arr} at {freq}".format(name=Name, arr=arr, freq=freq))
    plt.imshow(image_data[0,:,:])
    plt.colorbar()
      
    if show is not False:
      plt.show()
        
    if directory is not None:
      plt.savefig(directory + "{name}_snr_tot_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))
      
  else:
    print("No hits")
  
def flux_stack(name, arr, freq, directory = None, show = False):
  '''
  Inputs:
    name, type: string, name of object we're interested in
    arr, type: string, ACT array
    freq, type: string, frequency we want
    directory, type: string, directory to save the output file
    show, type: boolean, if true, display plot after calling flux
    
  Output:
    stack.fits, type: file, stack of snr or flux of object
    Rho and kappa maps without F weighting
  '''
  
  #path after running get_maps on depth1 maps  
  #path = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/" + name + "/" + arr + "/" + freq
  #temp 
  path = "/scratch/r/rbond/ricco/minorplanets/test/" + name + "/" + arr + "/" + freq  
  
  #get rho and kappa files
  rho_files = glob.glob(path + "/*rho.fits")
  kap_files = glob.glob(path + "/*kappa.fits")
    
  
  rho_tot = 0
  kap_tot = 0
  try:
    for i in range(len(rho_files)):
      #open files
      hdu_rho = fits.open(rho_files[i])
      hdu_kap = fits.open(kap_files[i])
      
      #get data from files
      data_rho = hdu_rho[0].data
      data_kap = hdu_kap[0].data
      
      #add
      rho_tot += data_rho
      kap_tot += data_kap
      
    #get flux
    if len(rho_files) != 0:
      flux = rho_tot / kap_tot    
      flux_unt = kap_tot**(-0.5)
      snr = flux / flux_unt
      print("flux for {name}_{arr}_{freq}: ".format(name=name, arr=arr, freq=freq), flux[0, 40, 40])
      print("flux uncertainty for {name}_{arr}_{freq}: ".format(name=name, arr=arr, freq=freq), flux_unt[0,40,40])    
    
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
      
  except UnboundLocalError:
    print("No hits on {name}_{arr}_{freq}".format(name=name, arr=arr, freq=freq))
    
  #std calc
  #hdu_pic = fits.open('flux.fits')
  #data_pic = hdu_pic[0].data
  
  #delete cols
  #new_pic = np.delete(data_pic, [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50], axis=1)
  #delete rows
  #pic = np.delete(new_pic, [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50], axis=2)
  
  #var_calc = np.std(pic[0, :, :])
  #print("std for {name}_{arr}_{freq}: ".format(name=name, arr=arr, freq=freq), var_calc)
    

def make_image(path, name, arr, freq, directory = None):
  '''
  Input:
    path, type: string, path to map.fits file generated from running get_maps
    name, type: string, name of object to plot
    arr, type: string, array ACT is on
    freq, type: string, frequency of map
    directory, type: string, path to save image file
    
  Output: 
    Image, plot of individual name on array arr at frequency freq
  '''
  
  #get data from path
  image_file = get_pkg_data_filename(path)
  image_data = fits.getdata(image_file, ext=0)
  
  #create plot
  Name = name.capitalize()
  
  plt.figure()
  plt.title("Image of {name} on {arr} at {freq}".format(name=Name, arr=arr, freq=freq))
  plt.imshow(np.fliplr(image_data[0, :, :]))
  
  if directory is not None: 
    plt.savefig(directory + "{name}_image_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))
    
  plt.colorbar()
  plt.show()
  
def make_gallery(name, arr, freq, directory = None, show = False):
  '''
  Inputs:
    name, type: string, name of object we want
    arr, type: string, array ACT is on
    freq, type: string, frequency we are interested in
    directory, type: string, path to store image
    show, type: boolean, if true, display image after calling make_gallery
    
  Output:
    Image, gallery of depth1 images for object name on array arr at frequency freq
  '''
  
  #path after running get_maps on depth1 maps  
  #path = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/" + name + "/" + arr + "/" + freq
  path = "/scratch/r/rbond/ricco/minorplanets/test/" + name + "/" + arr + "/" + freq  
  
  #array of paths to map files
  map_files = glob.glob(path + "/*map.fits") 
    
  all_image_files = []    
  try:  
    for files in map_files:
      image_file = get_pkg_data_filename(files)
      all_image_files.append(image_file)   
      
    #print(len(all_image_files))
    count = 0
    #based on length of all_image_files
    #proabably a better way to extract dimensions
    #for rows and cols
    rows = 8
    cols = 14
    fig, axarr = plt.subplots(rows, cols, figsize=(10,10))
  
    #false if there are more hits than gallery size
    print((rows*cols) > len(all_image_files))  
    
    #plot at each index in all_image_files
    for i in range(rows):
      for j in range(cols):
        if count >= len(all_image_files):
          break            
     
        ax = axarr[i,j]
        
        #build image        
        image_data = fits.getdata(all_image_files[count], ext=0)
        count += 1     
        
        #plot
        #should change color scale based on object(s) we're looking at (vmin=-2000, vmax=8000, for Ceres)
        im = ax.imshow(np.fliplr(image_data[0, :, :]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)               
      
    plt.colorbar(im, ax=axarr[:, :])
    if show is not False:  
      plt.show()
  
    if directory is not None:
      plt.savefig(directory + "{name}_gallery_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))  
      
  except UnboundLocalError:
    print("No hits")      
    
def ivar_stack(name, arr, freq, directory = None, show = False):
  '''
  Inputs:
    name, type: string, name of object we want
    arr, type: string, array ACT is on
    freq, type: string, frequency we are interested in
    directory, type: string, path to store image
    show, type: boolean, if true, display stack after calling make_stack
    
  Output:
    Image, stack of depth1 images for object name on array arr at frequency freq
  '''
  
  #path after running get_maps on depth1 maps  
  path = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/" + name + "/" + arr + "/" + freq    
  
  #array of paths to ivar and map files
  map_files = glob.glob(path + "/*map.fits")
  ivar_files = glob.glob(path + "/*ivar.fits")

  stack = 0
  wts = 0    
  num = 0
  try:
    for i in range(len(map_files)):
      if len(map_files) != len(ivar_files):
        print("ivar files does not equal map files")
        break
        
      #open files
      hdu_map = fits.open(map_files[i])
      hdu_ivar = fits.open(ivar_files[i])
        
      #get data from files
      data_map = hdu_map[0].data
      data_ivar = hdu_ivar[0].data  
        
      #weights
      num += data_map[0,:,:] * data_ivar 
      wts += data_ivar
        
    ans = num / wts
    hdu = fits.PrimaryHDU(ans)
    hdu.writeto('stack.fits', overwrite = True)
      
    image_file = get_pkg_data_filename('stack.fits')  
    image_data = fits.getdata(image_file, ext=0)  
                                  
    Name = name.capitalize()
    
    plt.figure()
    plt.title("Stack of {name} on array {arr} at {freq}".format(name=Name, arr=arr, freq=freq))
    plt.imshow(image_data)  
    plt.colorbar()    
      
    if show is not False:  
      plt.show()
    
    if directory is not None:
      plt.savefig(directory + "{name}_stack_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))  
      
  except UnboundLocalError:
    print("No hits")    
    
def lcurve(name, arr, freq, directory = None, show = False):
  '''
  Inputs:
    name, type: string, name of object
    arr, type: arr, ACT array
    freq, type: freq, frequency we want
    directory, type: string, optionally save file in diretory
    show, type: boolean, if true, display light curve after calling lcurve
  
  Outputs:
    figure, creates light curve for object based on hits after running get_maps
    also plots F weighting
  '''      
  
  #path after running get_maps on depth1 maps
  #temp
  path = "/scratch/r/rbond/ricco/minorplanets/test/" + name + "/" + arr + "/" + freq
  
  #get rho and kappa files
  rho_files = glob.glob(path + "/*rho.fits") 
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
    eph = np.load("/gpfs/fs0/project/r/rbond/sigurdkn/actpol/ephemerides/objects/" + name.capitalize() + ".npy").view(np.recarray)
    orbit = interpolate.interp1d(eph.ctime, [utils.unwind(eph.ra*utils.degree), eph.dec*utils.degree, eph.r, eph.rsun], kind=3)
    
    flux_data = []
    err_data = []
    times_data = []
    Fs = []
    for count, t in enumerate(int_times):
      #get distances
      pos = orbit(t)
      r_us = pos[2]
      r_sun = pos[3]
      
      #F weighting
      F = (r_us**-2) * (r_sun**-0.5)
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
      
      err = data_kap**(-0.5)
      err_data.append(err[0,40,40]) 
      
      times_data.append(t)
    
    mjd_date = utils.ctime2mjd(times_data)
    Name = name.capitalize()
    
    fig, host = plt.subplots(figsize=(5,5))
    
    par1 = host.twinx()
    host.set_xlabel("Time (MJD)")
    host.set_ylabel("Flux (mJy)")
    par1.set_ylabel("F (Arbitrary)")
    
    p1 = host.errorbar(mjd_date, flux_data, yerr=err_data, fmt='o', capsize=4, label='Flux')
    p2 = par1.scatter(mjd_date, Fs, label='F weighting', c='r')
    
    lns = [p1, p2]
    host.legend(handles = lns, loc='best')
    
    plt.title("Light curve of {name}".format(name=Name))
    
    if show is not False:
      plt.show()
      
    if directory is not None:
      plt.savefig(directory + "{name}_light_curve_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))
  
  else:
    print("No hits")
  
  
################################***CALL THINGS BELOW***##################################################################################################
#path to desired object
#astinfo = "/home/r/rbond/sigurdkn/project/actpol/ephemerides/objects/Ceres.npy"

#make sure to update with same name and astinfo
#make sure to change odir to correct name and freq
#get_maps(astinfo, "ceres", "pa5", "f150", show=True)

#make_image("/home/r/rbond/ricco/minorplanets/asteroids/ceres/pa5/f150/ceres_depth1_1623042128_pa5_f150_map.fits", "ceres", "pa5", "f150")

#also update save directory (if applicable)
#make_gallery("ceres", "pa5", "f090", show=True)

#ivar_stack("ceres", "pa5", "f150", show=True)  

#flux_stack("ceres", "pa5", "f150")

#snr("ceres", "pa5", "f150", show=True)

#lcurve("bamberga", "pa5", "f150", show=True) #vesta_pa4_f220, hygiea_pa5_f150, pallas_pa5_f150