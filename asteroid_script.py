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
 
 
 
def make_movie(astinfo, name, arr, freq, odir, rad=10.0, pad=30.0, tol=0.0, lknee=1500, alpha=3.5, beam=0, verbose=1, quiet=0):
  '''
  Inputs: 
    astinfo, type: string, specifies path to ephemerides files
    name, type: string, name of asteroid
    arr, type: string, array ACT is on
    freq, type: string, frequency we want
    make sure array and frequency correspond to each other
    odir, type: string, specify output directory
    
    rad, type: float, ???
    pad, type: float, ???
    tol, type: float, ???
    lknee, type: float, ???
    alpha, type: float, ???
    beam, type: float, ???
    verbose, type: ???, ???
    quiet, type: ???, ????
    verbose, type: ???, ???
    quiet, type: ???, ???
    #parser.add_argument("-v", "--verbose", default=1, action="count")
    #parser.add_argument("-e", "--quiet",   default=0, action="count")
    
  
  Output:
    map.fits, type: file, file of object name on array arr at frequency freq in directory odir
  '''
  
  #path of depth1 maps
  depth_path = "/home/r/rbond/sigurdkn/project/actpol/maps/depth1/release/"
  
  #make movie for each file in depth_path
  for i, dirname in enumerate(os.listdir(path = depth_path)):
    print("I'm working in dir ", dirname)
    try:
      some_ifiles = glob.glob(depth_path + dirname + "/*" + arr + "_" + freq + "_map.fits") 
      #print("ifiles ", ifiles)      
      
      comm    = mpi.COMM_WORLD
      verbose = verbose - quiet

      r_thumb = rad*utils.arcmin
      r_full  = r_thumb + pad*utils.arcmin
      
      alpha = -alpha
      beam     = beam*utils.fwhm*utils.arcmin
      site     = coordinates.default_site
      site_pos = np.array([site.lon,site.lat])*utils.degree
      time_tol  = 60
      time_sane = 3600*24
      
      ifiles  = sum([sorted(utils.glob(ifile)) for ifile in some_ifiles],[])
      info    = np.load(astinfo).view(np.recarray)
      orbit   = interpolate.interp1d(info.ctime, [utils.unwind(info.ra*utils.degree), info.dec*utils.degree, info.r, info.ang*utils.arcsec], kind=3)
      
      utils.mkdir(odir)

      for fi in range(comm.rank, len(ifiles), comm.size):		
      	ifile    = ifiles[fi]
      	infofile = utils.replace(ifile, "map.fits", "info.hdf")
      	tfile    = utils.replace(ifile, "map.fits", "time.fits")
      	ofname   = "%s/%s_%s" % (odir, name, os.path.basename(ifile))
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
      	# Ok, should try to read in this map. Decide on
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
      	except (TypeError, FileNotFoundError):
      		print("Error reading %s. Skipping" % ifile)
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
      	if verbose >= 1: 
      		print(colors.lgreen + message + " ok" + colors.reset)

    except:
      print("No map file in ", dirname)
      continue
   
def flux():
  pass 

def make_image(path, name, arr, freq, directory = None):
  '''
  Input:
    path, type: string, path to map.fits file
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
  plt.figure()
  plt.title("Plot of {name} on {arr} at {freq}".format(name=name, arr=arr, freq=freq))
  plt.imshow(image_data[0, :, :])
  
  if directory is not None: 
    plt.savefig(directory + "{name}_image_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))
    
  plt.colorbar()
  plt.show()
  
def make_gallery(name, arr, freq, directory = None):
  '''
  Inputs:
    name, type: string, name of object we want
    arr, type: string, array ACT is on
    freq, type: string, frequency we are interested in
    directory, type: string, path to store image
    
  Output:
    Image, gallery of depth1 images for object name on array arr at frequency freq
  '''
  
  path = "/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroid/" + name + "/" + freq
  
  for i, dirname in enumerate(os.listdir(path = path)):
    #array of paths to map files
    map_files = glob.glob(path + "/*" + arr + "_" + freq + "_map.fits") 
  
  all_image_files = []    
  for files in map_files:
    image_file = get_pkg_data_filename(files)
    all_image_files.append(image_file)
      
      
  count = 0
  #based on length of all_image_files
  #proabably a better way to extract dimensions
  #for rows and cols
  rows = 9
  cols = 15
  fig, axarr = plt.subplots(rows, cols, figsize=(9,6))
    
  #plot at each index in all_image_files  
  for i in range(rows):
    for j in range(cols):
      ax = axarr[i,j]
      
      #build image
      image_data = fits.getdata(all_image_files[count], ext=0)
      count += 1
      
      #plot
      im = ax.imshow(image_data[0, :, :], vmin=-2000, vmax=8000)
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      
  plt.colorbar(im, ax=axarr[:, :])
  plt.show()
  
  if directory is not None:
    fig.savefig(directory + "{name}_gallery_{arr}_{freq}.pdf".format(name=name, arr=arr, freq=freq))  
  
################################***RUN THINGS BELOW***##################################################################################################
#path to desired object
astinfo = "/gpfs/fs0/project/r/rbond/sigurdkn/actpol/ephemerides/objects/Ceres.npy"
#make sure to update with same name and astinfo
#make sure to change odir to correct name and freq
#make_movie(astinfo, "ceres", "pa5", "f090", "asteroid/ceres/f090")

#make_image("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroid/vesta/f090/vesta_depth1_1569982303_pa5_f090_map.fits", "Vesta", "pa5", "f090")

make_gallery("ceres", "pa5", "f150")