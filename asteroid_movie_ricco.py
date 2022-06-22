import argparse, os
#query for files, asteroid, etc.
parser = argparse.ArgumentParser()
parser.add_argument("astinfo") #Ephemerides file
parser.add_argument("name") #name,string
parser.add_argument("ifiles", nargs="+") 
parser.add_argument("odir") #output directory
parser.add_argument("-r", "--rad", type=float, default=10.0)
parser.add_argument("-p", "--pad", type=float, default=30.0)
parser.add_argument("-t", "--tol", type=float, default=0.0)
parser.add_argument("-l", "--lknee", type=float, default=1500)
parser.add_argument("-a", "--alpha", type=float, default=3.5)
parser.add_argument("-b", "--beam",  type=float, default=0)
parser.add_argument("-v", "--verbose", default=1, action="count")
parser.add_argument("-e", "--quiet",   default=0, action="count")
args = parser.parse_args()
import numpy as np, ephem
from numpy.lib import recfunctions
from pixell import utils, enmap, bunch, reproject, colors, coordinates, mpi
from scipy import interpolate, optimize
import glob
import matplotlib.pyplot as plt

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
			mtime = tmap.at(adata[1::-1]*utils.degree, order=1)
		except ValueError:
			mtime = 0
		return (mtime-ctime)**2
	ctime = optimize.fmin_powell(calc_chisq, 0, disp=False)+ctime0 #[0] see original script
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

comm    = mpi.COMM_WORLD
verbose = args.verbose - args.quiet

r_thumb = args.rad*utils.arcmin
r_full  = r_thumb + args.pad*utils.arcmin

lknee    = args.lknee
alpha    = -args.alpha
beam     = args.beam*utils.fwhm*utils.arcmin
site     = coordinates.default_site
site_pos = np.array([site.lon,site.lat])*utils.degree
time_tol  = 60
time_sane = 3600*24

ifiles  = sum([sorted(utils.glob(ifile)) for ifile in args.ifiles],[]) 
#print("ifiles:", ifiles) #use map.fits files
info    = np.load(args.astinfo).view(np.recarray)#bunch.read 
# see https://phy-act1.princeton.edu/~snaess/actpol/ephemerides/
#[("ctime","d"),("ra","d"),("dec","d"),("r","d"),("rsun","d"),("ang","d")]

#orbdata = recfunctions.structured_to_unstructured(info[args.name.capitalize()]).T # [{mjd,ra,dec,dist},npoint] 
# Spline the orbit
#orbit   = interpolate.interp1d(utils.mjd2ctime(orbdata[0]), orbdata[1:], kind=3)
ctime = info["ctime"]
ra = info["ra"]
dec = info["dec"]
dist = info["r"] #distance from us, AU
sun_dist = info["rsun"]
ang_diam = info["ang"]

#orbdata = [ctime, ra, dec, r]

orbit = interpolate.interp1d(ctime, [ra, dec, dist], kind = 3) #test modeling original orbit code, think this might be correct
#not sure this is right
#orbit   = interpolate.interp1d(ctime, [utils.unwind(ra*utils.degree), dec*utils.degree, dist], kind=3) #modeling example usage from Sigurd 
#y = orbit(ctime)
#plt.scatter(y[0],y[1])
#plt.show()
#plt.close()


# For a typical asteroid I can be off by O(5 arcmin) by using the central
# ctime instead of the ctime when the object was actually observed.
# This is good enough for the quick bounding box check, but it's not
# good enough for centering the object properly. What I should do is
# read in the time map, and find at which point the asteroid's
# (ctime,ra,dec) intersects the time map. This could be done by
# making
# calc_chisq(ctime):
#  adata = orbit(ctime)
#  mtime = time_map.at(adata[1::-1]*utils.degree, order=1)
#  return (ctime-mtime)**2
# and minimizing it using fmin_powell


utils.mkdir(args.odir)

for fi in range(comm.rank, len(ifiles), comm.size):	  
  ifile = ifiles[fi]  
  infofile = utils.replace(ifile, "map.fits", "info.hdf")
  tfile    = utils.replace(ifile, "map.fits", "time.fits")
  ofname   = "%s/%s_%s" % (args.odir, args.name, os.path.basename(ifile))
  info     = bunch.read(infofile)
	#if info.t < 1606412474: continue
	# Get the asteroid coordinates
  ctime0   = np.mean(info.period)
  adata0   = orbit(ctime0)
  ast_pos0 = utils.rewind(adata0[1::-1]*utils.degree)
	#geo_pos  = utils.rewind(adata[:2]*utils.degree)
	#ast_pos, ast_dist = geocentric_to_site(geo_pos, adata[2]*utils.AU, site_pos, site.alt, ctime)
	#ast_pos   = ast_pos[::-1]
	#ast_dist /= utils.AU
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
  ctime, err = calc_obs_ctime(orbit, tmap, ctime0) #ctime is too small 10^-6 small
  print("ctime: ", ctime)  
  if err > time_tol or abs(ctime-ctime0) > time_sane:
    if verbose >= 2: 
      print(colors.white + message + " time" + colors.reset)
      continue
	# Now that we have the proper time, get the asteroids actual position
  adata    = orbit(ctime) #NEED TO FIX THIS + 1604016280
  ast_pos  = utils.rewind(adata[1::-1]*utils.degree)
	# optionally transform to topocentric here. ~0.1 arcmin effect
  thumb_box = make_box(ast_pos, r_thumb)
	# Read the actual data
  try:
	  imap = enmap.read_map(ifile, box=full_box)
  except (TypeError, FileNotFoundError):
	  print("Error reading %s. Skipping" % ifile)
	  continue
  if np.mean(imap.submap(thumb_box) == 0) > args.tol:
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
 
#test works
#eph   = np.load("/gpfs/fs0/project/r/rbond/sigurdkn/actpol/ephemerides/objects/Ceres.npy").view(np.recarray)
#orbit = interpolate.interp1d(eph.ctime, [utils.unwind(eph.ra*utils.degree), eph.dec*utils.degree, eph.r], kind=3)
#ctime = 1604016390
#pos   = orbit(ctime)
#print("ra: %8.3f, dec: %7.3f, r: %6.3f" % (utils.rewind(pos[0])/utils.degree, pos[1]/utils.degree, pos[2]))
# ra:  -24.336, dec: -23.866, r:  2.466

#Sigurd's format to run script:
# python asteroid_movie.py bright_ephem.hdf ceres "out/big/*/*pa?_f[01]??*map.fits" asteroids/ceres
#my format to run script:
# python asteroid_movie_ricco.py /gpfs/fs0/project/r/rbond/sigurdkn/actpol/ephemerides/objects/Ceres.npy ceres /home/r/rbond/sigurdkn/project/actpol/maps/depth1/release/15999/*pa5_f150_map.fits asteroids/ceres 
