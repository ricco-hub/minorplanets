from pixell import enmap,utils, reproject, enplot
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from scipy.interpolate import interp1d
import math
import pandas as pd
import pickle as pk
import h5py
import time
from astropy import wcs
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from astropy.io import fits
from astropy.table import QTable
import re
from numba import jit


@jit(nopython=True)
def strange_check(strange_handles, target):
    #A small helper function that finds the index of our target plannet. Due to a difference 
    #in naming scheme we unfortunately can't just use np.where or something like that
    for i in range(len(strange_handles)):
        if strange_handles[i] == target:
            break
            
    return i

def tnoStamp(ra, dec, imap, width = 0.5):
    #Takes an individual stamp of a map at the requested ra/dec. We form the ratio later  
    #frhs is a matched filter map and kmap is the inverse variance per pixel
    #both maps are at a ~3 day cadence

    #Inputs: ra, dec in degrees, j2000
    #kmap or frhs, described above. They must have the same wcs
    #but if you're using this code and sigurd's maps they will
    #width, the desired width of the stamp in degrees

    #Output: a stamp, centered on ra/dec, with width width, where
    #each pixel in the map records the S/N. This stamp is for one
    #object, one 3 day map. These must then be stacked for each object
    #and then the objects can be stacked together

    #Find the pixel 
    coords = np.deg2rad(np.array((dec,ra)))
    ypix,xpix = enmap.sky2pix(imap.shape,imap.wcs,coords)

    
   
    
    #nans are formed when try to form S/n for pixels with no hits
    #I just set the S/N to 0 which I think is safe enough
    imap[~np.isfinite(imap)] = 0

    #Reproject will attempt to take a stamp at the desired location: if it can't
    #for whatever reason just return None. We don't want it throwing errors
    #while on the wall, however the skips should probably be checked after
    try:
        stamp = reproject.postage_stamp(imap, ra, dec, width*60, 0.5)
    except:
        return None
    
    return stamp

class OrbitInterpolator:
    '''
    Constructs a class that can predict, using an interpolation scheme, the location of an object given an identifier and time
    '''
    def __init__(self, table):
        '''
        Requires a table generated from astroquery, querying JPL HORIZONS. The 
        interpolation is done automatically, but, of course, only works
        if the table is sampled densely enough and in the time range for which
        the positions were queried
        '''
        self.table = table
        self.targets = np.unique(table['targetname'])

        self._construct_dictionary()

    def _interpolate_radec(self, target):
        '''
        "Hidden" function that constructs the interpolations for each target
        '''        
        table = self.table[self.table['targetname'] == target]
        zero = np.min(table['datetime_jd'])
        ra_interp = interp1d(table['datetime_jd'] - zero, table['RA'])
        dec_interp = interp1d(table['datetime_jd'] - zero, table['DEC'])
        delta_interp = interp1d(table['datetime_jd'] - zero, table['delta'])

        return zero, ra_interp, dec_interp, delta_interp

    def _construct_dictionary(self):
        '''
        "Hidden" function that creates the look-up dictionary of targets for simplicity of usage
        '''
        self.obj_dic = {}
            
        for j,i in enumerate(self.targets):
            print(j)
            z, r, d, delta = self._interpolate_radec(i)
            self.obj_dic[i] = {}
            self.obj_dic[i]['zero'] = z
            self.obj_dic[i]['RA'] = r
            self.obj_dic[i]['DEC'] = d
            self.obj_dic[i]['delta'] = delta

    def get_radec_dist(self, target, time):
        '''
        Specifying a target name (see self.obj_dic.keys() for a list of targets) and a time (in JD), finds
        the interpolated RA and Dec for the objects
        '''
        time = time + 2400000.5
        
        t_intep = time - self.obj_dic[target]['zero']

        ra = self.obj_dic[target]['RA'](t_intep)
        dec = self.obj_dic[target]['DEC'](t_intep)
        dist = self.obj_dic[target]['delta'](t_intep)

        return ra, dec, dist

class QueryHorizons:
    '''
    Constructs a class that can query a bunch of positions for objects from JPL-Horizons given a set of times and an observatory location
    '''

    def __init__(self, time_start, time_end, observer_location, step = '1d'):
        '''
        Initialization function

        Arguments:
        - time_start: start time for the query, should be in MJD
        - time_end: end time for the query, should be in MJD
        - observer_location: location of the observer
        - step: time step for the ephemeris
        The simples way to get the observer location variable is via the
        list of IAU observatory codes: https://en.wikipedia.org/wiki/List_of_observatory_codes
        
        Custom locations are also accepted by JPL
        '''
        t_st = Time(time_start, format='mjd')
        self.time_start = t_st.utc.iso  
        t_en = Time(time_end, format='mjd')
        self.time_end = t_en.utc.iso 
        
        self.observer = observer_location

        self.step = step 


    def queryObjects(self, objects):
        '''
        Returns a table (and saves it on the object as well) for the provided list of objects
        '''
        self.table = [] 

        for i in objects:
            query = Horizons(id = i, location = self.observer, 
                epochs = {'start' : self.time_start, 'stop' : self.time_end, 'step' : self.step})

            eph = query.ephemerides()

            self.table.append(eph['RA', 'DEC', 'datetime_jd', 'targetname', 'delta', 'r'])
        
        self.table = tb.vstack(self.table)

        return self.table


def tnoStacker(oribits, obj):
    #Returns a stack over ~~~1~~~ objects orbit
    
    #Inputs, orbits, and OrbitInterpolator instance, which contains all the orits of interest, i.e. for multiple objects
    #obj, the name/id of the object of interest
    
    #Path to the maps. Probably shouldn't be hard coded
   
    print('stacking on ', str(obj))

    path = '/home/r/rbond/sigurdkn/scratch/actpol/planet9/20200801/maps/combined/'
    
    #Initialize stack/divisor
    kstack = 0
    fstack = 0
   
    
    #Get the index of the tno we're working with
    tno_hdu = fits.open('/project/r/rbond/jorlo/act_tnos/dist_h_y4.fits')
    tno_data = tno_hdu[1].data


    #we're now going to check each directory in the path, each of which corresponds to one
    #~3 day coadd
    for dirname in os.listdir(path=path):
        print('In dir ', dirname)
        try:
            with h5py.File(path + dirname +"/info.hdf", "r") as hfile:
                #Find the (rough) mjd center of the map
                mjd_cent = hfile["mjd"][()]
        except:
            print('no info file in ', dirname)
            continue
        #Get the ra/dec of the object, as well as delta, the geocentric distance to the object
        #We use geocentric as we're looking at the IR emission, which scales as r**2 the distance
        #from the obj to earth. If we were looking at reflected sunlight, we'd care about the heliocentric
        ra, dec, delta = orbits.get_radec_dist(obj, mjd_cent)

        #Get wcs info from the kmap for this 3day coadd: for sigurd's maps the kmap and 
        #frhs map wcs info will be the same
        hdu = fits.open(path + dirname + '/kmap.fits')
        w = wcs.WCS(hdu[0].header)
        
        #Find pixel corresponding to our ra/dec. I actually can no longer recall why
        #I did this and it doesn't seem to do anything so it could probably be
        #removed
        c = SkyCoord(ra, dec, unit="deg")
        x, y = w.world_to_pixel(c)           
        
        #Read in the maps and take the stamp using tnoStamp
        kmap = enmap.read_map(path + dirname + '/kmap.fits')
        frhs = enmap.read_map(path + dirname + '/frhs.fits')

        kstamp = tnoStamp(ra, dec, kmap)
        fstamp = tnoStamp(ra, dec, frhs)
        #See tnoStamp but if stamp is None it means something went wrong.
        #We also check to see if the maps are uniformly zero 
       
        if (kstamp is None) or (fstamp is None):
            continue
        if (not np.any(kstamp[0])) or (not np.any(fstamp[0])):
            continue
        #if np.any(np.isinf(stamp[0])):
        #    continue

        #Scale by the flux compared to the reference flux at 1AU
        flux_scale = (delta/1)**2 #Div one just a placeholder for ref scale
        
                    

        #Stack it up, divide and return
        fstack += fstamp[0]/flux_scale
        kstack += kstamp[0]/flux_scale
        #print(stack)
        

    flux_stack = fstack/kstack

    return flux_stack
tic = time.perf_counter()

#Set path to project dir
path = '/project/r/rbond/jorlo/act_tnos/indv_stamps/'

#Command line arg that tells us which tno we're looking at by index
#This makes it reasonably convenient to 'parallelize' this in a submission script
#but isn't how you want to do this if you're not running on a cluster
tno_index = int(sys.argv[1])

#We make the stacks along an individual objects orbit, and then the resulting
#stacked maps can be combined with whatever waiting we desire

#Load orbits
hdu = fits.open('/project/r/rbond/jorlo/act_tnos/y6_interp.fits')
if os.path.exists('y6_orbits.p'):
     orbits = pk.load(open('y6_orbits.p', 'rb'))
else:
    orbits = OrbitInterpolator(hdu[1].data)
    pk.dump(orbits, open('/scratch/r/rbond/jorlo/y6_orbits.p', 'wb')) 


#Get a list of names, removing duplicates in a way that has consistant ordering
names = hdu[1].data['targetname']
names = np.unique(names)
names.sort()

name = names[tno_index]

#Run the stacking code
stack = tnoStacker(orbits,name)

tno_dict = {'Name':name, 'stack':stack}

#Make the individual tno dir if it doesn't exist
dir_name = name.replace(' ', '_')
dir_name = dir_name[1:-1]

print(dir_name)

full_path = os.path.join(path, dir_name)
if not os.path.exists(full_path):
    os.makedirs(full_path)
    print('Full path made')

pk.dump(tno_dict, open(path+dir_name+'/{}.p'.format(dir_name), 'wb'))

plt.imshow(stack, vmax = 5)
plt.title(name)
plt.savefig(path+dir_name+'/{}.pdf'.format(dir_name))
plt.close()

toc = time.perf_counter()
time_min = round((toc-tic)/60,2)
print('Job {} ran in {:0.2f} min'.format(tno_index, time_min))

