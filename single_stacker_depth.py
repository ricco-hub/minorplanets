#!/usr/bin/env python
# coding: utf-8

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
import astropy.table as tb 
from astropy.time import Time 
from astroquery.jplhorizons import Horizons
import glob

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
    #try:
    stamp = reproject.thumbnails(imap, [coords])
    #except:
    #    return None
    
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
        if type(time_start) == str: 
            t_st = Time(time_start, format='isot', scale='utc')
            
        else: 
            t_st = Time(time_start, format='mjd')
        self.time_start = t_st.utc.iso  
        if type(time_end) == str: 
            t_en = Time(time_end, format='isot', scale='utc')
            
        else: 
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


def tnoStacker(oribits_obj, obj, freq, arr):
    #Returns a stack over ~~~1~~~ objects orbit
    
    #Inputs, orbits, and OrbitInterpolator instance, which contains all the orbits of interest, i.e. for multiple objects
    #obj, the name/id of the object of interest
    #freq, the frequency the depth1 array is on    
    #array, the ACT array we are interested in    
    
    #Path to the maps. Probably shouldn't be hard coded
   
    print('stacking on ', str(obj))
    
    #depth-1 maps
    path = '/home/r/rbond/sigurdkn/project/actpol/maps/depth1/release/' 
    
    #Initialize stack/divisor
    kstack = 0
    fstack = 0
   
    
    #Get the index of the tno we're working with
    #tno_hdu = fits.open('/project/r/rbond/jorlo/act_tnos/dist_h_y4.fits')
    #tno_data = tno_hdu[1].data


    #we're now going to check each directory in the path, each of which corresponds to one
    #~3 day coadd                  
    for i, dirname in enumerate(os.listdir(path=path)):
        #if i>10: break   
        print("I'm working in dir ", dirname)                           
        try:     
            hdf_file = glob.glob(path + dirname + "/*" + "_" + arr + "_" + freq + "_info.hdf")                               
            #print(hdf_file)
            for h in hdf_file:                        
                with h5py.File(h, "r") as hfile:                                                                                       
                    #Find the (rough) mjd center of the map
                    unix_cent = hfile["t"][()] 
                    mjd_cent = Time(unix_cent, format='unix').mjd
                    isot_time = Time(unix_cent, format='unix').isot
                    #print("mjd_cent ", mjd_cent)                                                
                    #print("Readable time ", isot_time)                
                
                
                
        except:
            print('No info file in ', dirname)
            continue
        #Get the ra/dec of the object, as well as delta, the geocentric distance to the object
        #We use geocentric as we're looking at the IR emission, which scales as r**2 the distance
        #from the obj to earth. If we were looking at reflected sunlight, we'd care about the heliocentric
        ra, dec, delta = oribits_obj.get_radec_dist(obj, mjd_cent)

        #Get wcs info from the kmap for this 3day coadd: for sigurd's maps the kmap and 
        #frhs map wcs info will be the same
        ivar_file = glob.glob(path + dirname + "/*" + "_" + arr + "_" + freq + "_ivar.fits")
        for var in ivar_file:                             
          hdu = fits.open(var)            
          w = wcs.WCS(hdu[0].header) #might have to change this???
        
        #Read in the maps and take the stamp using tnoStamp
          kmap = enmap.read_map(var) 
        try:
            map_file = glob.glob(path + dirname + "/*" + "_" + arr + "_" + freq + "_map.fits")          
            for maps in map_file:            
              frhs = enmap.read_map(maps) 
        except:
            print('frhs too big')
            continue
            
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
        #flux_scale = (delta/1)**2 #Div one just a placeholder for ref scale
        #flux_scale = 1
                    

        #Stack it up, divide and return
        fstack += fstamp[0]#/flux_scale
        #kstack += kstamp[0]/flux_scale**2
        #print(stack)
        

    flux_stack = fstack#/kstack

    return np.array(flux_stack)#, np.array(fstack), np.array(kstack), np.array(flux_scale)

class minorplanet():
    '''
    Constructs a class that includes everything we want to do with a minor planet
    '''
    def __init__(self, name, frequency, array, obs = 'W99', t_start ='2010-01-01T00:00:00', t_end='2022-01-01T00:00:00'): #changed from 2020 to 2022
        '''
        Requires a name for the object, as well as an observatory code which can be found at
        https://en.wikipedia.org/wiki/List_of_observatory_codes
        See the QueryHorizons class for full details of observatory code options
        '''
        self.name = name
        self.obs = obs
        
        self.t_start = t_start
        self.t_end = t_end
        
        #We interpolate the orbit for the minor planet at initialization as it's fairly fast
        #Stacking is a method as it is much slower so we want to call it only when we actually care
        self.eph_table = QueryHorizons(time_start=self.t_start, time_end=self.t_end, 
                                       observer_location=obs, step = '1d').queryObjects([str(self.name)])
        
        self.interp_orbit = OrbitInterpolator(self.eph_table)
        
        #frequency of depth1 map we want to look at
        self.freq = frequency        
        
        #ACT array    
        self.arr = array            
        
    
    def show_orbit(self, t_orb_start = '2013-01-01T00:00:00', t_orb_end = '2022-01-01T00:00:00', #changed from 2020 to 2022 
                   directory = None):
        '''
        Function that plots the orbit of the minor planet from t_orb_start to t_orb_end
        '''
        
        t_start = Time(t_orb_start, format='isot', scale='utc')
        t_start = t_start.mjd

        t_en = Time(t_orb_end, format='isot', scale='utc')
        t_en = t_en.mjd
        mjds = np.linspace(t_start, t_en, 1000)

        ras, decs, delts = self.interp_orbit.get_radec_dist(self.eph_table['targetname'][0], mjds)

        plt.plot(ras, decs)
        plt.xlabel('ra')
        plt.ylabel('dec')
        plt.title('Orbital Plot of {}'.format(self.eph_table['targetname'][0]))
        if directory is not None:
            plt.savefig(directory+'{}_orbit.pdf'.format(str(self.eph_table['targetname'][0]).replace(' ', '_').replace('(','').replace(')','')))
        plt.show()
        plt.close()
        
    def make_stack(self):            
        self.flux_stack, self.fstack, self.flux_scale = tnoStacker(self.interp_orbit,self.eph_table['targetname'][0], self.freq, self.arr) #self.kstack,

    def save_stack(self, directory):
        aster_dict = {'flux_stack':self.flux_stack, 'fstack':self.fstack, 'flux_scale':self.flux_scale} #'kstack':self.kstack,
        with open(directory+'{}_stamp.pk'.format(str(self.eph_table['targetname'][0]).replace(' ', '_').replace('(','').replace(')','')), 'wb') as f:
            pk.dump(aster_dict, f)
            
    def plot_stack(self, directory = None, scale = None):
        plt.imshow(np.fliplr(self.flux_stack))
        plt.xlabel('ra')
        plt.ylabel('dec')
                
        plt.title('Plot of {name} Depth-1 Stack at {freq} on {arr}'.format(name = self.eph_table['targetname'][0], freq = self.freq, arr = self.arr))
        if directory is not None:
            plt.savefig(directory + '{}_stamp.pdf'.format(str(self.eph_table['targetname'][0]).replace(' ', '_').replace('(','').replace(')','')))
        plt.show()

#generate and save plot for asteroid/minor planet
asteroid = minorplanet("Ceres", "f090", "pa5")
asteroid.make_stack()
asteroid.save_stack('/scratch/r/rbond/ricco/plottest/pks/')
asteroid.plot_stack(directory = '/scratch/r/rbond/ricco/plottest/plots/Depth1/')