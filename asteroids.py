import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pixell import utils, enmap, bunch, reproject, colors, coordinates, mpi
from scipy import interpolate
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lcurve import *
from astro_utils import *
from pcurve import *

plt.style.use(astropy_mpl_style)


PATH_JACK = "/scratch/r/rbond/jorlo/actxminorplanets/sigurd/asteroids/"

class MinorObject:
    '''
    Minor object class that details asteroids, NEOs, dwarf planets, etc.
    Measurements taken from ACT in frequency range 90 - 220 GHz  
    '''

    def __init__(self, name:str, arr:str, freq:str, path:str = PATH_JACK, show:bool = False) -> None:
        '''
        Inputs:
            name, name of object in solar system
            arr, ACT array to collect data (chosen from pa4, pa5, pa6)
            freq, frequency of observation to look at (chosen from 90, 150, 220 GHz)
            path, path after running get_maps on depth1 maps  
            show, optionally display plots after calling methods
        '''

        if not name[0].isupper():
            upper = name[0].upper()
            name = upper + name[1:]

        self.name = name
        self.arr = arr
        self.freq = freq

        self.period = get_period(self.name.capitalize())
        self.path = path
        self.show = show


    def get_maps(self, astinfo:str, odir:str, directory:str = None, rad=10.0, pad=30.0, tol=0.0, lknee=1500, alpha=3.5, beam=0, verbose=2, quiet=0) -> None:
        '''
        Inputs: 
            astinfo, specifies path to ephemerides files
            odir, specifies path to output directory
            directory, specifies path to save plot of object's orbit + hits
            show, optionally show the path of the given object
        Outputs:
            map.fits, rho.fits, kappa.fits, ivar.fits, type: fits file, file of 
            object name on array arr at frequency freq saved in directory odir

        make sure to update with same name and astinfo
        make sure to change odir to correct name and freq            
        '''
        
        #path of depth1 maps
        depth_path = "/home/r/rbond/sigurdkn/project/actpol/maps/depth1/release/"
        
        #make map.fits for each file in depth_path 
        some_ifiles = glob.glob(depth_path + "*" + self.arr + "_" + self.freq + "*" + "map.fits")
            
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
            
        if self.show is not False:
        
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
            plt.savefig(directory + "{name}_plot_{arr}_{freq}.pdf".format(name=name, arr=self.arr, freq=self.freq))  


    def snr(self, directory:str = None) -> bytes:
        '''
        Inputs:
            directory, directory to save the output file         
        Output:
            stack.fits, type: file, stack of snr_tot of object
            F weighted maps
        '''

        path = self.path + self.name + "/" + self.arr + "/" + self.freq  
        
        #get rho and kappa files
        rho_files = glob.glob(path + "/*rho.fits")
        kap_files = [utils.replace(r, "rho.fits", "kappa.fits") for r in rho_files]
        
        if len(rho_files) != 0:  
            #find time
            str_times = []
            t_start = len(path) + len(self.name) + 9
            t_end = t_start + 10
            for time in rho_files:
                str_times.append(time[t_start:t_end])  
            int_times = [int(t) for t in str_times]
            
            #get data we want: ra, dec, geocentric dist, solar dist
            eph = np.load("/gpfs/fs0/project/r/rbond/sigurdkn/actpol/ephemerides/objects/" + self.name.capitalize() + ".npy").view(np.recarray)
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
            
            Name = self.name.capitalize()
            
            plt.figure()
            plt.title("snr tot of {name} on array {arr} at {freq}".format(name=Name, arr=self.arr, freq=self.freq))
            plt.imshow(image_data[0,:,:])
            plt.colorbar()
            
            if self.show is not False:
                plt.show()
                
            if directory is not None:
                plt.savefig(directory + "{name}_snr_tot_{arr}_{freq}.pdf".format(name=self.name, arr=self.arr, freq=self.freq))
            
        else:
            print("No hits")


    def flux_stack(self, directory:str = None, show_flux:bool = False) -> bytes:
        '''
        Inputs:
            directory, directory to save the output file
            show_flux, optionally show stack of flux of object
        Output:
            stack.fits, type: file, stack of snr or flux of object
            Rho and kappa maps without F weighting
        '''

        path = self.path + self.name
        
        #get rho and kappa files
        rho_files = glob.glob(path + "/*" + self.arr + "_" + self.freq + "_" + "rho.fits")
        kap_files = [utils.replace(r, "rho.fits", "kappa.fits") for r in rho_files]
            
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
                
            #get flux/snr
            if len(rho_files) != 0:
                flux = rho_tot / kap_tot    
                flux_unt = kap_tot**(-0.5)
                snr = flux / flux_unt
                
                if not show_flux:
                    hdu = fits.PrimaryHDU(snr)
                    hdu.writeto('snr.fits', overwrite = True)
                    image_file = get_pkg_data_filename('snr.fits')
                else:
                    hdu = fits.PrimaryHDU(flux)
                    hdu.writeto('flux.fits', overwrite = True)
                    image_file = get_pkg_data_filename('flux.fits')
                image_data = fits.getdata(image_file, ext = 0)            
            else:
                print("No hits on {name}_{arr}_{freq}".format(name=self.name, arr=self.arr, freq=self.freq))
            
            if self.show is not False:
                plt.figure()
                plt.title("Flux of {name} at {freq}".format(name=self.name, arr=self.arr, freq=self.freq))
                plt.imshow(image_data[0,:,:], cmap='coolwarm',vmin=-100, vmax=300)
                plt.colorbar(label='Flux (mJy)')                
                plt.show()
            
            if directory is not None:
                plt.savefig(directory + "{name}_snr_{arr}_{freq}.pdf".format(name=self.name, arr=self.arr, freq=self.freq))
            
        except UnboundLocalError:
            print("No hits on {name}_{arr}_{freq}".format(name=self.name, arr=self.arr, freq=self.freq))

    def plt_sum(self, arr, freq, directory:str = None) -> bytes:
        '''
        Inputs:
            arr, type: string array, ACT arrays that correspond to freq we want
            freq, type: string array, frequencies we want            
            directory, directory to save the output file            
        Output:
            stack.fits, type: file, stack of snr or flux of objects in summary plot across 3 freq bands
            Rho and kappa maps without F weighting
        '''

        path = self.path + self.name

        fig, ax = plt.subplots(1,3,figsize=(6,6))
        for count,f in enumerate(freq):   
            #get rho and kappa files
            rho_files = glob.glob(path + "/*" + arr[count] + "_" + f + "_" + "rho.fits")
            kap_files = [utils.replace(r, "rho.fits", "kappa.fits") for r in rho_files]
            
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
                
                #get flux/snr
                if len(rho_files) != 0:
                    flux = rho_tot / kap_tot    
                    flux_unt = kap_tot**(-0.5)
                    snr = flux / flux_unt
                
                    hdu = fits.PrimaryHDU(flux)
                    hdu.writeto('flux.fits', overwrite = True)
                
                    image_file = get_pkg_data_filename('flux.fits')
                    image_data = fits.getdata(image_file, ext = 0)
                
                    Name = self.name.capitalize()
                
                    ax[count].set_title("{name} on {arr} at {freq}".format(name=Name, arr=arr[count], freq=f))
                    divider = make_axes_locatable(ax[count])
                    cax = divider.append_axes('right',size='5%',pad=0.05)
                    fig.colorbar(ax[count].imshow(image_data[0,:,:],cmap='coolwarm'),cax=cax,label='Flux')
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=None)
                else:
                    print("No hits on {name}_{arr}_{freq}".format(name=self.name, arr=arr[count], freq=f))
                
                if count==len(freq)-1 and self.show is not False:
                    plt.show()
                    
                if directory is not None:
                    plt.savefig(directory + "{name}_snr_{arr}_{freq}.pdf".format(name=self.name, arr=arr[count], freq=f))
                
            except UnboundLocalError:
                print("No hits on {name}_{arr}_{freq}".format(name=self.name, arr=arr[count], freq=f))

    def make_image(self, path:str, directory:str = None) -> bytes:
        '''
        Input:
            path, type: string, path to map.fits file generated from running get_maps
            directory, type: string, path to save image file
        Output: 
            Image, plot of individual name on array arr at frequency freq
        '''
        
        #get data from path
        image_file = get_pkg_data_filename(path)
        image_data = fits.getdata(image_file, ext=0)
        
        #create plot
        Name = self.name.capitalize()
        
        plt.figure()
        plt.title("Image of {name} at {freq}".format(name=Name, arr=self.arr, freq=self.freq))
        plt.imshow(np.fliplr(image_data[0, :, :]))
        
        if directory is not None: 
            plt.savefig(directory + "{name}_image_{arr}_{freq}.pdf".format(name=self.name, arr=self.arr, freq=self.freq))
            
        plt.colorbar()
        plt.show()

    def make_gallery(self, directory:str = None) -> bytes:
        '''
        Inputs:
            directory, path to store image            
        Output:
            Image, gallery of depth1 images for object name on array arr at frequency freq
        '''
        
        path = self.path + self.name + "/" + self.arr + "/" + self.freq  
        
        #array of paths to map files
        map_files = glob.glob(path + "/*map.fits") 
            
        all_image_files = []    
        try:  
            for files in map_files:
                image_file = get_pkg_data_filename(files)
                all_image_files.append(image_file)   
            
            count = 0
            #based on length of all_image_files
            #proabably a better way to extract dimensions
            #for rows and cols
            rows = 8
            cols = 14
            fig, axarr = plt.subplots(rows, cols, figsize=(10,10))
        
            #false if there are more hits than gallery size
            if not ((rows*cols) > len(all_image_files)):
                raise IndexError
            
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
            if self.show is not False:  
                plt.show()
        
            if directory is not None:
               plt.savefig(directory + "{name}_gallery_{arr}_{freq}.pdf".format(name=self.name, arr=self.arr, freq=self.freq))  
            
        except UnboundLocalError:
            print("No hits")              

    def ivar_stack(self, directory:str = None):
        '''
        Inputs:
            directory, type: string, path to store image            
        Output:
            Image, stack of depth1 images for object name on array arr at frequency freq
        '''
        
        path = self.path + self.name + "/" + self.arr + "/" + self.freq    
        
        #array of paths to ivar and map files
        map_files = glob.glob(path + "/*map.fits")
        ivar_files = glob.glob(path + "/*ivar.fits")

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
                                        
            Name = self.name.capitalize()
            
            plt.figure()
            plt.title("Stack of {name} on array {arr} at {freq}".format(name=Name, arr=self.arr, freq=self.freq))
            plt.imshow(image_data)  
            plt.colorbar()    
            
            if self.show is not False:  
                plt.show()
            
            if directory is not None:
                plt.savefig(directory + "{name}_stack_{arr}_{freq}.pdf".format(name=self.name, arr=self.arr, freq=self.freq))  
            
        except UnboundLocalError:
            print("No hits")               

    def get_pk_index(self) -> int:
        '''
            Get index of object in asteroids.pk file
        '''

        index = get_index(self.name)

        return index

    def get_flux_predict(self) -> None:
        '''
            Return WISE flux prediction using F-weighting asteroids.pk file
        '''

        theory = get_theory(self.name, self.freq)

        return theory

    def plot_solar_phase(self, path:str = PATH_JACK, directory:str = None, save:bool = False) -> None:
        '''
            Plot solar phase angle over time
        '''

        get_alpha(path, self.name, self.arr, self.freq, directory, self.show, save)

    def plot_light_curve(self, n:int = 1, directory:str = None, pickle:bool = False) -> None:
        '''
            Plot light curve for object
        '''
        
        if n > 1:
            lcurves(self.arr, self.freq, n, directory, self.show, pickle)
        else:
            one_lcurve(self.name, self.arr, self.freq, directory, self.show, pickle)

    def get_light_curve_data(self) -> tuple:
        '''
            Get light curve data
        '''

        times, flux, error = lcurve_data(self.name, self.arr, self.freq)

        return times, flux, error

    def plot_phase_curve(self, n:int, stats:bool = False, spac:bool = False, directory:str = None, pickle:bool = False):
        '''
            Inputs:
                n, number of bins for inverse-weighted average
                directory, optionally save file in directory
                stats, optionally print statistics from best-fit
                pickle, optionally pickle phase data
                spac, optionally print solar phase angle correction
        '''
        
        better_phases(self.name, n, self.arr, self.freq, self.show, directory, stats, pickle, spac)

    def __str__(self) -> str:
        return f'I am {self.name} on ACT array {self.arr} at {self.freq} GHz'