import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

from scipy import optimize
from astropy.time import Time
from statistics import mean
from scipy.stats import chi2

from astro_utils import *
from lcurve import *


FREQ_LIST = ['f090', 'f150', 'f220']

def phases1(name:str, n:int, arr:str, freq:str, show:bool = False, directory:str = None, stats:bool = False, pickle:bool = False):
  '''
  Inputs:
    name, name of object we want phase curve for
    n, number of bins for inverse-weighted average
    freq, frequency we want to look at
    show, if true, display light curve after calling lcurve    
    directory, optionally save file in directory
    stats, optionally calculate statistics for fit
    pickle, optionally pickle phases, residuals, etc.
  Output:
    Phase curve of object given a period
    also returns residuals for phase
  '''

  period = get_period(name)
  
  pos = period / 24
  phases = []
  for f in FREQ_LIST:
    if f == freq:
      plt.rcParams.update({'font.size': 16})

      try:
        in_file = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_" + arr 
                       + "_" + f + ".pk", 'rb')
        dict_freq = pk.load(in_file)
        in_file.close()

        obj = dict_freq['Name']

        # get times
        for t in range(len(dict_freq['Time'])):
          num = (dict_freq['Time'][t]) % pos
          phases.append(num/pos)

        # get data
        fluxs = dict_freq['Flux']
        errors = dict_freq['Error']
        fWeights = dict_freq['Weighting']
      except KeyError:
        one_lcurve(name, arr, f, '/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/', pickle=True)
        phases1(name, n, arr, freq, show, directory, stats, pickle)

      res = [(fluxs[f] - fWeights[f]) for f in range(len(fluxs))]

      plt.clf()

      interval = 1/n
      center = interval * 0.5
      phase_bins = np.arange(0,1,interval) + center

      #inverse weighting
      err_prop = []
      ave_var = []

      for i in np.arange(0,1,interval):
        try:
          err_bin = [errors[f] for f in range(len(phases)) if i <= phases[f] <= (i+interval)]
          # find residuals in bin1
          res_bin = [res[r] for r in range(len(phases)) if i <= phases[r] <= (i+interval)]
          err_data_sqr = [err_bin[e]**2 for e in range(len(err_bin))]
          ave_var_temp, i_var = inv_var(res_bin, err_data_sqr)
          new_err = i_var**0.5

          ave_var.append(ave_var_temp)
          err_prop.append(new_err)
        except ZeroDivisionError:
          print('Zero division error')
          ave_var.append(np.nan)
          err_prop.append(np.nan)

      res_avg = mean(res)

      # sin fit
      params, params_covariance = optimize.curve_fit(fit_sin, phases, res, sigma=errors)   
      perr = np.sqrt(np.diag(params_covariance))   
      x = np.arange(0,1,0.01)
      y_fit = fit_sin(x, params[0], params[1], params[2])

      if stats:    
        #chi-square sin
        diff = (fit_sin(phases, params[0], params[1], params[2]) - res)**2
        errors = np.array(errors)
        chi_sqr = np.sum(diff / (errors**2))
        dof = len(res) - 3
        red_chi_sqr = chi_sqr / dof
        if abs(params[0]/perr[0]) > 5:       
          print('Possible detection') 
        print("Chi-squared for fit: ", chi_sqr)
        print("DOF: ", dof) 
        print("Reduced chi-squared for fit: ", red_chi_sqr)
        print(f'S/N: {abs(params[0]/perr[0])}')
        print("PTE for fit: ", chi2.sf(chi_sqr, dof))
        print("Amplitude: ", params[0], "with uncertainty: ", perr[0])

        # chi-square average
        diff_avg = (res_avg - res)**2
        chi_sqr_avg = np.sum(diff_avg / (errors**2))
        print('Chi-squared for average: ', chi_sqr_avg)

      # plot
      plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='s', label='Bin Flux', zorder=1)
      plt.errorbar(phases, res, yerr=errors, fmt='o', label='Flux', zorder=0,alpha=0.5)
      plt.axhline(y=res_avg, linestyle='--', label='Average Flux', zorder=0)
      plt.plot(x,y_fit, label='Fitted Ftn.')  
      plt.xlabel("Phase Number")
      plt.ylabel("Flux Residual (mJy)")
      plt.legend(loc='best')     
      
      if show:
        plt.show()
        
      if directory is not None:
        plt.savefig(directory + "{name}_phase_curve_{freq}.pdf".format(name=obj, freq=freq))           

      if pickle:
        data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Phase Number': phases, 'Residuals': res, 'Error': errors, 
                     'Measured':fluxs,'Model (F Weights)':fWeights}
        filename = directory + "phase_curve_data_" + name + "_" + arr + "_" + freq + ".pk"
        outfile = open(filename, 'wb')
        pk.dump(data_dict, outfile)
        outfile.close()              

    else:
      continue    

def better_phases(name:str, n:int, arr:str, freq:str, show:bool = False, directory:str = None, stats:bool = False, pickle:bool = False, 
                  spac:bool = False):
  '''
  Inputs:
    name, name of object we want phase curve for
    n, number of bins for inverse-weighted average
    freq, frequency we want to look at
    show, if true, display light curve after calling lcurve    
    directory, optionally save file in directory
    stats, optionally print statistics from best-fit
    pickle, optionally pickle phase data
    spac, optionally print solar phase angle correction
  Output:
    Phase curve of object given a period
    Incorporates rotational, orbital period
    solar phase angle in binning
  '''

  for f in FREQ_LIST:
    if f == freq:
      plt.rcParams.update({'font.size': 16})

      try:
        in_file = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_" + arr 
                       + "_" + f + ".pk", 'rb') 
        dict_freq = pk.load(in_file)
        in_file.close()

        # get data
        fluxs = dict_freq['Flux']
        errors = dict_freq['Error']
        fWeights = dict_freq['F'] # weighting
        obj = dict_freq['Name']
        time_freq = dict_freq['Time']
        
      except KeyError:
        one_lcurve(name, arr, f, '/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/', pickle=True)
        better_phases(name, n, arr, freq, show, directory, stats, pickle)

      # get phase numbers
      phi = rot_phase(obj, time_freq) # rotation phase numbers
      phi = np.array(phi)    
      theta, ignore_T = orb_phase(obj, time_freq) # orbital phase numbers
      theta = np.array(theta)    
      alpha = sunang_phase(obj, time_freq) # solar phase angle numbers
      alpha = np.array(alpha)          
      
      if spac:
        fluxs /= np.cos(alpha) # MAYBE DELETE flux correction
        fWeights /= np.cos(alpha) # MAYBE DELETE flux correction

      res = [(fluxs[f] - fWeights[f]) for f in range(len(fluxs))]
      res_avg = mean(res)    

      eta = phi + theta + alpha
      eta_phase = eta % 1       

      #binning
      ave_var, err_prop, phase_bins = inv_var_weight(n, errors, eta_phase, res)

      #sin fit        
      eta_phase, res, errors = zip(*sorted(zip(eta_phase, res, errors)))    
      params, params_covariance = optimize.curve_fit(fit_sin, eta_phase, res, sigma=errors)   
      perr = np.sqrt(np.diag(params_covariance))   
      x = np.arange(0,1,0.01)
      y_fit = fit_sin(x, params[0], params[1], params[2])    
      
      if stats:
        # chi-square sin
        diff = (fit_sin(eta_phase, params[0], params[1], params[2]) - res)**2
        errors = np.array(errors)
        chi_sqr = np.sum(diff / (errors**2))
        dof = len(res) - 3
        red_chi_sqr = chi_sqr / dof
        if abs(params[0]/perr[0]) > 5:
          print('Possible Detection')       
        print("Chi-squared for fit: ", chi_sqr)
        print("DOF: ", dof) 
        print("Reduced chi-squared for fit: ", red_chi_sqr)
        print(f'S/N: {abs(params[0]/perr[0])}')
        print("PTE for fit: ", chi2.sf(chi_sqr, dof))
        print("Amplitude: ", params[0], "with uncertainty: ", perr[0])

        # chi-square average
        diff_avg = (res_avg - res)**2
        chi_sqr_avg = np.sum(diff_avg / (errors**2))
        print('Chi-squared for average: ', chi_sqr_avg)   
      
      #plot      
      plt.clf()    
      plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='s', label='Bin Flux', zorder=1)
      plt.errorbar(eta_phase, res, yerr=errors, fmt='o', label='Flux', zorder=0,alpha=0.5)
      plt.axhline(y=res_avg, linestyle='--', label='Average Flux', zorder=0)
      plt.plot(x,y_fit, label='Fitted Ftn.')
      plt.tick_params(direction='in')
      plt.xlabel("Phase Number")
      plt.ylabel("Flux Residual (mJy)")
      plt.legend(loc='best')   

      if show:
        plt.show()
        
      if directory is not None:
        plt.savefig(directory + "{name}_phase_curve_{freq}.pdf".format(name=obj, freq=freq))           
        
      if pickle:
        data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Phase Number': eta_phase, 'Residuals': res, 'Error': errors,
                     'Measured':fluxs,'Model (F Weights)':fWeights}
        filename = directory + "phase_curve_data_" + name + "_" + arr + "_" + freq + ".pk"
        outfile = open(filename, 'wb')
        pk.dump(data_dict, outfile)
        outfile.close()     

    else:
      continue

def vesta_phases(name:str, n:int, arr:str, freq:str, show:bool = False, directory:str = None, stats:bool = False, pickle:bool = False):
  '''
  Inputs:
    name, name of object we want phase curve for
    n, number of bins for inverse-weighted average
    freq, frequency we want to look at
    show, if true, display light curve after calling lcurve    
    directory, optionally save file in directory
    stats, optionally print statistics for best-fit
    pickle, optionally pickle phase data
  Output:
    Phase curve of object given a period
    Incorporates rotational, orbital period
    solar phase angle in binning
  '''

  for f in FREQ_LIST:
    if f == freq:
      plt.rcParams.update({'font.size': 16})

      try:
        in_file = open("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/lcurve_data_" + name + "_" + arr 
                      + "_" + f + ".pk", 'rb')
        dict_freq = pk.load(in_file)
        in_file.close()

        # get data
        fluxs = dict_freq['Flux']
        errors = dict_freq['Error']
        fWeights = dict_freq['F'] # weighting
        obj = dict_freq['Name']
        times = dict_freq['Time']

      except KeyError:
        one_lcurve(name, arr, f, '/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/', pickle=True)
        vesta_phases(name, n, arr, freq, show, directory, stats, pickle)

      # get phase numbers
      phi = rot_phase(obj, -times) # rotation phase numbers
      phi = np.array(phi)    

      res = [(fluxs[f] - fWeights[f]) for f in range(len(fluxs))]
      res_avg = mean(res)    
      
      # binning
      ave_var, err_prop, phase_bins = inv_var_weight(n,errors,phi,res)   

      # sin fit
      phi, res, errors = zip(*sorted(zip(phi,res,errors)))
      params, params_covariance = optimize.curve_fit(vesta_fit, phi, res, sigma=errors)   
      perr = np.sqrt(np.diag(params_covariance))   
      x = np.arange(0,1,0.01)
      y_fit = vesta_fit(x, params[0], params[1], params[2], params[3], params[4])   

      if stats:     
        #chi-square sin
        diff = (vesta_fit(phi, params[0], params[1], params[2], params[3], params[4]) - res)**2
        errors = np.array(errors)
        chi_sqr = np.sum(diff / (errors**2))
        dof = len(res) - 5
        red_chi_sqr = chi_sqr / dof
        #pte
        print(f'Chi-square: {chi_sqr}')
        print(f'Reduced chi-square: {red_chi_sqr}')
        print(f'PTE: {chi2.sf(chi_sqr, dof)}')

        #chi-square average     
        diff_avg = (res_avg - res)**2
        chi_sqr_avg = np.sum(diff_avg / (errors**2))
        print("Chi-squared for average: ", chi_sqr_avg)   

      #plot      
      plt.clf()    
      plt.errorbar(phase_bins, ave_var, yerr=err_prop, fmt='s', label='Bin Flux', zorder=1)
      plt.errorbar(phi, res, yerr=errors, fmt='o', label='Flux', zorder=0,alpha=0.5)
      plt.axhline(y=res_avg, linestyle='--', label='Average Flux', zorder=0)
      plt.plot(x,y_fit, label='Fitted Ftn.')    
      plt.tick_params(direction='in')
      plt.xlabel("Phase Number")
      plt.ylabel("Flux Residual (mJy)")
      plt.legend(loc='best')

      if show:
        plt.show()      

      if directory is not None:
        plt.savefig(directory + "{name}_phase_curve_{freq}.pdf".format(name=obj, freq=freq))  

      if pickle:
        data_dict = {'Name': name, 'Array': arr, 'Frequency': freq, 'Phase Number': phi, 'Residuals': res, 'Error': errors, 
                     'Measured':fluxs, 'Model (F Weights)':fWeights}
        filename = directory + "phase_curve_data_" + name + "_" + arr + "_" + freq + ".pk"
        outfile = open(filename, 'wb')
        pk.dump(data_dict, outfile)
        outfile.close()                                         

    else:
      continue

def jack_vesta_phases():
  '''    
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
  y_fit = sin2model(x, 0.030, 0.181, 0.012, 0.461, 0.991)
  #y_fit = fit_sin(x, 0.03, 0.181, 0.991)
  
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
  plt.savefig("/gpfs/fs1/home/r/rbond/ricco/minorplanets/asteroids/light_curves/{name}_light_curve_all_{freq}.pdf".format(name=name, 
                                                                                                                          freq=freq))