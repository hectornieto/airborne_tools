import numpy as np
import scipy.interpolate as interp


def gamma2sigma(Gamma):
    '''Function to convert FWHM (Gamma) to standard deviation (sigma)
    
    Parameters
    ----------
    Gamma : float or numpy array
        Full With Half Maximum value
    
    Returns
    -------
    sigma : float or numpy array
        Standard deviation of the Gaussian Response Function        
    '''
    return Gamma * np.sqrt(2.0) / (np.sqrt(2.0 * np.log(2.0)) * 2.0)


def sigma2gamma(sigma):
    '''Function to convert standard deviation (sigma) to FWHM (Gamma)
    
    Parameters
    ----------
    sigma : float or numpy array
        Standard deviation of the Gaussian Response Function        
    
    Returns
    -------
    Gamma : float or numpy array
        Full With Half Maximum value
    '''
    return sigma * np.sqrt(2.0 * np.log(2.0)) * 2.0 / np.sqrt(2.0)


def create_gaussian_srf(wl, fwhm, width=(400, 2500), step=1):
    ''' Function to calculate a normalized Gaussian Spectral Response function
    Parameters
    ----------
    wl : float
        Center wavelenght        
    fwhm : float
        Full With Half Maximum value
    width : tuple (float,float)
        Tuple with  minimum and maximum wavelenths for bandhwidht response function
        default between 400 and 2500 nm
    step : float
        Ouput spectral resolution of the response function
    
    Returns
    -------
    wls , response:  2D numpy array
        Spectral response function. 1st element is wavelengnth 
        and 2nd is the contributing signal at that wavelenght
        
    '''

    wls = np.arange(width[0], width[1], step=step)
    sigma = gamma2sigma(fwhm)  # convert FWHM to standard deviation
    response = np.exp(- ((wls - wl) / sigma) ** 2)
    return wls, response


def apply_filter(signal, fltr):
    ''' Integrates a spectral signal to a spectral filter
    
    Parameters
    ----------
    signal : numpy array
        Spectral signal
    fltr : numpy array
        Spectral filter, last dimension must equal last dimension in signal
    
    Returns
    -------
    int_signal : float or numpy array
        Integrated signal
    '''
    signal = np.asarray(signal)
    fltr = np.asarray(fltr)
    dims = signal.shape
    if dims[-1] != fltr.shape[-1]:
        print('Input number of bands does not match with filter shape')
        return None
    output = signal * fltr
    n_dims = len(dims)
    int_signal = np.nansum(output, axis=n_dims - 1)
    return int_signal


def convolve_srf(spectra_in, srf):
    '''
    Convolves spectra based on an Spectral Response Function
    
    Parameters
    ----------
    spectra_in : numpy array
        1rst elements is wavelength and next elements are the corresponding reflectance factor, one colum per spectrum
    srf : 2D numpy array
        1rst elements is wavelength and second element is the corresponding contribution to the signal
    
    Returns
    -------
    reflectance_out : float or numpy array
        convolved reflectance value
    '''

    fltr = interpolate_srf(spectra_in[0], srf, normalize='sum')
    reflectance_out = apply_filter(spectra_in[1:], fltr)
    return reflectance_out


def interpolate_srf(out_wl, srf, normalize='max'):
    ''' Intepolates an Spectral Response Function to a given set of wavelenghts
    Parameters
    ----------
    out_wl : numpy array
        Wavelenths at which the spectral response function will be interpolated
    srf : 2D numpy array
        1rst elements is wavelength and second element is the corresponding contribution to the signal
    normalize : str
        Method used to normalize the output response signal: 'max', 'sum', 'none'
        
    Returns
    -------
    fltr : float or numpy array
        Integrated filter
    '''

    out_wl = np.asarray(out_wl)
    srf = np.asarray(srf)
    # Create the linear interpolation object
    f = interp.interp1d(srf[0], srf[1], bounds_error=False, fill_value='extrapolate')
    # Interpolate and normalize to max=1
    fltr = f(out_wl)
    if normalize == 'max':
        fltr = fltr / np.max(fltr)
    elif normalize == 'sum':
        fltr = fltr / np.sum(fltr)
    elif normalize == 'none':
        pass
    else:
        print('Wrong normalize keyword, use "max" or "sum"')
    return fltr


def integrage_srfs(srf_master, srf_slave):
    srf_master = np.asarray(srf_master)
    srf_slave = np.asarray(srf_slave)
    fltr = convolve_srf(srf_master, srf_slave)
    srf = srf_master[1] * fltr
    srf = np.sum(srf)
    return srf
