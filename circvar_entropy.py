"""
This module is to evaluate statistical characteristics of the obtained andles: circular variance, entropy and standard deviation.
"""
from astropy.stats import circvar
from scipy.stats import entropy
import numpy as np
import math

def compute_circvar(x):    
        # take only non nan
    all_angles = x[~np.isnan(x)]
        #  double angles
    all_angles = 2*all_angles
        # circvar in astropy is defined for angles in radians 
    all_angles_rad = np.array([math.radians(x) for x in all_angles])     
    cv = circvar(all_angles_rad)
    return cv
    
def compute_entropy(x):
    all_angles = x[~np.isnan(x)]
    for i in range(len(all_angles)):
        all_angles[i] = all_angles[i]*2

    Nbins = 200
    hist = np.histogram(all_angles, bins=Nbins)
        # frequencies(~probabilities) corresponding to the angles
    data = hist[0]
        # entropy in scipy.stats normalizes the probabilities first
    ent = entropy(data)
    norm_entropy = ent/math.log(Nbins)
    return norm_entropy

def compute_std(x):
    """
    The formula to compute standard deviation from circular variance: std = (-2ln(R1))^(1/2), where R1 = 1 - circvar 
        is taken from .pptx (2021-10.SHG bilan complet 8-12SL_6M). 
    """
    cv = compute_circvar(x) 
    R1  = 1 - cv
    std_rad = np.sqrt(-2*np.log(R1))
    std_deg = std_rad * (180/3.1415) /2
    return std_rad, std_deg