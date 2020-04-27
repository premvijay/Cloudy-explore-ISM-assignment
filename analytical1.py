import astropy.constants as const
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

def alpha_A(T):
    T_K = T/u.K
    return 2.065 * 10**-11 * T_K**(-1/2) * ( 6.414 - 1/2 * np.log(T_K) + 8.68 * 10**-3 *T_K**(1/3)) * u.cm**3 / u.s

from scipy.integrate import quad

def alpha_1(T):
    nu0 = 3.288465 * 10**15 * u.Hz
    factor = nu0**3 *8 * np.pi / const.c**2 * (const.h**2/(2 * np.pi * const.m_e * const.k_B))**(3/2) * 6.30* 10**-18 * u.cm**2
    def I(nu2):
        return np.exp(- const.h * nu2 *nu0/ (const.k_B * T)) * (1 + nu2)**-1
    integrated = quad(I,0,np.inf)[0]
    return (factor * (1/T**(3/2)) * integrated).to(u.cm**3/u.s)

def alpha_B(T):
    return alpha_A(T)-alpha_1(T)


def N_dot(T_star,R_star):
    nu0 = 3.288465 * 10**15 * u.Hz
    factor = nu0**3 *8 * np.pi**2 / const.c**2
    def I(nu3):
        return nu3**2 / (np.exp(const.h * nu3 *nu0/ (const.k_B * T_star))-1)
    integrated = quad(I,1,np.inf)[0]
    return (factor * (R_star**2) * integrated).to(1/u.s)


def depth(T_star,R_star,n,T):
    R = (3 * N_dot(T_star,R_star) /(4 * np.pi * n**2 * alpha_B(T) ) + (10**18.477121 * u.cm)**3)**(1/3)
    depth = R - (10**18.477121 * u.cm)
    return depth

def depth_nodim(T_star,log_R_star,log_n,T):
    T_star *= u.K
    R_star = 10**log_R_star * u.cm
    n = 10**log_n * u.cm**-3
    T *= u.K
    return float(depth(T_star,R_star,n,T)/u.cm)







