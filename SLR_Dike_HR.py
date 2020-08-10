# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 08:55:10 2020

@author: akurnizk
"""

import utm
import csv
import math
import flopy
import sys,os
import calendar
import dateutil
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rc('xtick', labelsize=22)     
mpl.rc('ytick', labelsize=22)
mpl.rcParams.update({'font.size': 22})
mpl.rcParams['pdf.fonttype'] = 42
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import flopy.utils.binaryfile as bf

cgw_code_dir = 'E:\Python KMB - CGW' # Location of BitBucket folder containing cgw folder
sys.path.insert(0,cgw_code_dir)

from mpmath import *

from matplotlib import pylab
from moviepy.editor import *
from scipy.io import loadmat
from scipy.optimize import fsolve
from shapely.geometry import Point
from datetime import datetime, time, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

# Assign name and create modflow model object

work_dir = os.path.join('E:\Herring Models\Seasonal')
data_dir = os.path.join('E:\Data')

mean_sea_level = 0.843 # Datum in meters at closest NOAA station (8447435), Chatham, Lydia Cove MA
# https://tidesandcurrents.noaa.gov/datums.html?units=1&epoch=0&id=8447435&name=Chatham%2C+Lydia+Cove&state=MA

#%% To Do

# Compare sea level measurements at Boston, Provincetown, and outside dike.

# Find land levels

# 

#%% Loading Information from HR Dike Sensors (Make sure times are in EDT)

with open(os.path.join(data_dir,"General Dike Data","USGS 011058798 Herring R at Chequessett Neck Rd.txt")) as f:
    reader = csv.reader(f, delimiter="\t")
    HR_dike_all_info = list(reader)
    
HR_dike_lev_disch_cond = HR_dike_all_info[32:]
HR_dike_all_df = pd.DataFrame(HR_dike_lev_disch_cond[2:], columns=HR_dike_lev_disch_cond[0])
HR_dike_all_df.drop(HR_dike_all_df.columns[[0,1,3,5,7,9,11,13]],axis=1,inplace=True)
HR_dike_all_df.columns = ["datetime","Gage height, ft, Ocean side","Discharge, cfs","Gage height, ft, HR side",
                          "Spec Con, microsiemens/cm, HR side","Spec Con, microsiemens/cm, Ocean side"]
# HR_dike_all_df = HR_dike_all_df.replace(r'^\s*$', np.nan, regex=True)
HR_dike_all_df = HR_dike_all_df.replace("Eqp", '', regex=True)
HR_dike_all_df["datetime"] = pd.to_datetime(HR_dike_all_df["datetime"])
HR_dike_all_df["Gage height, ft, Ocean side"] = pd.to_numeric(HR_dike_all_df["Gage height, ft, Ocean side"])
HR_dike_all_df["Discharge, cfs"] = pd.to_numeric(HR_dike_all_df["Discharge, cfs"])
HR_dike_all_df["Gage height, ft, HR side"] = pd.to_numeric(HR_dike_all_df["Gage height, ft, HR side"])
HR_dike_all_df["Spec Con, microsiemens/cm, HR side"] = pd.to_numeric(HR_dike_all_df["Spec Con, microsiemens/cm, HR side"])
HR_dike_all_df["Spec Con, microsiemens/cm, Ocean side"] = pd.to_numeric(HR_dike_all_df["Spec Con, microsiemens/cm, Ocean side"])

# Merging Duplicate Entries
HR_dike_all_df.set_index('datetime',inplace=True)
HR_dike_all_df = HR_dike_all_df.mean(level=0)
HR_dike_all_df.reset_index(inplace=True)

HR_dike_lev_disch_ft = HR_dike_all_df[["datetime","Gage height, ft, Ocean side","Gage height, ft, HR side","Discharge, cfs"]]
HR_dike_lev_disch_m = HR_dike_lev_disch_ft.copy()
HR_dike_lev_disch_m.columns = ["datetime","Gage height, m, Ocean side","Gage height, m, HR side","Discharge, cms"]
HR_dike_lev_disch_m["Gage height, m, Ocean side"] = HR_dike_lev_disch_ft["Gage height, ft, Ocean side"]*0.3048
HR_dike_lev_disch_m["Gage height, m, HR side"] = HR_dike_lev_disch_ft["Gage height, ft, HR side"]*0.3048
HR_dike_lev_disch_m["Discharge, cms"] = HR_dike_lev_disch_ft["Discharge, cfs"]*0.02832
# HR_dike_all_df = HR_dike_all_df.fillna('')
          
x_datenum_dike = mdates.date2num(HR_dike_lev_disch_m["datetime"])
HR_dike_lev_disch_m.insert(1,"datenum",x_datenum_dike,True)

ax = HR_dike_lev_disch_m.plot.scatter(x="datenum", y="Gage height, m, Ocean side", color='LightBlue', label = 'Gage height, m , Ocean side')
# ax = HR_dike_lev_disch_m.plot.scatter(x="datenum", y="Gage height, m, HR side", color='LightGreen', label = 'Gage height, m , HR side')
HR_dike_lev_disch_m.plot.scatter(x="datenum", y="Gage height, m, HR side", color='LightGreen', label = 'Gage height, m , HR side', ax=ax)
HR_dike_lev_disch_m.plot.scatter(x="datenum", y="Discharge, cms", color='Turquoise', label = 'Discharge, cms', ax=ax)
# ax = HR_dike_lev_disch_m.plot.scatter(x="datenum", y="Discharge, cms", color='Turquoise', label = 'Discharge, cms')

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=22)
plt.ylabel('Elevation (m), Discharge (m^3/s)', fontsize=22)
plt.legend()

#%% Loading Information from HR CTD Sensors (Make sure times are in EDT)

with open(os.path.join(data_dir,"General Dike Data","Water_Elevation,_NAVD88-File_Import-01-22-2020_15-04.txt")) as f:
    reader = csv.reader(f, delimiter="\t")
    HR_CTD_all_info = list(reader)

HR_CTD_lev = HR_CTD_all_info[1:]
HR_CTD_all_df = pd.DataFrame(HR_CTD_lev[2:], columns=HR_CTD_lev[0])
HR_CTD_all_df.drop(HR_CTD_all_df.columns[[0,2,4]],axis=1,inplace=True)
HR_CTD_all_df = HR_CTD_all_df.rename(columns={"Time (MDT to EDT)":"datetime"})
# HR_CTD_all_df = HR_CTD_all_df.replace(r'^s*$', np.nan, regex=True)
# HR_CTD_all_df = HR_CTD_all_df.replace("Eqp", '', regex=True)
HR_CTD_all_df["datetime"] = pd.to_datetime(HR_CTD_all_df["datetime"])
HR_CTD_all_df["High Toss Water Level, NAVD88"] = pd.to_numeric(HR_CTD_all_df["High Toss Water Level, NAVD88"])
HR_CTD_all_df["CNR U/S Water Level, NAVD88"] = pd.to_numeric(HR_CTD_all_df["CNR U/S Water Level, NAVD88"])
HR_CTD_all_df["Dog Leg Water Level, NAVD88"] = pd.to_numeric(HR_CTD_all_df["Dog Leg Water Level, NAVD88"])
HR_CTD_all_df["Old Saw Water Level, NAVD88"] = pd.to_numeric(HR_CTD_all_df["Old Saw Water Level, NAVD88"])

# Merging Duplicate Entries
HR_CTD_all_df.set_index('datetime',inplace=True)
HR_CTD_all_df = HR_CTD_all_df.mean(level=0)
HR_CTD_all_df.reset_index(inplace=True)

# Filtering
HR_CTD_all_df["High Toss Water Level, NAVD88"][HR_CTD_all_df["High Toss Water Level, NAVD88"] > 1.00] = np.nan
HR_CTD_all_df["High Toss Water Level, NAVD88"][HR_CTD_all_df["High Toss Water Level, NAVD88"] < -0.67] = np.nan
HR_CTD_all_df["CNR U/S Water Level, NAVD88"][HR_CTD_all_df["CNR U/S Water Level, NAVD88"] < -0.90] = np.nan
HR_CTD_all_df["CNR U/S Water Level, NAVD88"][HR_CTD_all_df["CNR U/S Water Level, NAVD88"] > 0.55] = np.nan
HR_CTD_all_df["Old Saw Water Level, NAVD88"][HR_CTD_all_df["Old Saw Water Level, NAVD88"] < -2.14] = np.nan

HR_CTD_lev_m = HR_CTD_all_df[["datetime","Old Saw Water Level, NAVD88","CNR U/S Water Level, NAVD88",
                               "Dog Leg Water Level, NAVD88","High Toss Water Level, NAVD88"]]
HR_CTD_lev_m.columns = ["datetime","Water Level, m, Old Saw","Water Level, m, CNR U/S","Water Level, m, Dog Leg",
                        "Water Level, m, High Toss"]

x_datenum_CTD = mdates.date2num(HR_CTD_lev_m["datetime"])
HR_CTD_lev_m.insert(1,"datenum",x_datenum_CTD,True)

ax = HR_CTD_lev_m.plot.scatter(x="datenum", y="Water Level, m, Old Saw", color='DarkBlue', label = 'Water Level, m, Old Saw')
# HR_CTD_lev_m.plot.scatter(x="datenum", y="Water Level, m, Old Saw", color='DarkBlue', label = 'Water Level, m, Old Saw', ax=ax)
HR_CTD_lev_m.plot.scatter(x="datenum", y="Water Level, m, CNR U/S", color='DarkGreen', label = 'Water Level, m, CNR U/S', ax=ax)
HR_CTD_lev_m.plot.scatter(x="datenum", y="Water Level, m, Dog Leg", color='DarkRed', label = 'Water Level, m, Dog Leg', ax=ax)
HR_CTD_lev_m.plot.scatter(x="datenum", y="Water Level, m, High Toss", color='DarkOrange', label = 'Water Level, m, High Toss', ax=ax)

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=22)
plt.ylabel('Elevation (m), Discharge (m^3/s)', fontsize=22)
# plt.ylabel('Elevation (m)', fontsize=22)
plt.legend(loc='upper right')
# plt.legend(loc='lower right')


#%% Combining Information from Dike and CTD, Interpolating CTD to multiples of 5 min.

HR_dike_lev_disch_m_di = HR_dike_lev_disch_m.set_index('datetime')
# HR_CTD_lev_m_di = HR_CTD_lev_m.set_index('datetime')

HR_dike_CTD_lev_disch_m = pd.merge_ordered(HR_dike_lev_disch_m, HR_CTD_lev_m)
HR_dike_CTD_lev_disch_m_di = HR_dike_CTD_lev_disch_m.set_index('datetime')
HR_dike_CTD_lev_disch_m_di.interpolate(method='index', limit=1,inplace=True)
# HR_dike_CTD_lev_disch_m_di.drop(HR_dike_CTD_lev_disch_m_di.columns[[0]],axis=1,inplace=True)

HR_dike_CTD_lev_disch_m_di_resam = HR_dike_CTD_lev_disch_m_di.loc[HR_dike_lev_disch_m_di.index]

ax = HR_dike_CTD_lev_disch_m_di_resam.plot.scatter(x="datenum", y="Gage height, m, Ocean side", color='LightBlue', label = 'Gage height, m , Ocean side')
# ax = HR_dike_CTD_lev_disch_m_di_resam.plot.scatter(x="datenum", y="Gage height, m, HR side", color='LightGreen', label = 'Gage height, m , HR side')
HR_dike_CTD_lev_disch_m_di_resam.plot.scatter(x="datenum", y="Gage height, m, HR side", color='LightGreen', label = 'Gage height, m , HR side', ax=ax)
HR_dike_CTD_lev_disch_m_di_resam.plot.scatter(x="datenum", y="Discharge, cms", color='Turquoise', label = 'Discharge, cms', ax=ax)
HR_dike_CTD_lev_disch_m_di_resam.plot.scatter(x="datenum", y="Water Level, m, Old Saw", color='DarkBlue', label = 'Water Level, m, Old Saw', ax=ax)
# HR_dike_CTD_lev_disch_m_di_resam.plot.scatter(x="datenum", y="Water Level, m, Old Saw", color='DarkBlue', label = 'Water Level, m, Old Saw', ax=ax)
HR_dike_CTD_lev_disch_m_di_resam.plot.scatter(x="datenum", y="Water Level, m, CNR U/S", color='DarkGreen', label = 'Water Level, m, CNR U/S', ax=ax)
HR_dike_CTD_lev_disch_m_di_resam.plot.scatter(x="datenum", y="Water Level, m, Dog Leg", color='DarkRed', label = 'Water Level, m, Dog Leg', ax=ax)
HR_dike_CTD_lev_disch_m_di_resam.plot.scatter(x="datenum", y="Water Level, m, High Toss", color='DarkOrange', label = 'Water Level, m, High Toss', ax=ax)

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=22)
plt.ylabel('Elevation (m), Discharge (m^3/s)', fontsize=22)
# plt.ylabel('Elevation (m)', fontsize=22)
plt.legend(loc='upper right')
# plt.legend(loc='lower right')

#%% Newton-Raphson Method (to be used in determining gate opening angle)

def newton(f,Df,x0,epsilon,max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None

#%% Analytical Estimation of Discharge Through Dike Using Water Levels, My Analysis (all SI) - Version 1

# # Add option for different configurations (number/size/type of openings)?

# """
# Sources:
# Sluice-gate Discharge Equations by Prabhata K. Swamee, Journal of Irrigation and Drainage Engineering, Vol. 118
# Herring River Full Final Report, Woods Hole Group June 2012
# Hydrodynamic and Salinity Modeling for Estuarine Habitat Restoration at HR, Wellfleet, MA. Spaulding and Grilli October 2001
#     (Higher frictional losses on the ebb than on the flood tide, pp. ii, and n~0.06 to 0.09 for HR bed)
#     *Loss coefficients hard to justify given difference in distances between the HR basin (S&G) and measurements around the dike*
    
# Can solve for the "additional coefficient" (make a K array) at each point by dividing the measured discharge by everything on the RHS.
# Need to make several K arrays - one for each scenario, and take the average K of each as the fitting parameter.
# """

# inv_el_open = -1.064
# slope_culv = 0.0067
# len_culv = 20.42
# inv_el_HRside = -0.928
# sluice_bot_el = -0.579
# y_sluice_open = sluice_bot_el-inv_el_open
# A_sluice_open = y_sluice_open*L_sluice_culv
# L_sluice_culv = 1.829
# L_center_culv = 2.184
# L_left_culv = 2.007
# L_flaps_in = 1.829
# L_flaps_out = 2.057
# angle_init_flaps = 0.0872 # radians, ~ 5 degrees
# dens_seawater = 1018 # kg/m^3, average is roughly the same on both sides of the dike.
# grav = 9.81 # m/s^2
# W_gate = 2000 # Newtons -> see excel calculations using gate parts, volumes, and densities.
# h_gate = 2.317 # meters from flap gate bottom to hinge. Assume weight is uniformly distributed.
# d_hinge_to_inv = 2.286
# hinge_el_open = inv_el_open+d_hinge_to_inv
# P_invert = 0.3048 # "weir" lip height

# # Sluice Gate Calculations (no variable coefficients like WHG and Spaulding and Grilli (2001) have used)
# # This is from Sluice-Gate Discharge Equations by Prabhata K. Swamee
# Q_dike_sluice_calc = np.zeros_like(HR_dike_lev_disch_m["datenum"]) # Add Q to this array (Add at each index the different culvert Qs)
# for i in range(len(HR_dike_lev_disch_m)):
#     H_sea_lev = HR_dike_lev_disch_m["Gage height, m, Ocean side"][i] - inv_el_open
#     y_d_HR_lev = HR_dike_lev_disch_m["Gage height, m, HR side"][i] - inv_el_open
#     crossover_sub_free_neg = 0.81*y_d_HR_lev*(y_d_HR_lev/y_sluice_open)**0.72 # from Swamee paper
#     crossover_sub_free_pos = 0.81*H_sea_lev*(H_sea_lev/y_sluice_open)**0.72
#     A_sluice_culv_HRside = (HR_dike_lev_disch_m["Gage height, m, HR side"][i] - inv_el_HRside)*L_sluice_culv
#     A_sluice_culv_oceanside = H_sea_lev*L_sluice_culv
#     if (H_sea_lev > y_d_HR_lev): # If sea level is greater than HR level -> Negative Flow
#         if (H_sea_lev > y_sluice_open): # If sea level is above sluice opening, apply sluice-gate discharge equations.
#             # High Tide Culvert Free Flow, Negative Direction (Flaps Closed)
#             if (H_sea_lev >= crossover_sub_free_neg):
#                 Q_dike_sluice_calc[i] = -0.864*A_sluice_open*math.sqrt(grav*H_sea_lev)*((H_sea_lev-y_sluice_open)/
#                            (H_sea_lev+15*y_sluice_open))**0.072
#             # High Tide Submerged Flow, Negative Direction (Flaps Closed)
#             if (H_sea_lev > y_d_HR_lev) & (H_sea_lev < crossover_sub_free_neg):
#                 Q_dike_sluice_calc[i] = -0.864*A_sluice_open*math.sqrt(grav*H_sea_lev)*((H_sea_lev-y_sluice_open)/
#                            (H_sea_lev+15*y_sluice_open))**0.072*(H_sea_lev-y_d_HR_lev)**0.7/(0.32*
#                            (0.81*y_d_HR_lev*(y_d_HR_lev/y_sluice_open)**0.72-H_sea_lev)**0.7+(H_sea_lev-y_d_HR_lev)**0.7)
#         else: # If H is less than y, assume underflow @ sluice and just use energy equation to determine Q(-).
#             # Do Manning for negative flow?
#             Q_dike_sluice_calc[i] = -math.sqrt((H_sea_lev-y_d_HR_lev)*2*grav*A_sluice_culv_HRside**2) # Assuming no head loss
#     elif (H_sea_lev <= y_d_HR_lev): # If sea level is less than HR level -> Positive Flow
#         if (y_d_HR_lev > y_sluice_open): # If HR level is above sluice opening, apply sluice-gate discharge equations.
#             # Low Tide Culvert Free Flow, Positive Direction (Flaps Open)
#             if (y_d_HR_lev >= crossover_sub_free_pos):
#                 Q_dike_sluice_calc[i] = 0.864*A_sluice_open*math.sqrt(grav*y_d_HR_lev)*((y_d_HR_lev-y_sluice_open)/
#                            (y_d_HR_lev+15*y_sluice_open))**0.072
#             # Low Tide Submerged Flow, Positive Direction (Flaps Open)
#             if (y_d_HR_lev > H_sea_lev) & (y_d_HR_lev < crossover_sub_free_pos):
#                 Q_dike_sluice_calc[i] = 0.864*A_sluice_open*math.sqrt(grav*y_d_HR_lev)*((y_d_HR_lev-y_sluice_open)/
#                            (y_d_HR_lev+15*y_sluice_open))**0.072*(y_d_HR_lev-H_sea_lev)**0.7/(0.32*
#                            (0.81*H_sea_lev*(H_sea_lev/y_sluice_open)**0.72-y_d_HR_lev)**0.7+(y_d_HR_lev-H_sea_lev)**0.7)
#         else: # If y_d is less than y, assume underflow @ sluice and just use energy equation to determine Q(+).
#             # Do weir? - only applies on HR discharge. Do Manning?
#             Q_dike_sluice_calc[i] = math.sqrt((y_d_HR_lev-H_sea_lev)*2*grav*A_sluice_culv_oceanside**2) # Assuming no head loss
#     else:
#         Q_dike_sluice_calc[i] = np.nan

# # Center Flap Gate Calculations
# Q_dike_centerflap_calc = np.zeros_like(HR_dike_lev_disch_m["datenum"])
# C_one = 1.375 # Discharge coefficient for supercritical weir flow
# C_two = 1.375 # Dischrage coefficient for subcritical weir flow
# # "Sluice" portion of flap gate only needs discharge coefficients during ebb tides (not flood, when tide moves inland)
# # Does ebb tide mean sluice upstream or downstream? Assume downstream -> smaller coefficients, smaller Q
# C_three = 0.6
# C_four = 0.8
# for i in range(len(HR_dike_lev_disch_m)):
#     d_hinge_to_H = hinge_el_open - HR_dike_lev_disch_m["Gage height, m, Ocean side"][i]
#     d_hinge_to_y_d = hinge_el_open - HR_dike_lev_disch_m["Gage height, m, HR side"][i]
#     H_sea_lev = HR_dike_lev_disch_m["Gage height, m, Ocean side"][i] - inv_el_open
#     y_d_HR_lev = HR_dike_lev_disch_m["Gage height, m, HR side"][i] - inv_el_open
#     A_center_flap_HRside = y_d_HR_lev*L_flaps_in
#     # A_center_flap_oceanside = H_sea_lev*L_flaps_out
#     # F_gate_HRside = 0.5*dens_seawater*grav*A_center_flap_HRside*y_d_HR_lev
#     # F_gate_oceanside = 0.5*dens_seawater*grav*A_center_flap_oceanside*H_sea_lev
#     # theta_center = np.arcsin((F_gate_HRside - F_gate_oceanside)/(W_gate))
#     # A_center_flap_open = complex geometry
    
#     # Using Newton Method (Need to fix )
#     # p = lambda theta: -W_gate*sin(theta-angle_init_flaps)*h_gate/dens_seawater/grav - L_flaps_out*(h_gate**2*
#     #                                   cos(theta-angle_init_flaps)**2 - 2*h_gate*d_hinge_to_H - d_hinge_to_H**2/
#     #                                   cos(theta-angle_init_flaps))*(h_gate-(1/3)*(h_gate-d_hinge_to_H/
#     #                                           cos(theta-angle_init_flaps))) + L_flaps_in*(h_gate**2*cos(theta+
#     #                                                   angle_init_flaps)**2-2*h_gate*d_hinge_to_y_d - d_hinge_to_y_d**2/
#     #                                           cos(theta-angle_init_flaps))*(h_gate-(1/3)*(h_gate - d_hinge_to_y_d/
#     #                                                                                          cos(theta-angle_init_flaps)))
#     # Dp = lambda theta: -W_gate*cos(theta-angle_init_flaps)*h_gate/dens_seawater/grav - L_flaps_out*((1/3)*(h_gate**2*
#     #                                   cos(theta-angle_init_flaps)**2 - 2*h_gate*d_hinge_to_H - d_hinge_to_H**2/
#     #                                   cos(theta-angle_init_flaps))*(d_hinge_to_H*tan(theta-angle_init_flaps)*
#     #                                           sec(theta-angle_init_flaps)) + (h_gate - (1/3)*(h_gate-d_hinge_to_H/
#     #                                                   cos(theta-angle_init_flaps)))*(-2*h_gate**2*sin(theta+
#     #                                                           angle_init_flaps)*cos(theta-angle_init_flaps)-d_hinge_to_H**2*
#     #                                                   tan(theta-angle_init_flaps)*
#     #                                                   sec(theta-angle_init_flaps)))+L_flaps_in*((1/3)*
#     #                                                   (h_gate**2*cos(theta-angle_init_flaps)**2-2*h_gate*
#     #                                                    d_hinge_to_y_d - d_hinge_to_y_d**2/
#     #                                                                cos(theta-angle_init_flaps))*(d_hinge_to_y_d*
#     #                                                                        tan(theta-angle_init_flaps)*sec(theta+
#     #                                                                                angle_init_flaps)) + (h_gate - (1/3)*
#     #                                                                                (h_gate-d_hinge_to_y_d/
#     #                                                                                 cos(theta-angle_init_flaps)))*
#     #                                                                                (-2*h_gate**2*sin(theta-angle_init_flaps)*
#     #                                                                                 cos(theta-angle_init_flaps)-
#     #                                                                                 d_hinge_to_y_d**2*
#     #                                                                                 tan(theta-angle_init_flaps)*
#     #                                                                                 sec(theta-angle_init_flaps)))
#     # approx = newton(p,Dp,0,1e-10,10)
#     # print(approx)
#     ### End Using Newton Method
    
#     # Using SciPy fsolve
#     def f(theta): 
#         # vn = theta
#         # np_sin = np.frompyfunc(mp.sin,1,1)
#         # np_cos = np.frompyfunc(mp.cos,1,1)
#         return -W_gate*np.sin(theta+angle_init_flaps)*h_gate/dens_seawater/grav - L_flaps_out*(h_gate**2*
#                                       np.cos(theta+angle_init_flaps)**2 - 2*h_gate*d_hinge_to_H*np.cos(theta+angle_init_flaps) + d_hinge_to_H**2/
#                                       np.cos(theta+angle_init_flaps))*(h_gate-(1/3)*(h_gate-d_hinge_to_H/
#                                               np.cos(theta+angle_init_flaps))) + L_flaps_in*(h_gate**2*np.cos(theta+
#                                                       angle_init_flaps)**2-2*h_gate*d_hinge_to_y_d*np.cos(theta+angle_init_flaps) + d_hinge_to_y_d**2/
#                                               np.cos(theta+angle_init_flaps))*(h_gate-(1/3)*(h_gate - d_hinge_to_y_d/
#                                                                                              np.cos(theta+angle_init_flaps)))                             
                                      
#     root = float(fsolve(f, 0)) # use root finder to find angle closest to zero (use ifs to deal with negative angles)
#     # potential issue with root finder: if there is a negative root closer to zero, but a positive root just slightly further away.
#     # may not be an issue since any negative roots would assume would be less than 5 degrees, keeping the gate closed, 
#     # and 1 is closer to 0 than -6 is, so it would just converge there if it had the opportunity (-6+5=-1, same result).
    
#     if h_gate*np.cos(root+angle_init_flaps) > d_hinge_to_H: # for gate to still be submerged, underflow is submerged sluice
#         if root <= 0: # if theta is less than or equal to zero, no flow (include leakiness?)
#             Q_dike_centerflap_calc[i] = 0
#         elif root > 0:
#             # calculate area that flow is passing through
#             A_both_sides = (d_hinge_to_y_d+y_d_HR_lev)**2*(np.tan(theta+angle_init_flaps)-np.tan(angle_init_flaps)) - d_hinge_to_y_d**2*(np.tan(theta+angle_init_flaps)-np.tan(angle_init_flaps))
#             A_under = L_flaps_out*(d_hinge_to_y_d+y_d_HR_lev)*(np.tan(theta+angle_init_flaps) - np.tan(angle_init_flaps))
#             A_sides_lower = (d_hinge_to_H+H_sea_lev)**2*(np.tan(theta+angle_init_flaps)-np.tan(angle_init_flaps)) - d_hinge_to_H**2*(np.tan(theta+angle_init_flaps)-np.tan(angle_init_flaps))
#             A_sides_upper = A_both_sides - A_sides_lower
#             A_sluice_calcs = A_sides_lower + A_under
#             A_weir_calcs = A_sides_upper
#             if y_d_HR_lev > H_sea_lev: # determine Q from weir equation (upper area) and submerged sluice (lower area) 
#                 # as long as HR levels are greater than sea levels. Assume area between invert and gate opening is smallest.
#                 # Too much flow with no losses
#                 # Q_dike_centerflap_calc[i] = np.sqrt((y_d_HR_lev-H_sea_lev)*2*grav*A_tot**2)
#                 # Q_dike_centerflap_calc[i] = np.sqrt((y_d_HR_lev-H_sea_lev)*2*grav*A_center_flap_HRside**2)
#                 if (y_d_HR_lev < L_flaps_in) & (H_sea_lev/y_d_HR_lev < 0.66): # supercritical weir flow, from WHG report
#                     # Q_weir_part = C_one*A_weir_calcs/(y_d_HR_lev-H_sea_lev)*(2/3)*np.sqrt((2/3)*grav)*y_d_HR_lev**(2/3)
#                     # Q_sluice_part = C_four*A_sluice_calcs*np.sqrt(2*grav*(y_d_HR_lev-H_sea_lev))
#                     # Q_dike_centerflap_calc[i] = Q_weir_part + Q_sluice_part
#                 else: # subcritical weir flow
#                     # Q_weir_part = C_two*A_weir_calcs/(y_d_HR_lev-H_sea_lev)*H_sea_lev*np.sqrt(2*grav*(y_d_HR_lev-H_sea_lev))
#                     # Q_sluice_part = C_four*A_sluice_calcs*np.sqrt(2*grav*(y_d_HR_lev-H_sea_lev))
#                     # Q_dike_centerflap_calc[i] = Q_weir_part + Q_sluice_part
#             else:
#                 Q_dike_centerflap_calc[i] = np.nan
#         else: # root is a nan
#              Q_dike_centerflap_calc[i] = np.nan
#     else: # assume gate maximum opening is the surface of the water on the ocean side, underflow becomes free sluice
#         root = np.arccos(d_hinge_to_H/h_gate)-angle_init_flaps
#         # calculate area that flow is passing through
#         A_both_sides = (d_hinge_to_y_d+y_d_HR_lev)**2*np.tan(theta+angle_init_flaps)-(d_hinge_to_y_d+y_d_HR_lev)**2*np.tan(angle_init_flaps)-d_hinge_to_y_d**2*np.tan(theta+angle_init_flaps)-d_hinge_to_y_d**2*np.tan(angle_init_flaps)
#         A_under = L_flaps_out*(d_hinge_to_y_d+y_d_HR_lev)*(np.tan(theta+angle_init_flaps) - np.tan(angle_init_flaps))
#         A_sides_lower = (d_hinge_to_H+H_sea_lev)**2*(np.tan(theta+angle_init_flaps)-np.tan(angle_init_flaps)) - d_hinge_to_H**2*(np.tan(theta+angle_init_flaps)-np.tan(angle_init_flaps))
#         A_sides_upper = A_both_sides - A_sides_lower
#         A_sluice_calcs = A_sides_lower + A_under
#         A_weir_calcs = A_sides_upper
#         if y_d_HR_lev > H_sea_lev: # determine Q from weir equation (upper area) and free sluice (lower area) 
#             # as long as HR levels are greater than sea levels. Assume area between invert and gate opening is smallest.
#             # Too much flow with no losses
#             # Q_dike_centerflap_calc[i] = np.sqrt((y_d_HR_lev-H_sea_lev)*2*grav*A_tot**2)
#             # Q_dike_centerflap_calc[i] = np.sqrt((y_d_HR_lev-H_sea_lev)*2*grav*A_center_flap_HRside**2)
#             if (y_d_HR_lev < L_flaps_in) & (H_sea_lev/y_d_HR_lev < 0.66): # supercritical weir flow, from WHG report
#                 # Q_weir_part = C_one*A_weir_calcs/(y_d_HR_lev-H_sea_lev)*(2/3)*np.sqrt((2/3)*grav)*y_d_HR_lev**(2/3)
#                 # Q_sluice_part = C_three*A_sluice_calcs*np.sqrt(2*grav*y_d_HR_lev)
#                 # Q_dike_centerflap_calc[i] = Q_weir_part + Q_sluice_part
#             else: # subcritical weir flow
#                 # Q_weir_part = C_two*A_weir_calcs/(y_d_HR_lev-H_sea_lev)*H_sea_lev*np.sqrt(2*grav*(y_d_HR_lev-H_sea_lev))
#                 # Q_sluice_part = C_three*A_sluice_calcs*np.sqrt(2*grav*y_d_HR_lev)
#                 # Q_dike_centerflap_calc[i] = Q_weir_part + Q_sluice_part
#         else:
#             Q_dike_centerflap_calc[i] = np.nan
#     # if H < y_d, use algorithm to determine angle, else use "leakiness"?
#     # or condition if negative or domain error? check some points.
    
    
    
# # Left Flap Gate Has Same Conditions as Center (smaller culvert, but same gate size)
# Q_dike_leftflap_calc = Q_dike_centerflap_calc.copy()

# Q_total = Q_dike_leftflap_calc + Q_dike_centerflap_calc + Q_dike_sluice_calc
    
# # Should I be using Manning instead of Energy Eqn to determine Q for open-channel flow through dike?
    
#%% Analytical Estimation of Discharge Through Dike Using Water Levels, My Analysis (all SI)

# Add option for different configurations (number/size/type of openings)?

"""
Sources:
Sluice-gate Discharge Equations by Prabhata K. Swamee, Journal of Irrigation and Drainage Engineering, Vol. 118
Herring River Full Final Report, Woods Hole Group June 2012
Hydrodynamic and Salinity Modeling for Estuarine Habitat Restoration at HR, Wellfleet, MA. Spaulding and Grilli October 2001
    (Higher frictional losses on the ebb than on the flood tide, pp. ii, and n~0.06 to 0.09 for HR bed)
    *Loss coefficients hard to justify given difference in distances between the HR basin (S&G) and measurements around the dike*
    
Can solve for the "additional coefficient" (make a K array) at each point by dividing the measured discharge by everything on the RHS.
Need to make several K arrays - one for each scenario, and take the average K of each as the fitting parameter.
"""
# slope_culv = 0.0067
# len_culv = 20.42
# L_center_culv = 2.184
# L_left_culv = 2.007
# P_invert = 0.3048 # "weir" lip height

inv_el_open = -1.064
inv_el_HRside = -0.928
sluice_bot_el = -0.579
y_sluice_open = sluice_bot_el-inv_el_open
L_sluice_culv = 1.829
A_sluice_open = y_sluice_open*L_sluice_culv
L_flaps_in = 1.829
L_flaps_out = 2.057
angle_init_flaps = 0.0872 # radians, ~ 5 degrees
dens_seawater = 1018 # kg/m^3, average is roughly the same on both sides of the dike.
grav = 9.81 # m/s^2
W_gate = 2000 # Newtons -> see excel calculations using gate parts, volumes, and densities.
h_gate = 2.317 # meters from flap gate bottom to hinge. Assume weight is uniformly distributed.
d_hinge_to_inv = 2.286
hinge_el_open = inv_el_open+d_hinge_to_inv
# HL_max = 0.6 # maximum headloss, meters, from WHG report
HL_max = 0.9 # 1.17 # maximum headloss, meters, tester (assumed to be maximum difference in levels (HR - ocean))
HLsluice_max = 1.0 # maximum sluice headloss/gain, meters, tests (assumed at ~ half maximum difference in levels (ocean - HR))
# D_HL = 0.884 # headloss parameter, meters, from WHG report
D_HL = 0.4 # 0.41 # headloss parameter, meters, tester (mean of the means of HR and Ocean levels)
Dsluice_HL = 1.0 # optimize
# n = 0.01
# C_d_ebb_update = 1
# C_d_ebb_std_array = []
# y_discharge_calc_maxes_mean_array = []
# while D_HL > 0.2:    # For optimizing flap gate head loss coefficients.
#     D_HL = D_HL - n

# Initialize Discharge Arrays and set to nans
Q_flood_free = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_flood_transit = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_flood_submer_or = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_flood_subcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_flood_supcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_free = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_transit = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_submer_or = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_subcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_supcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_flap_subcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_flap_supcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_flood_free[:] = np.nan
Q_flood_transit[:] = np.nan
Q_flood_submer_or[:] = np.nan
Q_flood_subcrit_weir[:] = np.nan
Q_flood_supcrit_weir[:] = np.nan
Q_ebb_free[:] = np.nan
Q_ebb_transit[:] = np.nan
Q_ebb_submer_or[:] = np.nan
Q_ebb_subcrit_weir[:] = np.nan
Q_ebb_supcrit_weir[:] = np.nan
Q_ebb_flap_subcrit_weir[:] = np.nan
Q_ebb_flap_supcrit_weir[:] = np.nan

# Initialize Discharge Coefficient Arrays and set to nans
C_Swamee = np.zeros_like(HR_dike_lev_disch_m["datenum"]) # This is from Free Flow Sluice-Gate C_d by Prabhata K. Swamee
C_d_flood_free = np.zeros_like(HR_dike_lev_disch_m["datenum"]) # This is in addition to the Swamee coefficient.
C_d_flood_transit = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_flood_submer_or = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_flood_subcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_flood_supcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_ebb_free = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_ebb_transit = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_ebb_submer_or = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_ebb_subcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_ebb_supcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_ebb_flap_subcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_ebb_flap_supcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_Swamee[:] = np.nan
C_d_flood_free[:] = np.nan
C_d_flood_transit[:] = np.nan
C_d_flood_submer_or[:] = np.nan
C_d_flood_subcrit_weir[:] = np.nan
C_d_flood_supcrit_weir[:] = np.nan
C_d_ebb_free[:] = np.nan
C_d_ebb_transit[:] = np.nan
C_d_ebb_submer_or[:] = np.nan
C_d_ebb_subcrit_weir[:] = np.nan
C_d_ebb_supcrit_weir[:] = np.nan
C_d_ebb_flap_subcrit_weir[:] = np.nan
C_d_ebb_flap_supcrit_weir[:] = np.nan

theta_ebb_flap_deg = np.zeros_like(HR_dike_lev_disch_m["datenum"])
theta_ebb_flap_deg[:] = np.nan

HL = np.zeros_like(HR_dike_lev_disch_m["datenum"])
HL[:] = np.nan
HLsluice = np.zeros_like(HR_dike_lev_disch_m["datenum"])
HLsluice[:] = np.nan

flow_frac_sluice_culv = np.zeros_like(HR_dike_lev_disch_m["datenum"])
flow_frac_sluice_culv[:] = np.nan
flow_frac_center_culv = np.zeros_like(HR_dike_lev_disch_m["datenum"])
flow_frac_center_culv[:] = np.nan
flow_frac_left_culv = flow_frac_center_culv.copy()

for i in range(len(HR_dike_lev_disch_m)):
    # Levels relative to culvert invert at sluice/flaps.
    H_sea_lev = HR_dike_lev_disch_m["Gage height, m, Ocean side"][i] - inv_el_open
    y_d_HR_lev = HR_dike_lev_disch_m["Gage height, m, HR side"][i] - inv_el_open
    # Vertical distances from flap gate hinge to water levels.
    d_hinge_to_H = hinge_el_open - HR_dike_lev_disch_m["Gage height, m, Ocean side"][i]
    d_hinge_to_y_d = hinge_el_open - HR_dike_lev_disch_m["Gage height, m, HR side"][i]
    if (H_sea_lev > y_d_HR_lev): # If sea level is greater than HR level -> Negative Flow (Flood Tide, Flap Gates Closed)
        """
        Test: Supercritical Broad-crested Weir/Free Sluice, Transitional, Subcritical Broad-crested Weir/Submerged Orifice
        """
        if (y_d_HR_lev/H_sea_lev < (2/3)): # supercritical BC weir/free sluice
            if (H_sea_lev < y_sluice_open): # Supercritical Broad-crested Weir Flow
                Q_flood_supcrit_weir[i] = -(2/3)*L_sluice_culv*H_sea_lev*np.sqrt((2/3)*grav*H_sea_lev)
                C_d_flood_supcrit_weir[i] = HR_dike_lev_disch_m["Discharge, cms"][i]/Q_flood_supcrit_weir[i]
            else: # Free Sluice Flow
                HLsluice[i] = HLsluice_max*(1-0.5*(y_d_HR_lev+H_sea_lev)/Dsluice_HL)
                C_Swamee[i] = 0.611*((H_sea_lev-y_d_HR_lev)/(H_sea_lev+15*y_d_HR_lev))**0.072
                Q_flood_free[i] = -A_sluice_open*np.sqrt(2*grav*(H_sea_lev-HLsluice[i]))
                C_d_flood_free[i] = HR_dike_lev_disch_m["Discharge, cms"][i]/Q_flood_free[i]
        else:
            if (H_sea_lev < y_sluice_open): # Subcritical Broad-crested Weir Flow
                Q_flood_subcrit_weir[i] = -L_sluice_culv*y_d_HR_lev*np.sqrt(2*grav*(H_sea_lev-y_d_HR_lev))
                C_d_flood_subcrit_weir[i] = HR_dike_lev_disch_m["Discharge, cms"][i]/Q_flood_subcrit_weir[i]
            elif (y_d_HR_lev/H_sea_lev > 0.8): # Submerged Orifice Flow
                Q_flood_submer_or[i] = -A_sluice_open*np.sqrt(2*grav*(H_sea_lev-y_d_HR_lev))
                C_d_flood_submer_or[i] = HR_dike_lev_disch_m["Discharge, cms"][i]/Q_flood_submer_or[i]
            else: # Transitional Flow
                Q_flood_transit[i] = -A_sluice_open*np.sqrt(2*grav*3*(H_sea_lev-y_d_HR_lev))
                C_d_flood_transit[i] = HR_dike_lev_disch_m["Discharge, cms"][i]/Q_flood_transit[i]
    else: # If sea level is less than HR level -> Positive Flow (Ebb Tide, Flap Gates Open)
        # Center Flap Gate Calculations
        A_center_flap_HRside = y_d_HR_lev*L_flaps_in
        A_center_flap_oceanside = H_sea_lev*L_flaps_out # Should L change?
        # Using SciPy fsolve
        def f(theta): 
            return -W_gate*np.sin(theta+angle_init_flaps)*h_gate/dens_seawater/grav - L_flaps_out*(h_gate**2*
                                          np.cos(theta+angle_init_flaps)**2 - 2*h_gate*d_hinge_to_H*np.cos(theta+angle_init_flaps) + d_hinge_to_H**2/
                                          np.cos(theta+angle_init_flaps))*(h_gate-(1/3)*(h_gate-d_hinge_to_H/
                                                  np.cos(theta+angle_init_flaps))) + L_flaps_in*(h_gate**2*np.cos(theta+
                                                          angle_init_flaps)**2-2*h_gate*d_hinge_to_y_d*np.cos(theta+angle_init_flaps) + d_hinge_to_y_d**2/
                                                  np.cos(theta+angle_init_flaps))*(h_gate-(1/3)*(h_gate - d_hinge_to_y_d/
                                                                                                 np.cos(theta+angle_init_flaps)))                                  
        root = float(fsolve(f, 0)) # use root finder to find angle closest to zero
        theta_ebb_flap_deg[i] = np.rad2deg(root)
        # Flow fractions of total measured discharge through each culvert (NEED TO OPTIMIZE)
        """
        Test: Supercritical/Free Sluice, Transitional, Subcritical/Submerged Orifice
        """
        if (H_sea_lev/y_d_HR_lev < (2/3)): # supercritical BC weir/free sluice - OPTIMIZE COEFFIENT BETWEEN FLAPS AND SLUICE!
            if (root > 0):
                HL[i] = HL_max*(1-0.5*(y_d_HR_lev+H_sea_lev)/D_HL)
                Q_ebb_flap_supcrit_weir[i] = (2/3)*(y_d_HR_lev+HL[i])*L_flaps_in*np.sqrt((2/3)*grav*(y_d_HR_lev+HL[i]))
            if (y_d_HR_lev < y_sluice_open): # Supercritical Broad-crested Weir Flow
                Q_ebb_supcrit_weir[i] = (2/3)*L_sluice_culv*y_d_HR_lev*np.sqrt((2/3)*grav*y_d_HR_lev)
            else: # Free Sluice Flow
                C_Swamee[i] = 0.611*((y_d_HR_lev-H_sea_lev)/(y_d_HR_lev+15*H_sea_lev))**0.072
                Q_ebb_free[i] = A_sluice_open*np.sqrt(2*grav*y_d_HR_lev)
        else: #  subcritical BC weir/submerged orifice - OPTIMIZE COEFFIENT BETWEEN FLAPS AND SLUICE!
            if (root > 0):
                HL[i] = HL_max*(1-0.5*(y_d_HR_lev+H_sea_lev)/D_HL)
                Q_ebb_flap_subcrit_weir[i] = A_center_flap_oceanside*np.sqrt(2*grav*((y_d_HR_lev+HL[i])-H_sea_lev))
            if (y_d_HR_lev < y_sluice_open): # Subcritical Broad-crested Weir Flow
                Q_ebb_subcrit_weir[i] = L_sluice_culv*H_sea_lev*np.sqrt(2*grav*(y_d_HR_lev-H_sea_lev))
            elif (H_sea_lev/y_d_HR_lev > 0.8): # Submerged Orifice Flow
                Q_ebb_submer_or[i] = A_sluice_open*np.sqrt(2*grav*(y_d_HR_lev-H_sea_lev))
            else: # Transitional Flow
                Q_ebb_transit[i] = A_sluice_open*np.sqrt(2*grav*3*(y_d_HR_lev-H_sea_lev))
        flow_sluice_culv = np.nansum((Q_ebb_free[i],Q_ebb_transit[i],Q_ebb_submer_or[i],Q_ebb_supcrit_weir[i],Q_ebb_subcrit_weir[i]))
        flow_flap_culv = np.nansum((Q_ebb_flap_supcrit_weir[i],Q_ebb_flap_subcrit_weir[i]))
        flow_frac_sluice_culv[i] = flow_sluice_culv/(flow_sluice_culv+2*flow_flap_culv)
        flow_frac_center_culv[i] = flow_flap_culv/(flow_sluice_culv+2*flow_flap_culv)
        flow_frac_left_culv[i] = flow_frac_center_culv[i]
        if (H_sea_lev/y_d_HR_lev < (2/3)): # supercritical BC weir/free sluice - OPTIMIZE COEFFIENT BETWEEN FLAPS AND SLUICE!
            if (root > 0):
                C_d_ebb_flap_supcrit_weir[i] = flow_frac_center_culv[i]*HR_dike_lev_disch_m["Discharge, cms"][i]/Q_ebb_flap_supcrit_weir[i]
            if (y_d_HR_lev < y_sluice_open): # Supercritical Broad-crested Weir Flow
                C_d_ebb_supcrit_weir[i] = flow_frac_sluice_culv[i]*HR_dike_lev_disch_m["Discharge, cms"][i]/Q_ebb_supcrit_weir[i]
            else: # Free Sluice Flow
                C_d_ebb_free[i] = flow_frac_sluice_culv[i]*HR_dike_lev_disch_m["Discharge, cms"][i]/Q_ebb_free[i]
        else: #  subcritical BC weir/submerged orifice - OPTIMIZE COEFFIENT BETWEEN FLAPS AND SLUICE!
            if (root > 0):
                C_d_ebb_flap_subcrit_weir[i] = flow_frac_center_culv[i]*HR_dike_lev_disch_m["Discharge, cms"][i]/Q_ebb_flap_subcrit_weir[i]
            if (y_d_HR_lev < y_sluice_open): # Subcritical Broad-crested Weir Flow
                C_d_ebb_subcrit_weir[i] = flow_frac_sluice_culv[i]*HR_dike_lev_disch_m["Discharge, cms"][i]/Q_ebb_subcrit_weir[i]
            elif (H_sea_lev/y_d_HR_lev > 0.8): # Submerged Orifice Flow
                if (HR_dike_lev_disch_m["Discharge, cms"][i] > 0):
                    C_d_ebb_submer_or[i] = flow_frac_sluice_culv[i]*HR_dike_lev_disch_m["Discharge, cms"][i]/Q_ebb_submer_or[i]
            else: # Transitional Flow
                C_d_ebb_transit[i] = flow_frac_sluice_culv[i]*HR_dike_lev_disch_m["Discharge, cms"][i]/Q_ebb_transit[i]

"""
Ebb C_d means and stdevs.
"""
C_d_ebb_free_mean = np.nanmean(C_d_ebb_free) + 0.05
C_d_ebb_transit_mean = np.nanmean(C_d_ebb_transit)
C_d_ebb_submer_or_mean = np.nanmean(C_d_ebb_submer_or)
C_d_ebb_subcrit_weir_mean = np.nanmean(C_d_ebb_subcrit_weir)
C_d_ebb_supcrit_weir_mean = np.nanmean(C_d_ebb_supcrit_weir)
C_d_ebb_flap_subcrit_weir_mean = np.nanmean(C_d_ebb_flap_subcrit_weir)
C_d_ebb_flap_supcrit_weir_mean = np.nanmean(C_d_ebb_flap_supcrit_weir) + 0.05

C_d_ebb_free_std = np.nanstd(C_d_ebb_free)
C_d_ebb_transit_std = np.nanstd(C_d_ebb_transit)
C_d_ebb_submer_or_std = np.nanstd(C_d_ebb_submer_or)
C_d_ebb_subcrit_weir_std = np.nanstd(C_d_ebb_subcrit_weir)
C_d_ebb_supcrit_weir_std = np.nanstd(C_d_ebb_supcrit_weir)
C_d_ebb_flap_subcrit_weir_std = np.nanstd(C_d_ebb_flap_subcrit_weir)
C_d_ebb_flap_supcrit_weir_std = np.nanstd(C_d_ebb_flap_supcrit_weir)                                                        
    
    # C_d_ebb_std_peak = C_d_ebb_flap_supcrit_weir_std + C_d_ebb_free_std # optimizing coefficients continued.
    
    # C_d_ebb_std_array.append(C_d_ebb_std_peak)
    # if (C_d_ebb_std_peak > C_d_ebb_update):
    #     break
    # else:
    #     C_d_ebb_update = C_d_ebb_std_peak
        

"""
Flood C_d means and stdevs.
"""
C_d_flood_free_mean = np.nanmean(C_d_flood_free) + 0.1
C_d_flood_transit_mean = np.nanmean(C_d_flood_transit)
C_d_flood_submer_or_mean = np.nanmean(C_d_flood_submer_or)
C_d_flood_subcrit_weir_mean = np.nanmean(C_d_flood_subcrit_weir)
C_d_flood_supcrit_weir_mean = np.nanmean(C_d_flood_supcrit_weir)

C_d_flood_free_std = np.nanstd(C_d_flood_free)
C_d_flood_transit_std = np.nanstd(C_d_flood_transit)
C_d_flood_submer_or_std = np.nanstd(C_d_flood_submer_or)
C_d_flood_subcrit_weir_std = np.nanstd(C_d_flood_subcrit_weir)
C_d_flood_supcrit_weir_std = np.nanstd(C_d_flood_supcrit_weir)

"""
Coefficients from Swamee Paper and WHG Report
"""
C_Swamee_mean = np.nanmean(C_Swamee)
C_Swamee_std = np.nanstd(C_Swamee)
C_one_flood = 1.375 # Discharge coefficient for supercritical b-c weir flow
C_two_flood = 1.375 # Dischrage coefficient for subcritical b-c weir flow
C_three_flood = 1.4 # Discharge coefficient for free sluice flow
C_four_flood = 1.35 # Discharge coefficient for submerged orifice flow
C_one_ebb = 1
C_two_ebb = 1
C_three_ebb = 0.6
C_four_ebb = 0.8    

"""
Total Flow
"""
Q_flood_free_adj = C_d_flood_free_mean*Q_flood_free
Q_flood_transit_adj = C_d_flood_transit_mean*Q_flood_transit
Q_flood_submer_or_adj = C_d_flood_submer_or_mean*Q_flood_submer_or
Q_flood_subcrit_weir_adj = C_d_flood_subcrit_weir_mean*Q_flood_subcrit_weir
Q_flood_supcrit_weir_adj = C_d_flood_supcrit_weir_mean*Q_flood_supcrit_weir
Q_ebb_free_adj = C_d_ebb_free_mean*Q_ebb_free
Q_ebb_transit_adj = C_d_ebb_transit_mean*Q_ebb_transit
Q_ebb_submer_or_adj = C_d_ebb_submer_or_mean*Q_ebb_submer_or
Q_ebb_subcrit_weir_adj = C_d_ebb_subcrit_weir_mean*Q_ebb_subcrit_weir
Q_ebb_supcrit_weir_adj = C_d_ebb_supcrit_weir_mean*Q_ebb_supcrit_weir
Q_ebb_flap_subcrit_weir_adj = C_d_ebb_flap_subcrit_weir_mean*Q_ebb_flap_subcrit_weir
Q_ebb_flap_supcrit_weir_adj = C_d_ebb_flap_supcrit_weir_mean*Q_ebb_flap_supcrit_weir

# Add Q to this array (Add at each index the different culvert Qs)
Q_dike_sluice_calc_flood = np.nansum((Q_flood_free_adj,Q_flood_transit_adj,Q_flood_submer_or_adj),axis=0)
Q_dike_sluice_weir_calc_flood = np.nansum((Q_flood_subcrit_weir_adj,Q_flood_supcrit_weir_adj),axis=0)
Q_dike_sluice_calc_ebb = np.nansum((Q_ebb_free_adj,Q_ebb_transit_adj,Q_ebb_submer_or_adj),axis=0)
Q_dike_sluice_weir_calc_ebb = np.nansum((Q_ebb_subcrit_weir_adj,Q_ebb_supcrit_weir_adj),axis=0)

Q_dike_sluice_calc = Q_dike_sluice_calc_flood+Q_dike_sluice_weir_calc_flood+Q_dike_sluice_calc_ebb+Q_dike_sluice_weir_calc_ebb

Q_dike_centerflap_calc = np.nansum((Q_ebb_flap_subcrit_weir_adj,Q_ebb_flap_supcrit_weir_adj),axis=0)
    
# Left Flap Gate Has Same Conditions as Center (smaller culvert, but same gate size)
Q_dike_leftflap_calc = Q_dike_centerflap_calc.copy()

Q_total = Q_dike_leftflap_calc + Q_dike_centerflap_calc + Q_dike_sluice_calc

Q_total[Q_total==0] = np.nan

tidal_peaktopeak_interval = 12/24 + 25/(60*24) # bin width in days

# Max/Min/Range of discharge through dike
bin_start = 0
x_discharge_rangedates = []    
y_discharge_calc_mins = []
y_discharge_calc_maxes = []
y_discharge_meas_mins = []
y_discharge_meas_maxes = []
for bin_index in range(len(x_datenum_dike)):
    datestart = x_datenum_dike[bin_start]
    dateend = datestart + (x_datenum_dike[bin_index] - x_datenum_dike[bin_start])
    date_interval = dateend - datestart
    bin_end = bin_index
    if (date_interval >= tidal_peaktopeak_interval):
            x_discharge_rangedates.append(x_datenum_dike[int((bin_start+bin_end)/2)])
            y_discharge_calc_mins.append(np.nanmin(Q_total[bin_start:bin_end]))
            y_discharge_calc_maxes.append(np.nanmax(Q_total[bin_start:bin_end]))
            y_discharge_meas_mins.append(np.nanmin(HR_dike_lev_disch_m["Discharge, cms"][bin_start:bin_end]))
            y_discharge_meas_maxes.append(np.nanmax(HR_dike_lev_disch_m["Discharge, cms"][bin_start:bin_end]))
            bin_start = bin_end
x_discharge_rangedates = np.array(x_discharge_rangedates)
y_discharge_calc_mins = np.array(y_discharge_calc_mins)
y_discharge_calc_maxes = np.array(y_discharge_calc_maxes)
y_discharge_calc_mins[y_discharge_calc_mins > np.nanmean(y_discharge_calc_maxes)] = np.nan
y_discharge_calc_maxes[y_discharge_calc_maxes < np.nanmean(y_discharge_calc_mins)] = np.nan
y_discharge_calc_ranges = y_discharge_calc_maxes - y_discharge_calc_mins
y_discharge_meas_mins = np.array(y_discharge_meas_mins)
y_discharge_meas_maxes = np.array(y_discharge_meas_maxes)
y_discharge_meas_mins[y_discharge_meas_mins > np.nanmean(y_discharge_meas_maxes)] = np.nan
y_discharge_meas_maxes[y_discharge_meas_maxes < np.nanmean(y_discharge_meas_mins)] = np.nan
y_discharge_meas_ranges = y_discharge_meas_maxes - y_discharge_meas_mins

y_discharge_calc_maxes_ovrlp_mean = np.nanmean(y_discharge_calc_maxes[61:66])
y_discharge_meas_maxes_ovrlp_mean = np.nanmean(y_discharge_meas_maxes[61:66])
y_discharge_calc_mins_ovrlp_mean = np.nanmean(y_discharge_calc_mins[61:66])
y_discharge_meas_mins_ovrlp_mean = np.nanmean(y_discharge_meas_mins[61:66])

y_discharge_calc_maxes_mean = np.nanmean(y_discharge_calc_maxes)
y_discharge_meas_maxes_mean = np.nanmean(y_discharge_meas_maxes)
y_discharge_calc_mins_mean = np.nanmean(y_discharge_calc_mins)
y_discharge_meas_mins_mean = np.nanmean(y_discharge_meas_mins)
    
    # y_discharge_calc_maxes_mean_array.append(y_discharge_calc_maxes_ovrlp_mean)
    # if (y_discharge_calc_maxes_ovrlp_mean > y_discharge_meas_maxes_ovrlp_mean):
    #     break

"""
Condition for optimization: Q_total[i] = HR_dike_lev_disch_m["Discharge, cms"][i]
"""

"""
Plots
"""
ax = HR_dike_lev_disch_m.plot.scatter(x="datenum", y="Discharge, cms", color='Turquoise', label = 'Discharge, cms')
plt.scatter(x_datenum_dike, Q_total, label = 'Calculated Discharge, cms')

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=22)
plt.ylabel('Discharge (m^3/s)', fontsize=22)
plt.legend(loc='upper right', bbox_to_anchor=(0.9,0.4))
    
# Should I be using Manning instead of Energy Eqn to determine Q for open-channel flow through dike?

#%% Analytical Estimation of Discharge Through Dike Using Water Levels, WHG Report Analysis (all SI)

# Add option for different configurations (number/size/type of openings)?

inv_el_open = -1.064
slope_culv = 0.0067
len_culv = 20.42
inv_el_HRside = -0.928
sluice_bot_el = -0.579
y_sluice_open = sluice_bot_el-inv_el_open
A_sluice_open = y_sluice_open*L_sluice_culv
L_sluice_culv = 1.829
L_center_culv = 2.184
L_left_culv = 2.007
L_flaps_in = 1.829
L_flaps_out = 2.057
angle_init_flaps = 0.0872 # radians, ~ 5 degrees
dens_seawater = 1018 # kg/m^3, average is roughly the same on both sides of the dike.
grav = 9.81 # m/s^2
W_gate = 2000 # gate weight, Newtons -> see excel calculations using gate parts, volumes, and densities.
h_gate = 2.317 # meters from flap gate bottom to hinge. Assume weight is uniformly distributed.
d_hinge_to_inv = 2.286
hinge_el_open = inv_el_open+d_hinge_to_inv
C_one_flood = 1.375
C_two_flood = 1.375
C_three_flood = 1.4
C_four_flood = 1.35
C_one_ebb = 1
C_two_ebb = 1
C_three_ebb = 0.6
C_four_ebb = 0.8
HL_max = 0.6 # maximum headloss, meters
D_HL = 0.884 # headloss parameter, meters

# Sluice Gate Calculations (no variable coefficients like WHG and Spaulding and Grilli (2001) have used)
# Q_supcrit_weir equation is wrong in WHG report.
Q_dike_sluice_calc_WHG = np.zeros_like(HR_dike_lev_disch_m["datenum"]) # Add Q to this array (Add at each index the different culvert Qs)
Q_dike_centerflap_calc_WHG = np.zeros_like(HR_dike_lev_disch_m["datenum"])
for i in range(len(HR_dike_lev_disch_m)):
    H_sea_lev = HR_dike_lev_disch_m["Gage height, m, Ocean side"][i] - inv_el_open
    y_d_HR_lev = HR_dike_lev_disch_m["Gage height, m, HR side"][i] - inv_el_open
    if (H_sea_lev > y_d_HR_lev): # If sea level is greater than HR level -> Negative Flow
        Q_supcrit_weir = -C_one_flood*L_sluice_culv*(2/3)*np.sqrt((2/3)*grav)*H_sea_lev**(3/2)
        Q_subcrit_weir = -C_two_flood*L_sluice_culv*(y_d_HR_lev)*np.sqrt(2*grav*(H_sea_lev-y_d_HR_lev))
        Q_free_sluice = -C_three_flood*L_sluice_culv*y_sluice_open*np.sqrt(2*grav*H_sea_lev)
        Q_sub_orifice = -C_four_flood*L_sluice_culv*y_sluice_open*np.sqrt(2*grav*(H_sea_lev-y_d_HR_lev))
        """
        Compare Upstream Head (H_sea_lev) to Downstream Head (y_d_HR_lev)
        """
        if (y_d_HR_lev/H_sea_lev < 0.64): # Supercritical
            """
            Compare Upstream Head (H_sea_lev) to Gate Opening (y_sluice_open)
            """
            if (H_sea_lev > 1.25*y_sluice_open): # Free sluice flow
                Q_dike_sluice_calc_WHG[i] = Q_free_sluice
            elif (H_sea_lev < y_sluice_open): # Supercritical weir flow
                Q_dike_sluice_calc_WHG[i] = Q_supcrit_weir
            else: # Weighted average of supercritical weir and free sluice flow
                Q_dike_sluice_calc_WHG[i] = (Q_free_sluice+Q_supcrit_weir)/2 # This is just the average - how to weight?
        elif (y_d_HR_lev/H_sea_lev > 0.68): # Subcritical
            """
            Compare Upstream Head (H_sea_lev) to Gate Opening (y_sluice_open)
            """
            if (H_sea_lev > 1.25*y_sluice_open): # Submerged orifice flow
                Q_dike_sluice_calc_WHG[i] = Q_sub_orifice
            elif (H_sea_lev < y_sluice_open): # Subcritical weir flow
                Q_dike_sluice_calc_WHG[i] = Q_subcrit_weir
            else: # Weighted average of subcritical weir and submerged orifice flow
                Q_dike_sluice_calc_WHG[i] = (Q_sub_orifice+Q_subcrit_weir)/2 # This is just the average - how to weight?
        else: # Weighted average of Supercritical and Subcritical
            """
            Compare Upstream Head (H_sea_lev) to Gate Opening (y_sluice_open)
            """
            if (H_sea_lev > 1.25*y_sluice_open): # Weighted average of free sluice and submerged orifice flow
                Q_dike_sluice_calc_WHG[i] = (Q_free_sluice+Q_sub_orifice)/2 # This is just the average - how to weight?
            elif (H_sea_lev < y_sluice_open): # Weighted average of supercritical weir and subcritical weir flow
                Q_dike_sluice_calc_WHG[i] = (Q_supcrit_weir+Q_subcrit_weir)/2 # This is just the average - how to weight?
            else: # Weighted average of weighted averages of weir and sluice flow.
                Q_dike_sluice_calc_WHG[i] = ((Q_free_sluice+Q_sub_orifice)/2+(Q_supcrit_weir+Q_subcrit_weir)/2)/2
        Q_dike_centerflap_calc_WHG[i] = 0
    elif (H_sea_lev <= y_d_HR_lev): # If sea level is less than HR level -> Positive Flow
        Q_supcrit_weir = C_one_ebb*L_sluice_culv*(2/3)*np.sqrt((2/3)*grav)*y_d_HR_lev**(3/2)
        Q_subcrit_weir = C_two_ebb*L_sluice_culv*(H_sea_lev)*np.sqrt(2*grav*(y_d_HR_lev-H_sea_lev))
        Q_free_sluice = C_three_ebb*L_sluice_culv*y_sluice_open*np.sqrt(2*grav*y_d_HR_lev)
        Q_sub_orifice = C_four_ebb*L_sluice_culv*y_sluice_open*np.sqrt(2*grav*(y_d_HR_lev-H_sea_lev))
        HL = HL_max*(1-0.5*(y_d_HR_lev+H_sea_lev)/D_HL)
        Q_supcrit_weir_flap = C_one_ebb*L_flaps_in*(2/3)*np.sqrt((2/3)*grav)*(y_d_HR_lev+HL)**(3/2)
        Q_subcrit_weir_flap = C_two_ebb*L_flaps_in*(H_sea_lev)*np.sqrt(2*grav*((y_d_HR_lev+HL)-H_sea_lev))
        """
        Compare Upstream Head (y_d_HR_lev) to Downstream Head (H_sea_lev)
        """
        if (H_sea_lev/y_d_HR_lev < 0.64): # Supercritical
            """
            Compare Upstream Head (y_d_HR_lev) to Gate Opening (y_sluice_open)
            """
            if (y_d_HR_lev > 1.25*y_sluice_open): # Free sluice flow
                Q_dike_sluice_calc_WHG[i] = Q_free_sluice
            elif (y_d_HR_lev < y_sluice_open): # Supercritical weir flow
                Q_dike_sluice_calc_WHG[i] = Q_supcrit_weir
            else: # Weighted average of supercritical weir and free sluice flow
                Q_dike_sluice_calc_WHG[i] = (Q_free_sluice+Q_supcrit_weir)/2 # This is just the average - how to weight?
        elif (H_sea_lev/y_d_HR_lev > 0.68): # Subcritical
            """
            Compare Upstream Head (y_d_HR_lev) to Gate Opening (y_sluice_open)
            """
            if (y_d_HR_lev > 1.25*y_sluice_open): # Submerged orifice flow
                Q_dike_sluice_calc_WHG[i] = Q_sub_orifice
            elif (y_d_HR_lev < y_sluice_open): # Subcritical weir flow
                Q_dike_sluice_calc_WHG[i] = Q_subcrit_weir
            else: # Weighted average of subcritical weir and submerged orifice flow
                Q_dike_sluice_calc_WHG[i] = (Q_sub_orifice+Q_subcrit_weir)/2 # This is just the average - how to weight?
            
        else: # Weighted average of Supercritical and Subcritical
            """
            Compare Upstream Head (y_d_HR_lev) to Gate Opening (y_sluice_open)
            """
            if (y_d_HR_lev > 1.25*y_sluice_open): # Weighted average of free sluice and submerged orifice flow
                Q_dike_sluice_calc_WHG[i] = (Q_free_sluice+Q_sub_orifice)/2 # This is just the average - how to weight?
            elif (y_d_HR_lev < y_sluice_open): # Weighted average of supercritical weir and subcritical weir flow
                Q_dike_sluice_calc_WHG[i] = (Q_supcrit_weir+Q_subcrit_weir)/2 # This is just the average - how to weight?
            else: # Weighted average of weighted averages of weir and sluice flow.
                Q_dike_sluice_calc_WHG[i] = ((Q_free_sluice+Q_sub_orifice)/2+(Q_supcrit_weir+Q_subcrit_weir)/2)/2
        """
        Flap Gate Conditions
        """
        if (H_sea_lev/(y_d_HR_lev) < 0.64): # Supercritical
            Q_dike_centerflap_calc_WHG[i] = Q_supcrit_weir_flap
        elif (H_sea_lev/(y_d_HR_lev) > 0.68): # Subcritical
            Q_dike_centerflap_calc_WHG[i] = Q_subcrit_weir_flap
        else: # Weighted average
            Q_dike_centerflap_calc_WHG[i] = (Q_supcrit_weir_flap+Q_subcrit_weir_flap)/2
    else: # One of the values is nan, can't calculate.
        Q_dike_sluice_calc_WHG[i] = np.nan
        Q_dike_centerflap_calc_WHG[i] = np.nan 
    
# Left Flap Gate Has Same Conditions as Center (smaller culvert, but same gate size)
Q_dike_leftflap_calc_WHG = Q_dike_centerflap_calc_WHG.copy()

Q_total_WHG = Q_dike_leftflap_calc_WHG + Q_dike_centerflap_calc_WHG + Q_dike_sluice_calc_WHG
    
# Should I be using Manning instead of Energy Eqn to determine Q for open-channel flow through dike?
    
        
#%% Calculating Discharge from Dog Leg to Dike

    
#%% Plot of 2D Side View of HR Dike

import pylab as pl
from matplotlib import collections  as mc

#                    WF Harbor Bath            WF Harbor to Base of Culvert                  Sluice Gate
dike_lines = [[(0,-1.369), (3.048,-1.369)], [(3.048,-1.369), (3.048,-1.064)], [(3.048,-0.579), (3.048,1.095)], 
               [(3.048,-1.064), (23.468,-0.928)], [(3.048,0.463), (23.468,0.600)], [(23.468,-0.928), (25.906,-0.926)]]

# Oceanside levels are H in sluice gate formula
oceanside_level = 1.73 # this is high tide, m
oceanside_level_co = 0.16 # crossover approaching low tide at HR peak level, m
# HR levels are y_d in sluice gate formula. Formula switches from submerged to free flow if y_d drops below base of sluice.
HR_level = -0.10 # this is at high tide, m
HR_level_co = 0.16 # crossover approaching low tide at HR peak level, m
# Sluice height above culvert is y in sluice gate formula
sluice_height = -0.579

WF_opening = 1.984 # height of opening to Wellfleet Harbor, m

dike_levels = [[(0, oceanside_level), (3.048, oceanside_level)], [(23.468, HR_level), (25.906, HR_level)]]
dl_colors = np.array([(0, 1, 0, 1), (0, 0, 1, 1)])
dike_levels_co = [[(0, oceanside_level_co), (3.048, oceanside_level_co)], [(23.468, HR_level_co), (25.906, HR_level_co)]]
dl_colors_co = np.array([(1, 0, 0, 1), (1, 0, 0, 1)])

lc_geom = mc.LineCollection(dike_lines, color='grey', linewidths=2)
lc_levels = mc.LineCollection(dike_levels, color=dl_colors, linewidths=2)
lc_levels_co = mc.LineCollection(dike_levels_co, color=dl_colors_co, linewidths=2)
fig, ax = pl.subplots()
ax.add_collection(lc_geom)
ax.add_collection(lc_levels)
ax.add_collection(lc_levels_co)
ax.autoscale()
ax.margins(0.1)
ax.set_xlim(0,25.906)
ax.set(xlabel='Distance from HR dike face [WF Harbor] to rear [HR] (m)', ylabel='Elevations (m NAVD88)')
ax.grid()

# if y_d < H < 0.81*y_d*(y_d/y)**0.72 then flow is submerged
# if H >= 0.81*y_d*(y_d/y)**0.72 then flow is free



#%% Translating Open-Channel Flow Project (WSP and Q with McCormack/Pr)

range_HRside_avg = 0.766
range_oceanside_avg = 2.535 # note that low tide is not well represented given the river discharge
tide_amp = range_oceanside_avg/2
meanmins_oceanside = -1.02
meanmins_HRside = -0.65

# River mouth depth

def tidal_cycle(time_sec):
        return mean_sea_level + tide_amp*math.sin(math.pi*time_sec/22350)

tide_times = np.arange(0,89700,300)
tide_heights = []
for x in tide_times:
    tide_heights = np.append(tide_heights,tidal_cycle(x))

fig, ax = plt.subplots()
ax.plot(tide_times,tide_heights)
ax.set_xlim(0,89400)
ax.xaxis.set_ticks(np.arange(0, 104300, 14900))
ax.set(xlabel='Time (s)', ylabel='Water Depth Outside Dike (m NAVD88)')
ax.grid()



