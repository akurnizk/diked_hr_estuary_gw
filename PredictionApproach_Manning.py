# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:54:55 2020

@author: akurnizk
"""

import os
import hydroeval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime #parse the datetimes we get from NOAA

from matplotlib import pylab
from scipy.optimize import fsolve

from pytides.tide import Tide

import seaborn as sns; sns.set(font_scale=2)
import matplotlib as mpl
mpl.rc('xtick', labelsize=22)     
mpl.rc('ytick', labelsize=22)
mpl.rcParams['pdf.fonttype'] = 42

map_dir = r'E:\Maps' # retrieved files from https://viewer.nationalmap.gov/basic/
data_dir = os.path.join('E:\Data')

#%% Load Data

"""
Dike + Herring River All
"""
# All Measured + Discharge Calculated off Measured
HR_all_resam_1hr_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","HR_All_Data_Resampled_HourlyMeans_8272017-1212020.csv"))
data_cols = HR_all_resam_1hr_df.columns.drop("datetime")
HR_all_resam_1hr_df[data_cols] = HR_all_resam_1hr_df[data_cols].apply(pd.to_numeric, errors='coerce')
HR_all_resam_1hr_df["datetime"] = pd.to_datetime(HR_all_resam_1hr_df["datetime"])
# WF Harbor, HR Predicted
pred_to_2100_dtHRocean_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","Dike_Data_HourlyPred_111946_12312100.csv"))
data_cols_max = pred_to_2100_dtHRocean_df.columns.drop("datetime")
pred_to_2100_dtHRocean_df[data_cols_max] = pred_to_2100_dtHRocean_df[data_cols_max].apply(pd.to_numeric, errors='coerce')
pred_to_2100_dtHRocean_df["datetime"] = pd.to_datetime(pred_to_2100_dtHRocean_df["datetime"])
# CNR U/S, High Toss Predicted
pred_to_2100_CNRUS_HT_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","CNRUS_HT_HourlyPred_111946_12312100.csv"))
data_cols_min = pred_to_2100_CNRUS_HT_df.columns.drop("datetime")
pred_to_2100_CNRUS_HT_df[data_cols_min] = pred_to_2100_CNRUS_HT_df[data_cols_min].apply(pd.to_numeric, errors='coerce')
pred_to_2100_CNRUS_HT_df["datetime"] = pd.to_datetime(pred_to_2100_CNRUS_HT_df["datetime"])
# Discharge Calculated off Predicted
Q_dike_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","Dike_Discharge_Calc_HourlyPred_111946_12312100.csv"))
data_cols_min = Q_dike_df.columns.drop("datetime")
Q_dike_df[data_cols_min] = Q_dike_df[data_cols_min].apply(pd.to_numeric, errors='coerce')
Q_dike_df["datetime"] = pd.to_datetime(Q_dike_df["datetime"])

"""
Channel Geometry
"""
out_x_stacked = np.loadtxt(os.path.join(map_dir, 'HR_XsecLines','HR_xsec_all_xcoords.csv'), delimiter=',')
out_y_stacked = np.loadtxt(os.path.join(map_dir, 'HR_XsecLines','HR_xsec_all_ycoords.csv'), delimiter=',')
elevs_interp = np.loadtxt(os.path.join(map_dir, 'HR_XsecLines','HR_xsec_all_elevs.csv'), delimiter=',')
intersect_newxy = np.loadtxt(os.path.join(map_dir, 'HR_XsecLines','HR_xsec_all_inscts.csv'), delimiter=',')
min_dist_dx = np.loadtxt(os.path.join(map_dir, 'HR_XsecLines','HR_xsec_all_dx.csv'), delimiter=',')

# make top of array the upstream-most section?
out_x_stacked = np.flip(out_x_stacked,axis=0)
out_y_stacked = np.flip(out_y_stacked,axis=0)
elevs_interp = np.flip(elevs_interp,axis=0)
intersect_newxy = np.flip(intersect_newxy,axis=0)
min_dist_dx = np.flip(min_dist_dx,axis=0)

# High Toss
out_x_HT = out_x_stacked[0]
out_y_HT = out_y_stacked[0]
elev_HT = elevs_interp[0]
intersect_HT = intersect_newxy[0]

# CNR U/S
out_x_CNR = out_x_stacked[-1]
out_y_CNR = out_y_stacked[-1]
elev_CNR = elevs_interp[-1]
intersect_CNR = intersect_newxy[-1]

#%% Plot of Measured, and Dike Q Calcs with Measured

ax = HR_all_resam_1hr_df.plot.scatter(x="datetime", y="Gage height, m, Ocean side", color='LightBlue', label = 'Gage height, m , Ocean side')
HR_all_resam_1hr_df.plot.scatter(x="datetime", y="Gage height, m, HR side", color='LightGreen', label = 'Gage height, m , HR side', ax=ax)
HR_all_resam_1hr_df.plot.scatter(x="datetime", y="Discharge, cms", color='Turquoise', label = 'Discharge, cms', ax=ax)
HR_all_resam_1hr_df.plot.scatter(x="datetime", y="CNR U/S Water Level, NAVD88", color='DarkGreen', label = 'Water Level, m, CNR U/S', ax=ax)
HR_all_resam_1hr_df.plot.scatter(x="datetime", y="Dog Leg Water Level, NAVD88", color='DarkRed', label = 'Water Level, m, Dog Leg', ax=ax)
HR_all_resam_1hr_df.plot.scatter(x="datetime", y="High Toss Water Level, NAVD88", color='DarkOrange', label = 'Water Level, m, High Toss', ax=ax)
HR_all_resam_1hr_df.plot.scatter(x="datetime", y="Discharge, Dike Calc, cms", color='DarkBlue', label = 'Dike Calculated Discharge, cms', ax=ax)

# Show X-axis major tick marks as dates
def DateAxisFmt(yax_label):
    loc = mdates.AutoDateLocator()
    plt.gca().xaxis.set_major_locator(loc)
    plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date', fontsize=22)
    plt.ylabel(yax_label, fontsize=22)

ylabel_elev_disch = 'Elevation (m), Discharge (m^3/s)'
DateAxisFmt(ylabel_elev_disch)
plt.legend(loc='upper right')

#%% Remove NaNs from necessary data.

# For Measured Discharge
HR_disch_measend = HR_all_resam_1hr_df["Discharge, cms"].last_valid_index()

HR_all_meas_disch_df_slice = HR_all_resam_1hr_df.iloc[0:HR_disch_measend]
dt_CNR_HT_disch_cols = ['datetime','CNR U/S Water Level, NAVD88','High Toss Water Level, NAVD88','Discharge, cms']
HR_CNR_HT_disch_df_slice = HR_all_meas_disch_df_slice.filter(dt_CNR_HT_disch_cols, axis=1)
HR_CNR_HT_disch_df_slice.dropna(inplace=True) # Doesn't change anything...

# For Calculated Discharge
HR_disch_calcstrt = HR_all_resam_1hr_df["Discharge, Dike Calc, cms"].first_valid_index()
HR_disch_calcend = HR_all_resam_1hr_df["Discharge, Dike Calc, cms"].last_valid_index()

HR_all_calc_disch_df_slice = HR_all_resam_1hr_df.iloc[HR_disch_calcstrt:HR_disch_calcend]
dt_CNR_HT_calcdisch_cols = ['datetime','CNR U/S Water Level, NAVD88','High Toss Water Level, NAVD88','Discharge, Dike Calc, cms']
HR_CNR_HT_calcdisch_df_slice = HR_all_calc_disch_df_slice.filter(dt_CNR_HT_calcdisch_cols, axis=1)
HR_CNR_HT_calcdisch_df_slice.dropna(inplace=True)
HR_CNR_HT_calcdisch_df_slice.reset_index(drop=True, inplace=True)

# Make sure to re-merge with full time series!

#%% Starting conditions

grav = 9.81 # m/s^2
nsec = len(min_dist_dx) # number of spaces between xsecs
np11 = 2*nsec + 2
tlast = 89400*1 # time for transient flow computation (measurements are at 5 min (300s) intervals - use these?)
chl = np.nansum(min_dist_dx) # channel length
s = 2 # channel lateral slope (assumed for now)

# minimum elevations for each xsec
min_elevs = np.amin(elevs_interp,axis=1)

xsec_loc = np.cumsum(min_dist_dx)
xsec_loc = np.append([0],xsec_loc)

"""
Should I find a linear function that fits the elevations for an average slope?
"""
idx_elevs = np.isfinite(xsec_loc) & np.isfinite(min_elevs)
z_elevs = np.polyfit(xsec_loc[idx_elevs], min_elevs[idx_elevs], 1)
p_elevs = np.poly1d(z_elevs)
polyX_elevs = np.linspace(xsec_loc.min(), xsec_loc.max(), 100)
pylab.plot(polyX_elevs,p_elevs(polyX_elevs),"red", label='Average Slope of HR, HT to Dike')
plt.legend()

s0 = -z_elevs[0] # Reduce flows before changing slope.

Z = z_elevs[0]*xsec_loc + z_elevs[1]

dx = min_dist_dx

# Plot Channel Distance and Bottom Elev

plt.plot(xsec_loc, min_elevs)
plt.ylabel('Herring River Bed Elevation [m NAVD88]')
plt.xlabel('Herring River Distance Downstream from High Toss [m]')

#%% Necessary functions, defined at end of Fortran code
"""
D is depth in deepest part of xsec in each.
"""

def AR(D, elev, out_x, out_y): 
    """
    Satement function for flow area.
    """
    wsp = D + np.nanmin(elev) # add depth to lowest point in channel
    wsp_mask = (elev <= wsp)
    area_xsec = []
    for xsec_i in range(len(elev)-1):
        pt_to_pt_x = np.sqrt((out_x[xsec_i]-out_x[xsec_i+1])**2+(out_y[xsec_i]-out_y[xsec_i+1])**2)
        pt_to_pt_y = abs(elev[xsec_i]-elev[xsec_i+1])
        if (wsp_mask[xsec_i] != wsp_mask[xsec_i+1]):
            pt_to_pt_y = wsp-min(elev[xsec_i],elev[xsec_i+1])
            pt_to_pt_x = pt_to_pt_x*pt_to_pt_y/abs(elev[xsec_i]-elev[xsec_i+1])
            wsp_mask[xsec_i] = True
        area_up = max(min(wsp-elev[xsec_i],wsp-elev[xsec_i+1])*pt_to_pt_x,0) # sets the area_up (rect) to 0 on the edges.
        area_triang = pt_to_pt_y*pt_to_pt_x/2
        area_xsec.append(((area_up+area_triang)*wsp_mask[xsec_i]))
    return np.nansum(area_xsec)

def HR(D, elev=None, out_x=None, out_y=None): # area/w_perim
    """
    Satement function for hydraulic radius.
    """
    wsp = D + np.nanmin(elev) # add depth to lowest point in channel
    wsp_mask = (elev <= wsp)
    w_perim_xsec = []
    for xsec_i in range(len(elev.T)-1):
        pt_to_pt_x = np.sqrt((out_x[xsec_i]-out_x[xsec_i+1])**2+(out_y[xsec_i]-out_y[xsec_i+1])**2)
        pt_to_pt_y = abs(elev[xsec_i]-elev[xsec_i+1])
        pt_to_pt_hyp = np.sqrt(pt_to_pt_x**2 + pt_to_pt_y**2)
        if (wsp_mask[xsec_i] != wsp_mask[xsec_i+1]):
            y_edge = wsp-min(elev[xsec_i],elev[xsec_i+1])
            pt_to_pt_hyp = pt_to_pt_hyp*y_edge/pt_to_pt_y
            wsp_mask[xsec_i] = True
        w_perim_xsec.append((pt_to_pt_hyp*wsp_mask[xsec_i]))
    return AR(D,elev,out_x,out_y)/np.array(np.nansum(w_perim_xsec))

def CHLSLOPE(YUP, YDN, elev_HT=None, elev_CNR=None, chl=None):
    """
    Statement function for top slope of river
    """
    return abs((YUP+elev_HT)-(YDN+elev_CNR))/chl

def MANNING_CMN(cmn, *data):
    """
    Statement function for Manning Equation
    """
    area_ = AR(y_,elev_,out_x_,out_y_)
    hydrad_ = HR(y_,elev_,out_x_,out_y_)
    slope_CNRUStoHT = CHLSLOPE(y_HT, y_CNR, np.nanmin(elev_HT), np.nanmin(elev_CNR), chl)
    return (1/cmn)*area_*hydrad_**(2/3)*np.sqrt(slope_CNRUStoHT) - discharge

def MANNING_HT(y_HT):
    """
    Statement function for Manning Equation
    """
    area_CNRUS = AR(y_CNR,elev_CNR,out_x_CNR,out_y_CNR)
    hydrad_CNRUS = HR(y_CNR,elev_CNR,out_x_CNR,out_y_CNR)
    slope_CNRUStoHT = CHLSLOPE(y_HT, y_CNR, np.nanmin(elev_HT), np.nanmin(elev_CNR), chl)
    return (1/cmn)*area_CNRUS*hydrad_CNRUS**(2/3)*np.sqrt(slope_CNRUStoHT) - discharge
    
#%% Iterate

"""
Using Measured Data: CNR U/S, High Toss, and Dike Discharge
Using CNR and High Toss because Dog Leg has an unusual jump in the data.
"""
cmn = 0.1 # manning's coefficient, initial guess, to be optimized

# Manning Coefficient at CNR U/S
cmn_CNR_arr = []
for row in range(len(HR_CNR_HT_disch_df_slice)):
    discharge = abs(HR_CNR_HT_disch_df_slice["Discharge, cms"].iloc[row])
    starting_guess = cmn
    y_CNR = HR_CNR_HT_disch_df_slice["CNR U/S Water Level, NAVD88"].iloc[row]-np.nanmin(elev_CNR) # depth at location
    y_HT = HR_CNR_HT_disch_df_slice["High Toss Water Level, NAVD88"].iloc[row]-np.nanmin(elev_HT)
    y_, elev_, out_x_, out_y_ = y_CNR, elev_CNR, out_x_CNR, out_y_CNR
    data = y_, elev_, out_x_, out_y_
    temp_HT, = fsolve(MANNING_CMN, starting_guess, args=data)
    cmn_CNR_arr.append(temp_HT)
cmn_CNR_arr = np.array(cmn_CNR_arr)

# Shouldn't be greater than 0.1
cmn_CNR_arr[cmn_CNR_arr>=0.1] = np.nan
plt.plot(HR_CNR_HT_disch_df_slice["datetime"],cmn_CNR_arr)
sns.distplot(cmn_CNR_arr)

# Manning Coefficient at High Toss
cmn_HT_arr = []
for row in range(len(HR_CNR_HT_disch_df_slice)):
    discharge = abs(HR_CNR_HT_disch_df_slice["Discharge, cms"].iloc[row])
    starting_guess = cmn
    y_HT = HR_CNR_HT_disch_df_slice["High Toss Water Level, NAVD88"].iloc[row]-np.nanmin(elev_HT) # depth at location
    y_CNR = HR_CNR_HT_disch_df_slice["CNR U/S Water Level, NAVD88"].iloc[row]-np.nanmin(elev_CNR)
    y_, elev_, out_x_, out_y_ = y_HT, elev_HT, out_x_HT, out_y_HT
    data = y_, elev_, out_x_, out_y_
    temp_HT, = fsolve(MANNING_CMN, starting_guess, args=data)
    cmn_HT_arr.append(temp_HT)
cmn_HT_arr = np.array(cmn_HT_arr)

# Shouldn't be greater than 0.1
cmn_HT_arr[cmn_HT_arr>=0.1] = np.nan
plt.plot(HR_CNR_HT_disch_df_slice["datetime"],cmn_HT_arr)
sns.distplot(cmn_HT_arr)





"""

"""
# Set datetime index, drop nans, save index.

HR_all_30minslice_reindex = HR_all_resam30min_df_slice.reset_index()
HR_all_30minslice_reindex.drop(columns="index",inplace=True)
test_slice = HR_all_30minslice_reindex.iloc[0:HR_all_30minslice_reindex["Discharge, cms"].last_valid_index()]

y_HT_arr = []
for row in range(len(test_slice)):
    discharge = abs(test_slice["Discharge, cms"].iloc[row])
    starting_guess = test_slice["High Toss Water Level, NAVD88"].iloc[row]-np.nanmin(elev_HT)
    y_CNR = test_slice["High Toss Water Level, NAVD88"].iloc[row]-np.nanmin(elev_CNR)
    if ~np.isnan(discharge+starting_guess+y_CNR):
        temp_HT, = fsolve(MANNING, starting_guess)
        y_HT_arr.append(temp_HT)
    else:
        y_HT_arr.append(np.nan)
y_HT_arr = np.array(y_HT_arr)+np.nanmin(elev_HT)

plt.plot(test_slice["datetime"],y_HT_arr)
plt.plot(test_slice["datetime"],test_slice["High Toss Water Level, NAVD88"])









