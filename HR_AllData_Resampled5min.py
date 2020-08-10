# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:54:55 2020

@author: akurnizk
"""

import csv
import math
import time
import sys,os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib import pylab
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

map_dir = r'E:\Maps' # retrieved files from https://viewer.nationalmap.gov/basic/
data_dir = os.path.join('E:\Data')

#%% Interpolate nans in arrays

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

#%% Loading Information from HR Dike Sensors (Make sure times are in EDT)

with open(os.path.join(data_dir,"General Dike Data","USGS 011058798 Herring R at Chequessett Neck Rd.txt")) as f:
    reader = csv.reader(f, delimiter="\t")
    HR_dike_all_info = list(reader)
    
HR_dike_lev_disch_cond = HR_dike_all_info[32:]
HR_dike_all_df = pd.DataFrame(HR_dike_lev_disch_cond[2:], columns=HR_dike_lev_disch_cond[0])
HR_dike_all_df.drop(HR_dike_all_df.columns[[0,1,3,5,7,9,11,13]],axis=1,inplace=True)
HR_dike_all_df.columns = ["datetime","Gage height, ft, Ocean side","Discharge, cfs","Gage height, ft, HR side",
                          "Spec Con, microsiemens/cm, HR side","Spec Con, microsiemens/cm, Ocean side"]

# Make strings numeric
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

# Remove conductivity columns, convert to metric system
HR_dike_lev_disch_ft = HR_dike_all_df[["datetime","Gage height, ft, Ocean side","Gage height, ft, HR side","Discharge, cfs"]]
HR_dike_lev_disch_m = HR_dike_lev_disch_ft.copy()
HR_dike_lev_disch_m.columns = ["datetime","Gage height, m, Ocean side","Gage height, m, HR side","Discharge, cms"]
HR_dike_lev_disch_m["Gage height, m, Ocean side"] = HR_dike_lev_disch_ft["Gage height, ft, Ocean side"]*0.3048
HR_dike_lev_disch_m["Gage height, m, HR side"] = HR_dike_lev_disch_ft["Gage height, ft, HR side"]*0.3048
HR_dike_lev_disch_m["Discharge, cms"] = HR_dike_lev_disch_ft["Discharge, cfs"]*0.02832

#%% Load HR Geometry and CTD data

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

"""
Loading Information from HR CTD Sensors (Update to USGS filtered CNR U/S data)
"""
with open(os.path.join(data_dir,"General Dike Data","Water_Elevation,_NAVD88-File_Import-01-22-2020_15-04.txt")) as f:
    reader = csv.reader(f, delimiter="\t")
    HR_CTD_all_info = list(reader)

HR_CTD_lev = HR_CTD_all_info[1:]
HR_CTD_all_df = pd.DataFrame(HR_CTD_lev[2:], columns=HR_CTD_lev[0])

# If time needs adjustment
HR_CTD_all_df.drop(HR_CTD_all_df.columns[[0,2,4]],axis=1,inplace=True)
HR_CTD_all_df = HR_CTD_all_df.rename(columns={"Time (MDT to EDT)":"datetime"}) 

# If time is just mislabled
# HR_CTD_all_df.drop(HR_CTD_all_df.columns[[1,2,4]],axis=1,inplace=True)
# HR_CTD_all_df = HR_CTD_all_df.rename(columns={"Time (America/Denver)":"datetime"}) 

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
HR_CTD_all_df["High Toss Water Level, NAVD88"].loc[HR_CTD_all_df["High Toss Water Level, NAVD88"] > 1.00] = np.nan
HR_CTD_all_df["High Toss Water Level, NAVD88"].loc[HR_CTD_all_df["High Toss Water Level, NAVD88"] < -0.67] = np.nan
HR_CTD_all_df["CNR U/S Water Level, NAVD88"].loc[HR_CTD_all_df["CNR U/S Water Level, NAVD88"] < -0.90] = np.nan
HR_CTD_all_df["CNR U/S Water Level, NAVD88"].loc[HR_CTD_all_df["CNR U/S Water Level, NAVD88"] > 0.55] = np.nan
HR_CTD_all_df["Old Saw Water Level, NAVD88"].loc[HR_CTD_all_df["Old Saw Water Level, NAVD88"] < -2.14] = np.nan

#%% CNR U/S Updates from USGS

with open(os.path.join(data_dir,"General Dike Data","CNR_WL_USGS_Fixed.txt")) as f:
    reader = csv.reader(f, delimiter="\t")
    CNRUS_CTD_elevs = list(reader)
    
CNRUS_CTD_df = pd.DataFrame(CNRUS_CTD_elevs[3:], columns=["datetime", "CNR U/S Water Level, NAVD88"])
CNRUS_CTD_df["datetime"] = pd.to_datetime(CNRUS_CTD_df["datetime"])
CNRUS_CTD_df["CNR U/S Water Level, NAVD88"] = pd.to_numeric(CNRUS_CTD_df["CNR U/S Water Level, NAVD88"])

# Plot comparison of old and updates
ax = HR_CTD_all_df.plot.scatter(x="datetime", y="CNR U/S Water Level, NAVD88", color = 'Red', label = 'CNR U/S Levels, Original')
CNRUS_CTD_df.plot.scatter(x="datetime", y="CNR U/S Water Level, NAVD88", color = 'Blue', label = 'CNR U/S Levels, USGS Fix', ax=ax)

# Replace old with new in dataframe
HR_CTD_all_df = pd.merge(HR_CTD_all_df, CNRUS_CTD_df, how="left", left_on="datetime", right_on="datetime")
df_cols = list(HR_CTD_all_df)
HR_CTD_all_df[[df_cols[2], df_cols[5]]] = HR_CTD_all_df[[df_cols[5], df_cols[2]]]
HR_CTD_all_df = HR_CTD_all_df.drop(columns=HR_CTD_all_df[[df_cols[-1]]])
HR_CTD_all_df = HR_CTD_all_df.rename(columns={"CNR U/S Water Level, NAVD88_x":"CNR U/S Water Level, NAVD88"})

#%% Load Calculated Dike Discharge Data

with open(os.path.join(data_dir,"Discharge Data","Q_total_dikecalc.csv")) as f:
    reader = csv.reader(f, delimiter=",")
    Q_total_dikecalc = list(reader)

Q_total_dikecalc_df = pd.DataFrame(Q_total_dikecalc, columns=["datetime", "Discharge, Dike Calc, cms"]) # Calculated
Q_total_dikecalc_df["datetime"] = HR_dike_lev_disch_m["datetime"] # Times are equivalent
Q_total_dikecalc_df["Discharge, Dike Calc, cms"] = pd.to_numeric(Q_total_dikecalc_df["Discharge, Dike Calc, cms"],errors='coerce')

"""
Plots for determining where to take a slice for model comparisons
"""
ax = HR_dike_lev_disch_m.plot.scatter(x="datetime", y="Discharge, cms", color='Turquoise', label='Measured Discharge, cms')
Q_total_dikecalc_df.plot.scatter(x="datetime", y="Discharge, Dike Calc, cms", color='Red', label='Dike Calculated Discharge, cms',ax=ax)
HR_CTD_all_df.plot.scatter(x="datetime", y="High Toss Water Level, NAVD88", color='Green', label='High Toss Levels', ax=ax)
HR_CTD_all_df.plot.scatter(x="datetime", y="CNR U/S Water Level, NAVD88", color='Purple', label='CNR U/S Levels', ax=ax)
plt.legend()

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=22)
plt.ylabel('Elevation (m), Discharge (m^3/s)', fontsize=22)
plt.legend(loc='upper right')

#%% Combining Information from Dike and CTD, Interpolating CTD to multiples of 5 min.

del_fivemin_range = pd.date_range(start=HR_dike_lev_disch_m["datetime"].iloc[0], end=HR_dike_lev_disch_m["datetime"].iloc[-1], freq='5min')
del_fivemin_range = pd.DataFrame(del_fivemin_range,columns=['datetime'])

# HR_dike_lev_disch_m_di = HR_dike_lev_disch_m.set_index('datetime')

HR_dike_CTD_lev_disch_m = pd.merge_ordered(HR_dike_lev_disch_m, HR_CTD_all_df)
HR_dike_CTD_lev_disch_m = pd.merge_ordered(HR_dike_CTD_lev_disch_m, Q_total_dikecalc_df)
HR_dike_CTD_lev_disch_m_di = HR_dike_CTD_lev_disch_m.set_index('datetime')
HR_dike_CTD_lev_disch_m_di.interpolate(method='index', limit=1, inplace=True)

# HR_all_di_resam = HR_dike_CTD_lev_disch_m_di.loc[HR_dike_lev_disch_m_di.index]
HR_all_di_resam = HR_dike_CTD_lev_disch_m_di.reindex(del_fivemin_range["datetime"])
HR_all_resam = HR_all_di_resam.reset_index()

# Merge duplicate datetimes


#%% Save Dataframe
HR_all_resam.to_csv(os.path.join(data_dir,"General Dike Data", "HR_All_Data_Resampled.csv"), index=False, header=True)