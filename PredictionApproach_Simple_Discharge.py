# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:54:55 2020

@author: akurnizk
"""

import os
import csv
import dateutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime #parse the datetimes we get from NOAA
from datetime import timedelta

from matplotlib import pylab

from functools import reduce
from scipy.optimize import curve_fit

import uncertainties.unumpy as unp
import uncertainties as unc

from hydroeval import *
import scipy
from scipy import integrate
from scipy.interpolate import interp1d
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
Boston: Need to find literature on trends at this gage.
"""
Boston_8443970_dir = os.path.join(data_dir, 'Surface Water Level Data', 'NOAA 8443970 Boston MA')
df_elevs_Boston = pd.read_csv(os.path.join(Boston_8443970_dir, 'CO-OPS_Boston_8443970_wl_hourly_111946_12312019.csv')) # Calculated
df_elevs_Boston['Verified NAVD Water Levels'] = pd.to_numeric(df_elevs_Boston['Verified NAVD Water Levels'], errors='coerce')
df_elevs_Boston["datetime"] = pd.to_datetime(df_elevs_Boston["datetime"], infer_datetime_format=True)
df_elevs_Boston_di = df_elevs_Boston.set_index('datetime')
df_elevs_Boston = df_elevs_Boston_di.shift(periods=-4) # shift GMT to EST
df_elevs_Boston.reset_index(inplace=True)
dates_Boston_hourly = (mdates.date2num(df_elevs_Boston['datetime'])-mdates.date2num(df_elevs_Boston['datetime'].iloc[0]))*24

# Plot Data and Trend with Hours on x-axis
plt.figure()
plt.scatter(dates_Boston_hourly,df_elevs_Boston['Verified NAVD Water Levels'],marker='.')
idx_Boston_hourly = np.isfinite(dates_Boston_hourly) & np.isfinite(df_elevs_Boston['Verified NAVD Water Levels'])
z_Boston_hourly = np.polyfit(dates_Boston_hourly[idx_Boston_hourly], df_elevs_Boston['Verified NAVD Water Levels'].loc[idx_Boston_hourly], 2)
p_Boston_hourly = np.poly1d(z_Boston_hourly)
polyX_Boston_hourly = np.linspace(dates_Boston_hourly.min(), dates_Boston_hourly.max(), 100)
pylab.plot(polyX_Boston_hourly,p_Boston_hourly(polyX_Boston_hourly),"Red", label='Boston Trend')
plt.xlabel('Hours Since Jan. 1, 1946', fontsize=22)
plt.ylabel('Water Level [m NAVD88]', fontsize=22)
plt.legend(loc='best', fontsize=22)

# Polynomial equation
print("c is acceleration in m/hr^2, b is linear rate of SLR in m/hr, and y0 is the mean sea level in 1946")
print("y=(c/2)t^2+(bt)+(y0)")
print("y=(%.6e/2)t^2+(%.6et)+(%.6f)"%(z_Boston_hourly[0]*2,z_Boston_hourly[1],z_Boston_hourly[2]))
print("Acceleration is %.6f mm/yr^2, Linear rate of SLR is %.6f mm/yr, \
      \n and %.6f m is the mean sea level on Jan 1, 2018"%(z_Boston_hourly[0]*2*1000*8760**2,abs(z_Boston_hourly[1])*1000*8760,z_Boston_hourly[2]))

# Plot Data and Trend with Datetime on x-axis
plt.figure()
plt.scatter(df_elevs_Boston['datetime'],df_elevs_Boston['Verified NAVD Water Levels'],marker='.',label='Boston Tide Gage Hourly Levels')
dates_Boston = mdates.date2num(df_elevs_Boston['datetime'])
idx_Boston = np.isfinite(dates_Boston) & np.isfinite(df_elevs_Boston['Verified NAVD Water Levels'])
n = 2 # degree of polynomial
z_Boston, C_p = np.polyfit(dates_Boston[idx_Boston], df_elevs_Boston['Verified NAVD Water Levels'].loc[idx_Boston], n, cov=True)
p_Boston = np.poly1d(z_Boston)
polyX_Boston = np.linspace(dates_Boston.min(), dates_Boston.max(), 100) # interp for plotting
pylab.plot(polyX_Boston,p_Boston(polyX_Boston),"Red", label='Boston Tide Gage Trend with $\pm1\sigma$-interval')

# Matrix with rows 1, polyX_Boston, polyX_Boston**2:
TT = np.vstack([polyX_Boston**(n-i) for i in range(n+1)]).T
yi = np.dot(TT, z_Boston) # matrix multiplication calculates the polynomial values
C_yi = np.dot(TT, np.dot(C_p, TT.T)) # C_y = TT*C_z*TT.T
sig_yi = np.sqrt(np.diag(C_yi)) # Standard deviations are sqrt of diagonal

plt.fill_between(polyX_Boston, yi+sig_yi, yi-sig_yi, alpha=0.5)

quadtrend_Boston = p_Boston(dates_Boston)

# Show X-axis major tick marks as dates
def DateAxisFmt(yax_label):
    loc = mdates.AutoDateLocator()
    plt.gca().xaxis.set_major_locator(loc)
    plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date', fontsize=22)
    plt.ylabel(yax_label, fontsize=22)

ylabel_wtrlvl = 'Water Level [m NAVD88]'
DateAxisFmt(ylabel_wtrlvl)
plt.legend(loc='best', bbox_to_anchor=(0.6,0.9), fontsize=22, markerscale=4)
plt.xlabel('Year', fontsize=22)

# Boston Rolling Average
df_elevs_Boston["Verified NAVD Water Levels, Annual Rolling Average"] = df_elevs_Boston["Verified NAVD Water Levels"].rolling(window=8766, min_periods=8000).mean() # 709 hrs/Julian month 
plt.figure()
plt.scatter(df_elevs_Boston['datetime'],df_elevs_Boston['Verified NAVD Water Levels, Annual Rolling Average'],marker='.',label='Boston Tide Gage Annual Rolling Average')
# Boston Mean Data
Boston_YearlyMeans = df_elevs_Boston.groupby(df_elevs_Boston["datetime"].dt.year).mean().reset_index()

"""
Dike + Herring River All
"""
# All
HR_all_resam_1hr_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","HR_All_Data_Resampled_HourlyMeans_8272017-1212020.csv")) # Calculated
data_cols = HR_all_resam_1hr_df.columns.drop("datetime")
HR_all_resam_1hr_df[data_cols] = HR_all_resam_1hr_df[data_cols].apply(pd.to_numeric, errors='coerce')
HR_all_resam_1hr_df["datetime"] = pd.to_datetime(HR_all_resam_1hr_df["datetime"])
# Max
HR_all_resam_1hr_maxes_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","HR_All_Data_Resampled_HourlyMaxes_8272017-1212020.csv")) # Calculated
data_cols_max = HR_all_resam_1hr_maxes_df.columns.drop("datetime")
HR_all_resam_1hr_maxes_df[data_cols_max] = HR_all_resam_1hr_maxes_df[data_cols_max].apply(pd.to_numeric, errors='coerce')
HR_all_resam_1hr_maxes_df["datetime"] = pd.to_datetime(HR_all_resam_1hr_maxes_df["datetime"])
# Min
HR_all_resam_1hr_mins_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","HR_All_Data_Resampled_HourlyMins_8272017-1212020.csv")) # Calculated
data_cols_min = HR_all_resam_1hr_mins_df.columns.drop("datetime")
HR_all_resam_1hr_mins_df[data_cols_min] = HR_all_resam_1hr_mins_df[data_cols_min].apply(pd.to_numeric, errors='coerce')
HR_all_resam_1hr_mins_df["datetime"] = pd.to_datetime(HR_all_resam_1hr_mins_df["datetime"])

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

#%% Compare Boston and Dike (Old Saw elevations higher than expected - discarded)

# Plots
ax = HR_all_resam_1hr_df.plot.scatter(x="datetime", y="Gage height, m, Ocean side", color='LightBlue', label = 'WF Harbor Side of Dike')
HR_all_resam_1hr_df.plot.scatter(x="datetime", y="Old Saw Water Level, NAVD88", color='LightGreen', label = 'Old Saw', ax=ax)
df_elevs_Boston.plot.scatter(x="datetime", y="Verified NAVD Water Levels", color='Turquoise', label = 'Boston', ax=ax)

ylabel_wtrlvl = 'Water Level [m NAVD88]'
DateAxisFmt(ylabel_wtrlvl)
plt.legend(loc='best', fontsize=22)

HR_all_resam_1hr_maxes_df_di = HR_all_resam_1hr_maxes_df.set_index('datetime')
oceanside_maxes_df_di = HR_all_resam_1hr_maxes_df_di[['Max Gage height, m, Ocean side']]
df_di_elevs_Boston = df_elevs_Boston.set_index('datetime')
Boston_1hr_df_di = df_di_elevs_Boston[['Verified NAVD Water Levels']]
Boston_oceanside_df = pd.merge(oceanside_maxes_df_di, Boston_1hr_df_di, left_index=True, right_index=True)

Boston_oceanside_df['Boston_OldSaw Diff'] = Boston_oceanside_df['Verified NAVD Water Levels'] - Boston_oceanside_df['Max Gage height, m, Ocean side']

Boston_oceanside_meandiff = np.nanmean(Boston_oceanside_df['Boston_OldSaw Diff'])
Boston_oceanside_stddiff = np.nanstd(Boston_oceanside_df['Boston_OldSaw Diff'])

#%% Plot of everything

ax = HR_all_resam_1hr_df.plot.scatter(x="datetime", y="Gage height, m, Ocean side", color='LightBlue', label = 'Gage height, m , Ocean side')
HR_all_resam_1hr_df.plot.scatter(x="datetime", y="Gage height, m, HR side", color='LightGreen', label = 'Gage height, m , HR side', ax=ax)
HR_all_resam_1hr_df.plot.scatter(x="datetime", y="Discharge, cms", color='Turquoise', label = 'Discharge, cms', ax=ax)
HR_all_resam_1hr_df.plot.scatter(x="datetime", y="CNR U/S Water Level, NAVD88", color='DarkGreen', label = 'Water Level, m, CNR U/S', ax=ax)
HR_all_resam_1hr_df.plot.scatter(x="datetime", y="Dog Leg Water Level, NAVD88", color='DarkRed', label = 'Water Level, m, Dog Leg', ax=ax)
HR_all_resam_1hr_df.plot.scatter(x="datetime", y="High Toss Water Level, NAVD88", color='DarkOrange', label = 'Water Level, m, High Toss', ax=ax)
HR_all_resam_1hr_df.plot.scatter(x="datetime", y="Discharge, Dike Calc, cms", color='DarkBlue', label = 'Dike Calculated Discharge, cms', ax=ax)

ylabel_elev_disch = 'Elevation (m), Discharge (m^3/s)'
DateAxisFmt(ylabel_elev_disch)
plt.legend(loc='upper right')

#%% Load Dike Discharges
"""
Discharge through dike
Measurements are taken every 5 minutes
Filtering takes a 1 hour average
"""
with open(os.path.join(data_dir,"General Dike Data","USGS 011058798 Discharge 2015-2017.txt")) as f:
    reader = csv.reader(f, delimiter="\t")
    HR_dike_all_discharge = list(reader)

HR_dike_discharge = []
HR_dike_discharge_filtered = []
for line in range(len(HR_dike_all_discharge)-30):
    HR_dike_discharge.append([HR_dike_all_discharge[line+30][2],HR_dike_all_discharge[line+30][4]])
    HR_dike_discharge_filtered.append([HR_dike_all_discharge[line+30][2],HR_dike_all_discharge[line+30][6]])

HR_dike_discharge = np.array(HR_dike_discharge)
x_discharge, y_discharge = HR_dike_discharge.T
dates_discharge = [dateutil.parser.parse(x) for x in x_discharge]
x_discharge_datenum = mdates.date2num(dates_discharge)
y_discharge[np.where(y_discharge == '')] = np.nan
y_discharge = y_discharge.astype(np.float)*0.028316847 # cfs to cms

Q_dike_meas_list = np.vstack((dates_discharge, y_discharge)).T
Q_dike_meas_df = pd.DataFrame(Q_dike_meas_list, columns=['datetime', 'Discharge, cms'])
Q_dike_meas_df['Discharge, cms'] = pd.to_numeric(Q_dike_meas_df['Discharge, cms'])

Q_dike_meas_df_di = Q_dike_meas_df.set_index('datetime') # set date index

# Merging Duplicate Entries
Q_dike_meas_df_di = Q_dike_meas_df_di.mean(level=0)
Q_dike_meas_df = Q_dike_meas_df_di.reset_index()

#%% Interpolating to multiples of 5 min.

del_fivemin_range = pd.date_range(start=Q_dike_meas_df["datetime"].iloc[0], end=Q_dike_meas_df["datetime"].iloc[-1], freq='5min')
del_fivemin_range = pd.DataFrame(del_fivemin_range,columns=['datetime'])

HR_dike_disch_m = pd.merge_ordered(del_fivemin_range, Q_dike_meas_df)
HR_dike_disch_m_di = HR_dike_disch_m.set_index('datetime')
HR_dike_disch_m_di.interpolate(method='index', limit=1, inplace=True)

HR_dike_Qmeas_di_resam = HR_dike_disch_m_di.reindex(del_fivemin_range["datetime"])
HR_dike_Qmeas_resam = HR_dike_Qmeas_di_resam.reset_index()

# Plotting 5 minute sampling and original
ax = Q_dike_meas_df.plot.scatter('datetime', 'Discharge, cms', marker='+')
HR_dike_Qmeas_resam.plot.scatter('datetime', 'Discharge, cms', color='Green', marker='.', ax=ax)

#%% Save Dataframe
# HR_dike_Qmeas_resam.to_csv(os.path.join(data_dir,"General Dike Data", "HR_Meas_Discharge_Data_Resampled.csv"), index=False, header=True)

#%% Load Dataframe
HR_dike_Qmeas_resam = pd.read_csv(os.path.join(data_dir, 'General Dike Data', 'HR_Meas_Discharge_Data_Resampled.csv'))

#%% Get hourly averages

hourlyQ_summary_onhr = [] # faster to append to list then make frame
bin_old=HR_dike_Qmeas_di_resam.index[0]
for row in range(len(HR_dike_Qmeas_di_resam)):
    if (HR_dike_Qmeas_di_resam.index.minute[row]==30):
        bin_new = HR_dike_Qmeas_di_resam.index[row]
        hourlyQ_summary_onhr.append(HR_dike_Qmeas_di_resam[bin_old:bin_new].mean())
        bin_old = HR_dike_Qmeas_di_resam.index[row]
                               
hourlyQ_summary_onhr_df = pd.DataFrame(hourlyQ_summary_onhr)
hourlyQ_summary_onhr_df = hourlyQ_summary_onhr_df[1:]

# this line is just for the datetime indexes
hourlyQ_summary_onhr_di = HR_dike_Qmeas_di_resam.resample('H').mean()[1:] # first isn't a representative mean.

hourlyQ_summary_onhr_df.index = hourlyQ_summary_onhr_di.index
hourlyQ_summary_onhr_df.reset_index(inplace=True)

#%% Save Dataframe
# hourlyQ_summary_onhr_df.to_csv(os.path.join(data_dir,"General Dike Data","Dike_Q_Data_Resampled_HourlyMeans_6242015-1012017.csv"), index = False)

#%% Read Dataframe
hourlyQ_summary_onhr_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","Dike_Q_Data_Resampled_HourlyMeans_6242015-1012017.csv"))
hourlyQ_summary_onhr_df["datetime"] = pd.to_datetime(hourlyQ_summary_onhr_df["datetime"], infer_datetime_format=True)

# Plot hourly discharge data
hourlyQ_summary_onhr_df.plot.scatter("datetime","Discharge, cms")

#%% Integration - total volume passed each cycle

def Q_Integrated(dates, discharge):
    """
    Function for extracting the volume flux of a time series
    of discharge through a dike with inflow and outflow
    components. 

    Parameters
    ----------
    dates : Series
        Series object of pandas.core.series module
    discharge : Series
        Series object of pandas.core.series module

    Returns
    -------
    V_in_dates : , Array of object
        ndarray object of numpy module
    V_in : , Array of float64
        same shape as V_in_dates
    V_out_dates : , Array of object
        ndarray object of numpy module
    V_out : , Array of float64
        same shape as V_out_dates

    """
    datestart_neg = 0
    datestart_pos = 0
    date_interval_neg = 0
    date_interval_pos = 0
    bin_start_neg = 0
    bin_start_pos = 0
    V_in_dates = []
    V_out_dates = []
    V_in = []
    V_out = []
    for bin_index in range(len(dates)-1):
        disch_start = discharge[bin_index]
        disch_end = discharge[bin_index+1]
        if (disch_start*disch_end<0)&(disch_start<disch_end): # Transfer from neg to pos Q = start of pos Q/end neg Q (calc neg)
            datestart_pos = dates.iloc[bin_index+1]
            m_intsct_posstrt_negend = (disch_start-disch_end)/3600 # slope of line that intersects x-axis
            intsct_pt_posstrt_negend = (disch_start/m_intsct_posstrt_negend) # linear approximation of time where discharge is zero
            bin_start_pos = bin_index+1
            dateend_neg = dates.iloc[bin_index]
            if (datestart_neg!=0):
                date_interval_neg = (dateend_neg - datestart_neg).seconds # date interval in seconds
                if (date_interval_neg > 6000): # Make sure small fluctuations aren't being counted
                    disch_strt_end = np.array([0.0,0.0])
                    temp_interval_inner = discharge.iloc[bin_start_neg:(bin_index+1)]
                    temp_interval = np.insert(disch_strt_end,1,np.array(temp_interval_inner))
                    time_strt_end = np.array([intsct_pt_negstrt_posend-3600,(intsct_pt_posstrt_negend+date_interval_neg)])
                    temp_seconds_interval = np.insert(time_strt_end,1,np.arange(0,(date_interval_neg+3600),3600))
                    min_index = temp_interval_inner.loc[temp_interval_inner==np.nanmin(temp_interval_inner)].index.values[0] # for assigning date
                    if (len(V_in_dates) == 0)&(~np.isnan(temp_interval.mean())):
                        V_in.append(integrate.trapz(temp_interval,temp_seconds_interval))
                        V_in_dates.append(dates.iloc[min_index])
                    if (dates.iloc[min_index] != V_in_dates[-1])&(~np.isnan(temp_interval.mean())): # makes sure duplicates aren't being printed
                        V_in.append(integrate.trapz(temp_interval,temp_seconds_interval)) # duplicates are somehow the result of nans
                        V_in_dates.append(dates.iloc[min_index])
        if (disch_start*disch_end<0)&(disch_start>disch_end): # start of neg Q/end pos Q (don't use else, as it would capture nans)
            datestart_neg = dates.iloc[bin_index+1]
            m_intsct_negstrt_posend = (disch_start-disch_end)/3600
            intsct_pt_negstrt_posend = (disch_start/m_intsct_negstrt_posend)
            bin_start_neg = bin_index+1
            dateend_pos = dates.iloc[bin_index]
            if (datestart_pos!=0):
                date_interval_pos = (dateend_pos - datestart_pos).seconds # date interval in seconds
                if (date_interval_pos > 6000): # Make sure small fluctuations aren't being counted
                    disch_strt_end = np.array([0.0,0.0])
                    temp_interval_inner = discharge.iloc[bin_start_pos:(bin_index+1)]
                    temp_interval = np.insert(disch_strt_end,1,np.array(temp_interval_inner))
                    time_strt_end = np.array([intsct_pt_posstrt_negend-3600,(intsct_pt_negstrt_posend+date_interval_pos)])
                    temp_seconds_interval = np.insert(time_strt_end,1,np.arange(0,(date_interval_pos+3600),3600))
                    max_index = temp_interval_inner.loc[temp_interval_inner==np.nanmax(temp_interval_inner)].index.values[0] # for assigning date 
                    if (len(V_out_dates) == 0)&(~np.isnan(temp_interval.mean())):
                        V_out.append(integrate.trapz(temp_interval,temp_seconds_interval))
                        V_out_dates.append(dates.iloc[max_index])
                    if (dates.iloc[max_index] != V_out_dates[-1])&(~np.isnan(temp_interval.mean())): # makes sure duplicates aren't being printed
                        V_out.append(integrate.trapz(temp_interval,temp_seconds_interval))
                        V_out_dates.append(dates.iloc[max_index]) # duplicates are somehow the result of nans
    V_in_dates = np.array(V_in_dates)
    V_out_dates = np.array(V_out_dates)
    V_in = np.array(V_in)
    V_out = np.array(V_out)
    return V_in_dates, V_in, V_out_dates, V_out

#%% Integration and Analysis of Measured Discharge Data
# Q_dike_meas_int_df = hourlyQ_summary_onhr_di.rolling('149H').apply(integrate.trapz)
# Q_dike_meas_int_df = integrate.cumtrapz(hourlyQ_summary_onhr_df["Discharge, cms"], np.arange(0,len(hourlyQ_summary_onhr_df)))

dike_discharge_1hr_meas = hourlyQ_summary_onhr_df["Discharge, cms"]
dates_disch_1hr_meas = hourlyQ_summary_onhr_df["datetime"]
V_dike_in_dates, V_dike_in, V_dike_out_dates, V_dike_out = Q_Integrated(dates_disch_1hr_meas, dike_discharge_1hr_meas)

# Combine Discharge and Associated Dates
dike_outflow_vol_arr = np.vstack([V_dike_out_dates,V_dike_out]).T
dike_outflow_vol_df = pd.DataFrame(dike_outflow_vol_arr, columns=["datetime", "Volume Through Dike, m^3, Out"])
dike_outflow_vol_df["datetime"] = pd.to_datetime(dike_outflow_vol_df["datetime"])
dike_outflow_vol_df["Volume Through Dike, m^3, Out"] = pd.to_numeric(dike_outflow_vol_df["Volume Through Dike, m^3, Out"])

dike_inflow_vol_arr = np.vstack([V_dike_in_dates,V_dike_in]).T
dike_inflow_vol_df = pd.DataFrame(dike_inflow_vol_arr, columns=["datetime", "Volume Through Dike, m^3, In"])
dike_inflow_vol_df["datetime"] = pd.to_datetime(dike_inflow_vol_df["datetime"])
dike_inflow_vol_df["Volume Through Dike, m^3, In"] = pd.to_numeric(dike_inflow_vol_df["Volume Through Dike, m^3, In"])

total_V_in = np.nansum(V_dike_in)
total_V_out = np.nansum(V_dike_out)
net_V_discharged = total_V_in+total_V_out
total_time = (dike_outflow_vol_df["datetime"].iloc[-1]-dike_outflow_vol_df["datetime"].iloc[0]).total_seconds()
fw_discharge = net_V_discharged/total_time # only 0.04 m^3/s -> very small compared to Masterson's estimate.

ax = dike_outflow_vol_df.plot.scatter("datetime", "Volume Through Dike, m^3, Out")
dike_inflow_vol_df.plot.scatter("datetime", "Volume Through Dike, m^3, In", color='Green', ax=ax)

#%% Function for determining mins reached at a sensor resulting (res) its maxes

def MaxToMin(min_lev, max_lev, datetime_maxes):
    """
    Function for determing minimum levels reached at a gage proceeding
    the maximum levels at that gage in each tidal cycle.

    Parameters
    ----------
    min_lev : , Series
        Series object of pandas.core.series module, single column from 
        datetime ordered dataframe of gage min levels
    max_lev : , Series
        Series object of pandas.core.series module, single column from 
        datetime ordered dataframe of gage max levels
    datetime_maxes : , Series
        Series object of pandas.core.series module, single column from 
        datetime ordered dataframe of oceanside and river gage max levels

    Returns
    -------
    deltaT_ranges : Array of object
        ndarray object of numpy module, the elapsed time from maxes
        to mins in each tidal cycle
    y_res_mins : Array of float64
        the minimum levels at a gage inside the dike, the same size as the
        max levels (possibly including nans)

    """
    y_res_mins = []
    deltaT_ranges = []
    for row in range(len(max_lev)-1):
        range_length = (datetime_maxes.iloc[row+1]-datetime_maxes.iloc[row]).seconds/3600 # hours
        if ~np.isnan(max_lev.iloc[row]) & (range_length>=3.5) & (range_length<=7.5): # peak to trough should take 6.2 hours in ocean
            y_res_mins.append(min_lev.iloc[row+1])
            deltaT_ranges.append(datetime_maxes.iloc[row+1]-datetime_maxes.iloc[row])
        elif ~np.isnan(max_lev.iloc[row]): # if peak lag is more than 4 hours, erroneous data
            y_res_mins.append(np.nan)
            deltaT_ranges.append(np.nan)
    # if last value in oceanside array is not nan, append another nan on the return arrays
    if (datetime_maxes.iloc[row+1]==datetime_maxes.iloc[-1]) & ~np.isnan(max_lev.iloc[row+1]): 
        y_res_mins.append(np.nan)
        deltaT_ranges.append(np.nan)
    y_res_mins = np.array(y_res_mins)
    deltaT_ranges = np.array(deltaT_ranges)
    return deltaT_ranges, y_res_mins

#%% Discharge Differences through dike

# Combine outflow and inflow arrays, keeping and ordering dates.

# Harbor Side of Dike
inoutvols_dike_ordered_df = pd.merge_ordered(dike_outflow_vol_df, dike_inflow_vol_df, on="datetime")
dike_outflows = inoutvols_dike_ordered_df["Volume Through Dike, m^3, Out"]
dike_inflows = inoutvols_dike_ordered_df["Volume Through Dike, m^3, In"]
datetime_dike_outin = inoutvols_dike_ordered_df["datetime"]

deltaT_disch_ranges, y_dike_res_inflows = MaxToMin(dike_inflows, dike_outflows, datetime_dike_outin)
deltaT_disch_ranges_inv, y_dike_res_outflows = MaxToMin(dike_outflows, dike_inflows, datetime_dike_outin)

# Net Outflow from Dike
dike_inflow_arr = np.vstack([V_dike_out_dates, y_dike_res_inflows]).T
dike_inflow_df = pd.DataFrame(dike_inflow_arr, columns=["datetime","Proceeding Volume Influx, m^3"])
dike_inflow_df["datetime"] = pd.to_datetime(dike_inflow_df["datetime"])
dike_inflow_df["Proceeding Volume Influx, m^3"] = pd.to_numeric(dike_inflow_df["Proceeding Volume Influx, m^3"])
outflow_dike_inflow_dike_df = pd.merge(dike_outflow_vol_df, dike_inflow_df)
net_outflow = (outflow_dike_inflow_dike_df["Volume Through Dike, m^3, Out"]+outflow_dike_inflow_dike_df["Proceeding Volume Influx, m^3"])
V_dike_outinmean_dates = V_dike_out_dates + timedelta(hours=3) # Centers datetime between volume flux out and in.
net_outflow_arr = np.vstack([V_dike_outinmean_dates, net_outflow]).T
net_outflow_df = pd.DataFrame(net_outflow_arr, columns=["Centered Datetime","Net Dike Ouflow, m^3"])
net_outflow_df["Centered Datetime"] = pd.to_datetime(net_outflow_df["Centered Datetime"])
net_outflow_df["Net Dike Ouflow, m^3"] = pd.to_numeric(net_outflow_df["Net Dike Ouflow, m^3"])

net_outflow_df.plot.scatter("Centered Datetime","Net Dike Ouflow, m^3")

net_outflow_mean = np.nanmean(net_outflow_df["Net Dike Ouflow, m^3"])
net_outflow_std = np.nanstd(net_outflow_df["Net Dike Ouflow, m^3"])

#%% Volume Flux Trendline (eliminates no-data points from consideration)

x_V_flux_datenum = mdates.date2num(net_outflow_df["Centered Datetime"])

x_V_flux_cycle = np.arange(0,1520,1)
# idx_V_flux = np.isfinite(net_outflow)
# z_V_flux = np.polyfit(x_V_flux_cycle[idx_V_flux], net_outflow[idx_V_flux], 1)
# p_V_flux = np.poly1d(z_V_flux)
# polyX_V_flux = np.linspace(x_V_flux_cycle.min(), x_V_flux_cycle.max(), 100)

plt.scatter(x_V_flux_cycle, net_outflow, label="Volume Flux Per Tidal Cycle")
# pylab.plot(polyX_V_flux, p_V_flux(polyX_V_flux),"g", label='Regressed Fit (Linear) of Tidal Flux')
# the line equations:
# print("Equation for Linear Fit, "+("y=%.6fx+(%.6f)"%(z_V_flux[0],z_V_flux[1])))
# Over the time period, HR gained approximately 5 m^3 per cycle

plt.xlabel('Tidal Cycle from 2015-06-24 to 2017-09-30', fontsize=22)
plt.ylabel(r'In  <=  Volume Flux $\left[\frac{m^3}{cycle}\right]$  =>  Out', fontsize=22)
plt.legend(loc='best', fontsize=22)

#%% Using PyTides to get tidal constituents
# Downloaded version to work with Python 3.7 and above, from github user yudevan
# https://github.com/yudevan/pytides/commit/507f2bc5d19fa5e427045cc2bf9ed724daf67f0c
# Would converting all times to UTC make a better fit? How best to compare?

"""
Plot of residuals of each?
"""

"""
Wellfleet Harbor Side of Dike
"""
oceanside_levels = np.array(HR_all_resam_1hr_df["Gage height, m, Ocean side"])
oceanside_levels_df = HR_all_resam_1hr_df.filter(["datetime", "Gage height, m, Ocean side"], axis=1)
oceanside_levels_df.rename(columns={"Gage height, m, Ocean side": "Gage height, m, Ocean side, Measured"}, inplace=True)
# NEED TO NAN LOW LEVELS TO GET BETTER PREDICTION

ax = HR_all_resam_1hr_df.plot.scatter(x="datetime", y="Gage height, m, Ocean side", color='LightBlue', label = 'Gage height, m , Ocean side')
HR_all_resam_1hr_mins_df.plot.scatter("datetime","Min Gage height, m, Ocean side",color="g",ax=ax)

# Try to nan everything below -0.97 m to try to capture low signal (from eyeballing graph)
oceanside_levels_nanmins = oceanside_levels.copy()
oceanside_levels_nanmins[oceanside_levels_nanmins<-0.5] = np.nan
oceanside_nanmins_df = pd.concat([HR_all_resam_1hr_df["datetime"],pd.DataFrame(oceanside_levels_nanmins,columns=["Gage height, m, Ocean side"])], axis=1)

# Need to correlate Boston and WF Harbor Side of Dike to make theoretical mins to append.
oceanside_datestart = HR_all_resam_1hr_df["datetime"].iloc[0]
oceanside_dateend = HR_all_resam_1hr_df["datetime"].iloc[-1]
df_elevs_Boston_di = df_elevs_Boston.set_index("datetime")
df_elevs_roll_Boston_dikedates = df_elevs_Boston_di.loc[oceanside_datestart:oceanside_dateend]
df_elevs_roll_Boston_dikedates.reset_index(inplace=True)
df_elevs_Boston_dikedates = df_elevs_roll_Boston_dikedates.drop(columns=['Verified NAVD Water Levels, Annual Rolling Average'])
elevs_Boston_dikeWF_df = pd.merge(df_elevs_Boston_dikedates, oceanside_nanmins_df)
# Shift Boston data 4 hours back to match
elevs_Boston_dikeWF_df["Verified NAVD Water Levels"] = elevs_Boston_dikeWF_df["Verified NAVD Water Levels"].shift(periods=-4)

# Plot levels together
plt.figure()
plt.plot(elevs_Boston_dikeWF_df["datetime"], elevs_Boston_dikeWF_df["Verified NAVD Water Levels"], label='Boston Tide Gage')
plt.plot(elevs_Boston_dikeWF_df["datetime"], elevs_Boston_dikeWF_df["Gage height, m, Ocean side"], color='Purple', marker='.', label='WF Harbor Near-Dike Tide Gage')
ylabel_elev = 'Elevation [m NAVD88]'
DateAxisFmt(ylabel_elev)
plt.legend(fontsize=22)

# Find mean and std of residuals to see if mean offset is okay to use. # observed=WF, expected=Boston
elevs_Boston_dikeWF_df["Residuals"] = elevs_Boston_dikeWF_df["Gage height, m, Ocean side"] - elevs_Boston_dikeWF_df["Verified NAVD Water Levels"]
elevs_Boston_dikeWF_resid_mean = np.nanmean(elevs_Boston_dikeWF_df["Residuals"])
elevs_Boston_dikeWF_resid_std = np.nanstd(elevs_Boston_dikeWF_df["Residuals"]) # Not good, too much back/forth phase shift

"""
FIX!!!!!!!!!! It's making random low levels for some reason.
"""

# Fill low tide levels by keeping the same horizontal displacement as that between the last valid Boston/WF Harbor values.
oceanside_levels_start_i = elevs_Boston_dikeWF_df["Gage height, m, Ocean side"].first_valid_index()
oceanside_levels_end_i = elevs_Boston_dikeWF_df["Gage height, m, Ocean side"].last_valid_index()
elevs_Boston_dikeWF_df = elevs_Boston_dikeWF_df.iloc[oceanside_levels_start_i:oceanside_levels_end_i]
elevs_Boston_dikeWF_df.reset_index(drop=True, inplace=True)
elevs_Boston_dikeWF_df = pd.merge(elevs_Boston_dikeWF_df, oceanside_levels_df)

for row in range(len(elevs_Boston_dikeWF_df)): # make sure first and last values are not nan!
    if np.isnan(elevs_Boston_dikeWF_df["Gage height, m, Ocean side"].iloc[row]):
        WF_old = elevs_Boston_dikeWF_df["Gage height, m, Ocean side"].iloc[row-1]
        Bost_old = elevs_Boston_dikeWF_df["Verified NAVD Water Levels"].iloc[row-1]
        Bost_now = elevs_Boston_dikeWF_df["Verified NAVD Water Levels"].iloc[row]
        Bost_next = elevs_Boston_dikeWF_df["Verified NAVD Water Levels"].iloc[row+1]
        horiz_disp = (WF_old - Bost_old)/(Bost_now - Bost_old)
        elevs_Boston_dikeWF_df["Gage height, m, Ocean side"].iloc[row] = (Bost_next-Bost_now)*horiz_disp + Bost_now
        actmin_lt_predmin_cond = (elevs_Boston_dikeWF_df["Gage height, m, Ocean side, Measured"].iloc[row] < elevs_Boston_dikeWF_df["Gage height, m, Ocean side"].iloc[row])
        if actmin_lt_predmin_cond:
            elevs_Boston_dikeWF_df["Gage height, m, Ocean side"].iloc[row] = elevs_Boston_dikeWF_df["Gage height, m, Ocean side, Measured"].iloc[row]
        if ~np.isnan(elevs_Boston_dikeWF_df["Gage height, m, Ocean side"].iloc[row-1]):
            max_Bost = np.nanmax(elevs_Boston_dikeWF_df["Verified NAVD Water Levels"].iloc[(row-9):row])
            max_WF = np.nanmax(elevs_Boston_dikeWF_df["Gage height, m, Ocean side"].iloc[(row-9):row])
            max_diff = max_WF - max_Bost
        if (Bost_next > Bost_now)&~actmin_lt_predmin_cond:
            elevs_Boston_dikeWF_df["Gage height, m, Ocean side"].iloc[row] = Bost_now + max_diff # keep ranges about the same

plt.plot(elevs_Boston_dikeWF_df["datetime"], elevs_Boston_dikeWF_df["Gage height, m, Ocean side"], color='Orange', label='WF Harbor Near-Dike Tide Gage, Predicted')
ylabel_elev = 'Elevation [m NAVD88]'
DateAxisFmt(ylabel_elev)
plt.legend(fontsize=22)

oceanside_levels_predlows = np.array(elevs_Boston_dikeWF_df["Gage height, m, Ocean side"])

oceanside_nans = np.isnan(oceanside_levels_predlows)
oceanside_levels_nonans = oceanside_levels_predlows[~oceanside_nans]
oceanside_dates_nonans = elevs_Boston_dikeWF_df['datetime'][~oceanside_nans].reset_index(drop=True)

# Prepare a list of datetimes over oceanside date range
oceanside_t0 = oceanside_dates_nonans.iloc[0].to_pydatetime()
oceanside_tdelta = oceanside_dates_nonans.iloc[-1].to_pydatetime() - oceanside_t0
oceanside_hours = np.arange(oceanside_tdelta.total_seconds()/3600) # 56486 days from May 8, 1945 to Jan 1, 2100 (replace with "7" to represent a week)
hours_to_midpoint = int(oceanside_tdelta.total_seconds()/3600/2)
oceanside_times = Tide._times(oceanside_t0, oceanside_hours)

# Fit the tidal data to the harmonic model using Pytides
oceanside_tide = Tide.decompose(oceanside_levels_nonans.T, oceanside_dates_nonans)

# Predict the tides using the Pytides model for oceanside measurement range
oceanside_prediction = oceanside_tide.at(oceanside_times)
oceanside_prediction_mean = oceanside_prediction.mean() # This is less than a millimeter away from the mean of the data.
oceanside_prediction_df = pd.DataFrame(oceanside_prediction, index=oceanside_times)

# Plot the results: Note that adjustments should be made to handle the river discharge.
plt.figure()
plt.plot(elevs_Boston_dikeWF_df["datetime"], oceanside_levels_predlows, label="WF Harbor Side of Dike, Measured Highs, Predicted Lows")
plt.plot(HR_all_resam_1hr_df['datetime'], oceanside_levels, label="WF Harbor Side of Dike, Measured Data")
plt.plot(oceanside_times, oceanside_prediction, label="Pytides")
ylabel_elev = 'Elevation [m NAVD88]'
DateAxisFmt(ylabel_elev)
plt.legend(fontsize=22)

"""
HR Side of Dike
"""
HRside_levels = np.array(HR_all_resam_1hr_df["Gage height, m, HR side"])
HRside_nans = np.isnan(HRside_levels)
HRside_levels_nonans = HRside_levels[~HRside_nans]
HRside_dates_nonans = HR_all_resam_1hr_df['datetime'][~HRside_nans].reset_index(drop=True)

# Prepare a list of datetimes over HRside date range
HRside_t0 = HRside_dates_nonans.iloc[0].to_pydatetime()
HRside_tdelta = HRside_dates_nonans.iloc[-1].to_pydatetime() - HRside_t0
HRside_hours = np.arange(HRside_tdelta.total_seconds()/3600) # 56486 days from May 8, 1945 to Jan 1, 2100 (replace with "7" to represent a week)
hours_to_midpoint = int(HRside_tdelta.total_seconds()/3600/2)
HRside_times = Tide._times(HRside_t0, HRside_hours)

# Fit the tidal data to the harmonic model using Pytides
HRside_tide = Tide.decompose(HRside_levels_nonans.T, HRside_dates_nonans)

# Predict the tides using the Pytides model for HRside measurement range
HRside_prediction = HRside_tide.at(HRside_times)
HRside_prediction_mean = HRside_prediction.mean() # This is less than a millimeter away from the mean of the data.
HRside_prediction_df = pd.DataFrame(HRside_prediction, index=HRside_times)

# Plot the results: Note that adjustments should be made to handle the river discharge.
plt.figure()
plt.plot(HR_all_resam_1hr_df['datetime'], HRside_levels)
plt.plot(HRside_times, HRside_prediction, label="Pytides")
ylabel_elev = 'Elevation [m NAVD88]'
DateAxisFmt(ylabel_elev)
plt.legend()

"""
CNR U/S Sensor
"""
CNRUS_levels = np.array(HR_all_resam_1hr_df["CNR U/S Water Level, NAVD88"])
CNRUS_nans = np.isnan(CNRUS_levels)
CNRUS_levels_nonans = CNRUS_levels[~CNRUS_nans]
CNRUS_dates_nonans = HR_all_resam_1hr_df['datetime'][~CNRUS_nans].reset_index(drop=True)

# Prepare a list of datetimes over CNRUS date range
CNRUS_t0 = CNRUS_dates_nonans.iloc[0].to_pydatetime()
CNRUS_tdelta = CNRUS_dates_nonans.iloc[-1].to_pydatetime() - CNRUS_t0
CNRUS_hours = np.arange(CNRUS_tdelta.total_seconds()/3600) # 56486 days from May 8, 1945 to Jan 1, 2100 (replace with "7" to represent a week)
hours_to_midpoint = int(CNRUS_tdelta.total_seconds()/3600/2)
CNRUS_times = Tide._times(CNRUS_t0, CNRUS_hours)

# Fit the tidal data to the harmonic model using Pytides
CNRUS_tide = Tide.decompose(CNRUS_levels_nonans.T, CNRUS_dates_nonans)

# Predict the tides using the Pytides model for CNRUS measurement range
CNRUS_prediction = CNRUS_tide.at(CNRUS_times)
CNRUS_prediction_mean = CNRUS_prediction.mean() # This is less than a millimeter away from the mean of the data.
CNRUS_prediction_df = pd.DataFrame(CNRUS_prediction, index=CNRUS_times)

# Plot the results: Note that adjustments should be made to handle the river discharge.
plt.figure()
plt.plot(HR_all_resam_1hr_df['datetime'], CNRUS_levels)
plt.plot(CNRUS_times, CNRUS_prediction, label="Pytides")
ylabel_elev = 'Elevation [m NAVD88]'
DateAxisFmt(ylabel_elev)
plt.legend()

"""
Dog Leg Sensor
"""
DogLeg_levels = np.array(HR_all_resam_1hr_df["Dog Leg Water Level, NAVD88"])
DogLeg_nans = np.isnan(DogLeg_levels)
DogLeg_levels_nonans = DogLeg_levels[~DogLeg_nans]
DogLeg_dates_nonans = HR_all_resam_1hr_df['datetime'][~DogLeg_nans].reset_index(drop=True)

# Prepare a list of datetimes over DogLeg date range
DogLeg_t0 = DogLeg_dates_nonans.iloc[0].to_pydatetime()
DogLeg_tdelta = DogLeg_dates_nonans.iloc[-1].to_pydatetime() - DogLeg_t0
DogLeg_hours = np.arange(DogLeg_tdelta.total_seconds()/3600) # 56486 days from May 8, 1945 to Jan 1, 2100 (replace with "7" to represent a week)
hours_to_midpoint = int(DogLeg_tdelta.total_seconds()/3600/2)
DogLeg_times = Tide._times(DogLeg_t0, DogLeg_hours)

# Fit the tidal data to the harmonic model using Pytides
DogLeg_tide = Tide.decompose(DogLeg_levels_nonans.T, DogLeg_dates_nonans)

# Predict the tides using the Pytides model for DogLeg measurement range
DogLeg_prediction = DogLeg_tide.at(DogLeg_times)
DogLeg_prediction_mean = DogLeg_prediction.mean() # This is less than a millimeter away from the mean of the data.
DogLeg_prediction_df = pd.DataFrame(DogLeg_prediction, index=DogLeg_times)

# Plot the results: Note that adjustments should be made to handle the river discharge.
plt.figure()
plt.plot(HR_all_resam_1hr_df['datetime'], DogLeg_levels)
plt.plot(DogLeg_times, DogLeg_prediction, label="Pytides")
ylabel_elev = 'Elevation [m NAVD88]'
DateAxisFmt(ylabel_elev)
plt.legend()

"""
High Toss Sensor
"""
HighToss_levels = np.array(HR_all_resam_1hr_df["High Toss Water Level, NAVD88"])
HighToss_nans = np.isnan(HighToss_levels)
HighToss_levels_nonans = HighToss_levels[~HighToss_nans]
HighToss_dates_nonans = HR_all_resam_1hr_df['datetime'][~HighToss_nans].reset_index(drop=True)

# Prepare a list of datetimes over HighToss date range
HighToss_t0 = HighToss_dates_nonans.iloc[0].to_pydatetime()
HighToss_tdelta = HighToss_dates_nonans.iloc[-1].to_pydatetime() - HighToss_t0
HighToss_hours = np.arange(HighToss_tdelta.total_seconds()/3600) # 56486 days from May 8, 1945 to Jan 1, 2100 (replace with "7" to represent a week)
hours_to_midpoint = int(HighToss_tdelta.total_seconds()/3600/2)
HighToss_times = Tide._times(HighToss_t0, HighToss_hours)

# Fit the tidal data to the harmonic model using Pytides
HighToss_tide = Tide.decompose(HighToss_levels_nonans.T, HighToss_dates_nonans)

# Predict the tides using the Pytides model for HighToss measurement range
HighToss_prediction = HighToss_tide.at(HighToss_times)
HighToss_prediction_mean = HighToss_prediction.mean() # This is less than a millimeter away from the mean of the data.
HighToss_prediction_df = pd.DataFrame(HighToss_prediction, index=HighToss_times)

# Plot the results: Note that adjustments should be made to handle the river discharge.
plt.figure()
plt.plot(HR_all_resam_1hr_df['datetime'], HighToss_levels)
plt.plot(HighToss_times, HighToss_prediction, label="Pytides")
ylabel_elev = 'Elevation [m NAVD88]'
DateAxisFmt(ylabel_elev)
plt.legend()

#%% 1-1-2018 to 12-31-2019 using Boston Trend at all Locations

HR_all_resam_1hr_2yr_df = HR_all_resam_1hr_df.set_index("datetime")
HR_all_resam_1hr_2yr_df = HR_all_resam_1hr_2yr_df.loc['2018-1-1':'2019-12-31']
dates_2yr_hourly = (mdates.date2num(HR_all_resam_1hr_2yr_df.index)-mdates.date2num(HR_all_resam_1hr_2yr_df.index[0]))*24
df_elevs_Boston_2yr = df_elevs_Boston.set_index("datetime")
df_elevs_Boston_2yr = df_elevs_Boston_2yr.loc['2018-1-1':'2019-12-31']

# Boston
idx_Boston_2yr = np.isfinite(dates_2yr_hourly) & np.isfinite(df_elevs_Boston_2yr['Verified NAVD Water Levels'])
z_Boston_2yr = np.polyfit(dates_2yr_hourly[idx_Boston_2yr], df_elevs_Boston_2yr['Verified NAVD Water Levels'].loc[idx_Boston_2yr], 2)
p_Boston_2yr = np.poly1d(z_Boston_2yr)
polyX_Boston_2yr = np.linspace(dates_2yr_hourly.min(), dates_2yr_hourly.max(), 100)
# Polynomial equation
print("c is acceleration in m/hr^2, b is linear rate of SLR in m/hr, and y0 is the initial mean level")
print("y=(c/2)t^2+(bt)+(y0)")
print("y=(%.6e/2)t^2+(%.6et)+(%.6f)"%(z_Boston_2yr[0]*2,z_Boston_2yr[1],z_Boston_2yr[2]))
print("Acceleration is %.6f mm/yr^2, Linear rate of SLR is %.6f mm/yr, and \n %.6f m is "
      "the mean sea level on Jan 1, 2018"%(z_Boston_2yr[0]*2*1000*8760**2,abs(z_Boston_2yr[1])*1000*8760,z_Boston_2yr[2]))
plt.figure()
plt.scatter(dates_2yr_hourly,df_elevs_Boston_2yr['Verified NAVD Water Levels'],marker='.')
pylab.plot(polyX_Boston_2yr,p_Boston_2yr(polyX_Boston_2yr),"Red", label='2018-2019 Boston Trend')
plt.xlabel('Hours Since Jan. 1, 1946', fontsize=22)
plt.ylabel('Water Level [m NAVD88]', fontsize=22)
plt.legend(loc='best', fontsize=22)

# Wellfleet Harbor Side of Dike
idx_oceanside_2yr = np.isfinite(dates_2yr_hourly) & np.isfinite(HR_all_resam_1hr_2yr_df["Gage height, m, Ocean side"])
z_oceanside_2yr = np.polyfit(dates_2yr_hourly[idx_oceanside_2yr], HR_all_resam_1hr_2yr_df["Gage height, m, Ocean side"].loc[idx_oceanside_2yr], 2)
p_oceanside_2yr = np.poly1d(z_oceanside_2yr)
polyX_oceanside_2yr = np.linspace(dates_2yr_hourly.min(), dates_2yr_hourly.max(), 100)
# Polynomial coefficients
print("Acceleration is %.6f mm/yr^2, Linear rate rise is %.6f mm/yr, and \n %.6f m is "
      "the mean level on Jan 1, 2018"%(z_oceanside_2yr[0]*2*1000*8760**2,abs(z_oceanside_2yr[1])*1000*8760,z_oceanside_2yr[2]))

# Herring River Side of Dike
idx_HRside_2yr = np.isfinite(dates_2yr_hourly) & np.isfinite(HR_all_resam_1hr_2yr_df["Gage height, m, HR side"])
z_HRside_2yr = np.polyfit(dates_2yr_hourly[idx_HRside_2yr], HR_all_resam_1hr_2yr_df["Gage height, m, HR side"].loc[idx_HRside_2yr], 2)
p_HRside_2yr = np.poly1d(z_HRside_2yr)
polyX_HRside_2yr = np.linspace(dates_2yr_hourly.min(), dates_2yr_hourly.max(), 100)
# Polynomial coefficients
print("Acceleration is %.6f mm/yr^2, Linear rate rise is %.6f mm/yr, and \n %.6f m is "
      "the mean level on Jan 1, 2018"%(z_HRside_2yr[0]*2*1000*8760**2,abs(z_HRside_2yr[1])*1000*8760,z_HRside_2yr[2]))

# CNR U/S
idx_CNRUS_2yr = np.isfinite(dates_2yr_hourly) & np.isfinite(HR_all_resam_1hr_2yr_df["CNR U/S Water Level, NAVD88"])
z_CNRUS_2yr = np.polyfit(dates_2yr_hourly[idx_CNRUS_2yr], HR_all_resam_1hr_2yr_df["CNR U/S Water Level, NAVD88"].loc[idx_CNRUS_2yr], 2)
p_CNRUS_2yr = np.poly1d(z_CNRUS_2yr)
polyX_CNRUS_2yr = np.linspace(dates_2yr_hourly.min(), dates_2yr_hourly.max(), 100)
# Polynomial coefficients
print("Acceleration is %.6f mm/yr^2, Linear rate rise is %.6f mm/yr, and \n %.6f m is "
      "the mean level on Jan 1, 2018"%(z_CNRUS_2yr[0]*2*1000*8760**2,abs(z_CNRUS_2yr[1])*1000*8760,z_CNRUS_2yr[2]))

# Dog Leg
idx_DogLeg_2yr = np.isfinite(dates_2yr_hourly) & np.isfinite(HR_all_resam_1hr_2yr_df["Dog Leg Water Level, NAVD88"])
z_DogLeg_2yr = np.polyfit(dates_2yr_hourly[idx_DogLeg_2yr], HR_all_resam_1hr_2yr_df["Dog Leg Water Level, NAVD88"].loc[idx_DogLeg_2yr], 2)
p_DogLeg_2yr = np.poly1d(z_DogLeg_2yr)
polyX_DogLeg_2yr = np.linspace(dates_2yr_hourly.min(), dates_2yr_hourly.max(), 100)
# Polynomial coefficients
print("Acceleration is %.6f mm/yr^2, Linear rate rise is %.6f mm/yr, and \n %.6f m is "
      "the mean level on Jan 1, 2018"%(z_DogLeg_2yr[0]*2*1000*8760**2,abs(z_DogLeg_2yr[1])*1000*8760,z_DogLeg_2yr[2]))

# High Toss
idx_HighToss_2yr = np.isfinite(dates_2yr_hourly) & np.isfinite(HR_all_resam_1hr_2yr_df["High Toss Water Level, NAVD88"])
z_HighToss_2yr = np.polyfit(dates_2yr_hourly[idx_HighToss_2yr], HR_all_resam_1hr_2yr_df["High Toss Water Level, NAVD88"].loc[idx_HighToss_2yr], 2)
p_HighToss_2yr = np.poly1d(z_HighToss_2yr)
polyX_HighToss_2yr = np.linspace(dates_2yr_hourly.min(), dates_2yr_hourly.max(), 100)
# Polynomial coefficients
print("Acceleration is %.6f mm/yr^2, Linear rate rise is %.6f mm/yr, and \n %.6f m is "
      "the mean level on Jan 1, 2018"%(z_HighToss_2yr[0]*2*1000*8760**2,abs(z_HighToss_2yr[1])*1000*8760,z_HighToss_2yr[2]))

# Trendline Comparison
plt.figure()
pylab.plot(polyX_Boston_2yr,p_Boston_2yr(polyX_Boston_2yr),label='2018-2019 Boston Trend, 63 mm/yr')
pylab.plot(polyX_oceanside_2yr,p_oceanside_2yr(polyX_oceanside_2yr),label='2018-2019 WF Harbor Trend, 37 mm/yr')
pylab.plot(polyX_HRside_2yr,p_HRside_2yr(polyX_HRside_2yr),label='2018-2019 HR Near-Dike Trend, 173 mm/yr')
pylab.plot(polyX_CNRUS_2yr,p_CNRUS_2yr(polyX_CNRUS_2yr),label='2018-2019 CNR U/S Trend, 37 mm/yr')
pylab.plot(polyX_DogLeg_2yr,p_DogLeg_2yr(polyX_DogLeg_2yr),label='2018-2019 Dog Leg Trend, 144 mm/yr')
pylab.plot(polyX_HighToss_2yr,p_HighToss_2yr(polyX_HighToss_2yr),label='2018-2019 High Toss Trend, 51 mm/yr')
plt.xlabel('Hours From Jan. 1, 2018 to Dec. 31, 2019', fontsize=22)
plt.ylabel('Water Level [m NAVD88]', fontsize=22)
plt.legend(loc='best', fontsize=16)

#%% 1-1-2018 to 12-31-2019 Rolling Averages and Trends

HR_all_resam_1hr_2yr_rolling_df = HR_all_resam_1hr_2yr_df.copy()

HR_all_resam_1hr_2yr_rolling_df["Ocean side, 1hr Roll Mean, Annual Window"] = HR_all_resam_1hr_2yr_rolling_df["Gage height, m, Ocean side"].rolling(window=8766, min_periods=8000, center=True).mean()
HR_all_resam_1hr_2yr_rolling_df["HR side, 1hr Roll Mean, Annual Window"] = HR_all_resam_1hr_2yr_rolling_df["Gage height, m, HR side"].rolling(window=8766, min_periods=8000, center=True).mean()
HR_all_resam_1hr_2yr_rolling_df["CNR U/S, 1hr Roll Mean, Annual Window"] = HR_all_resam_1hr_2yr_rolling_df["CNR U/S Water Level, NAVD88"].rolling(window=8766, min_periods=6000, center=True).mean()
HR_all_resam_1hr_2yr_rolling_df["Dog Leg, 1hr Roll Mean, Annual Window"] = HR_all_resam_1hr_2yr_rolling_df["Dog Leg Water Level, NAVD88"].rolling(window=8766, min_periods=5000, center=True).mean()
HR_all_resam_1hr_2yr_rolling_df["High Toss, 1hr Roll Mean, Annual Window"] = HR_all_resam_1hr_2yr_rolling_df["High Toss Water Level, NAVD88"].rolling(window=8766, min_periods=6000, center=True).mean()

# Pearson R
y_Boston = df_elevs_Boston_2yr["Verified NAVD Water Levels, Annual Rolling Average"].values
y_ocean = HR_all_resam_1hr_2yr_rolling_df["Ocean side, 1hr Roll Mean, Annual Window"].values
y_river = HR_all_resam_1hr_2yr_rolling_df["HR side, 1hr Roll Mean, Annual Window"].values
y_hightoss = HR_all_resam_1hr_2yr_rolling_df["High Toss, 1hr Roll Mean, Annual Window"].values
np.ma.corrcoef(np.ma.masked_invalid(y_ocean),np.ma.masked_invalid(y_Boston))[0,1]
np.ma.corrcoef(np.ma.masked_invalid(y_ocean),np.ma.masked_invalid(y_river))[0,1]
np.ma.corrcoef(np.ma.masked_invalid(y_river),np.ma.masked_invalid(y_hightoss))[0,1]
np.ma.corrcoef(np.ma.masked_invalid(y_ocean),np.ma.masked_invalid(y_hightoss))[0,1]

plt.figure()
plt.scatter(HR_all_resam_1hr_2yr_rolling_df.index, HR_all_resam_1hr_2yr_rolling_df["Ocean side, 1hr Roll Mean, Annual Window"], s=1, label='WF Harbor Near-Dike Levels')
plt.scatter(HR_all_resam_1hr_2yr_rolling_df.index, HR_all_resam_1hr_2yr_rolling_df["HR side, 1hr Roll Mean, Annual Window"], s=1, label='Herring River Near-Dike Levels')
plt.scatter(HR_all_resam_1hr_2yr_rolling_df.index, HR_all_resam_1hr_2yr_rolling_df["CNR U/S, 1hr Roll Mean, Annual Window"], s=1, label='CNR U/S Levels')
plt.scatter(HR_all_resam_1hr_2yr_rolling_df.index, HR_all_resam_1hr_2yr_rolling_df["Dog Leg, 1hr Roll Mean, Annual Window"], s=1, label='Dog Leg Levels')
plt.scatter(HR_all_resam_1hr_2yr_rolling_df.index, HR_all_resam_1hr_2yr_rolling_df["High Toss, 1hr Roll Mean, Annual Window"], s=1, label='High Toss Levels')

ylabel_delelev = 'Rolling Mean, Annual Window \n Water Surface Elevation [m NAVD88]'
DateAxisFmt(ylabel_delelev)
plt.xlabel('Date [YYYY-MM]')
plt.legend(loc='lower right', bbox_to_anchor=(0.7,0.4), fontsize=22, markerscale=5)

# Differences
HR_all_resam_1hr_2yr_rolling_diff_df = pd.DataFrame()
HR_all_resam_1hr_2yr_rolling_diff_df["HR side, 1hr Roll Mean Diff, Annual Window"] = HR_all_resam_1hr_2yr_rolling_df["HR side, 1hr Roll Mean, Annual Window"] - HR_all_resam_1hr_2yr_rolling_df["Ocean side, 1hr Roll Mean, Annual Window"]
HR_all_resam_1hr_2yr_rolling_diff_df["CNR U/S, 1hr Roll Mean Diff, Annual Window"] = HR_all_resam_1hr_2yr_rolling_df["CNR U/S, 1hr Roll Mean, Annual Window"] - HR_all_resam_1hr_2yr_rolling_df["Ocean side, 1hr Roll Mean, Annual Window"]
HR_all_resam_1hr_2yr_rolling_diff_df["Dog Leg, 1hr Roll Mean Diff, Annual Window"] = HR_all_resam_1hr_2yr_rolling_df["Dog Leg, 1hr Roll Mean, Annual Window"] - HR_all_resam_1hr_2yr_rolling_df["Ocean side, 1hr Roll Mean, Annual Window"]
HR_all_resam_1hr_2yr_rolling_diff_df["High Toss, 1hr Roll Mean Diff, Annual Window"] = HR_all_resam_1hr_2yr_rolling_df["High Toss, 1hr Roll Mean, Annual Window"] - HR_all_resam_1hr_2yr_rolling_df["Ocean side, 1hr Roll Mean, Annual Window"]

# Pearson R
y_riverresid = HR_all_resam_1hr_2yr_rolling_diff_df["HR side, 1hr Roll Mean Diff, Annual Window"].values
y_hightossresid = HR_all_resam_1hr_2yr_rolling_diff_df["High Toss, 1hr Roll Mean Diff, Annual Window"].values
np.ma.corrcoef(np.ma.masked_invalid(y_ocean),np.ma.masked_invalid(y_riverresid))[0,1]
np.ma.corrcoef(np.ma.masked_invalid(y_river),np.ma.masked_invalid(y_riverresid))[0,1]
np.ma.corrcoef(np.ma.masked_invalid(y_hightoss),np.ma.masked_invalid(y_riverresid))[0,1]
np.ma.corrcoef(np.ma.masked_invalid(y_ocean),np.ma.masked_invalid(y_hightossresid))[0,1]
np.ma.corrcoef(np.ma.masked_invalid(y_river),np.ma.masked_invalid(y_hightossresid))[0,1]
np.ma.corrcoef(np.ma.masked_invalid(y_hightoss),np.ma.masked_invalid(y_hightossresid))[0,1]

plt.figure()
plt.scatter(HR_all_resam_1hr_2yr_rolling_diff_df.index, HR_all_resam_1hr_2yr_rolling_diff_df["HR side, 1hr Roll Mean Diff, Annual Window"], s=1, label='Herring River Near-Dike')
plt.scatter(HR_all_resam_1hr_2yr_rolling_diff_df.index, HR_all_resam_1hr_2yr_rolling_diff_df["CNR U/S, 1hr Roll Mean Diff, Annual Window"], s=1, label='CNR U/S')
plt.scatter(HR_all_resam_1hr_2yr_rolling_diff_df.index, HR_all_resam_1hr_2yr_rolling_diff_df["High Toss, 1hr Roll Mean Diff, Annual Window"], s=1, label='High Toss')

ylabel_delelev = 'Difference in Rolling Mean, Annual Window \n WSE in HR - WSE outside dike [m NAVD88]'
DateAxisFmt(ylabel_delelev)
plt.xlabel('Date [YYYY-MM]')
plt.legend(loc='lower right', bbox_to_anchor=(0.7,0.2), fontsize=22, markerscale=5)

# HR Side
idx_HRsideroll_2yr = np.isfinite(mdates.date2num(HR_all_resam_1hr_2yr_rolling_diff_df.index)) & np.isfinite(HR_all_resam_1hr_2yr_rolling_diff_df["HR side, 1hr Roll Mean Diff, Annual Window"])
z_HRsideroll_2yr = np.polyfit(mdates.date2num(HR_all_resam_1hr_2yr_rolling_diff_df.index)[idx_HRsideroll_2yr], HR_all_resam_1hr_2yr_rolling_diff_df["HR side, 1hr Roll Mean Diff, Annual Window"].loc[idx_HRsideroll_2yr], 1)
p_HRsideroll_2yr = np.poly1d(z_HRsideroll_2yr)
polyX_HRsideroll_2yr = np.linspace(mdates.date2num(HR_all_resam_1hr_2yr_rolling_diff_df["HR side, 1hr Roll Mean Diff, Annual Window"].first_valid_index()), mdates.date2num(HR_all_resam_1hr_2yr_rolling_diff_df["HR side, 1hr Roll Mean Diff, Annual Window"].last_valid_index()), 100)
# Polynomial coefficients
print("Linear separation decrease is %.6f mm/yr " %(abs(z_HRsideroll_2yr[0])*1000*365.25))

# CNR U/S
idx_CNRUSroll_2yr = np.isfinite(mdates.date2num(HR_all_resam_1hr_2yr_rolling_diff_df.index)) & np.isfinite(HR_all_resam_1hr_2yr_rolling_diff_df["CNR U/S, 1hr Roll Mean Diff, Annual Window"])
z_CNRUSroll_2yr = np.polyfit(mdates.date2num(HR_all_resam_1hr_2yr_rolling_diff_df.index)[idx_CNRUSroll_2yr], HR_all_resam_1hr_2yr_rolling_diff_df["CNR U/S, 1hr Roll Mean Diff, Annual Window"].loc[idx_CNRUSroll_2yr], 1)
p_CNRUSroll_2yr = np.poly1d(z_CNRUSroll_2yr)
polyX_CNRUSroll_2yr = np.linspace(mdates.date2num(HR_all_resam_1hr_2yr_rolling_diff_df["CNR U/S, 1hr Roll Mean Diff, Annual Window"].first_valid_index()), mdates.date2num(HR_all_resam_1hr_2yr_rolling_diff_df["CNR U/S, 1hr Roll Mean Diff, Annual Window"].last_valid_index()), 100)
# Polynomial coefficients
print("Linear separation decrease is %.6f mm/yr " %(abs(z_CNRUSroll_2yr[0])*1000*365.25))

# High Toss
idx_HighTossroll_2yr = np.isfinite(mdates.date2num(HR_all_resam_1hr_2yr_rolling_diff_df.index)) & np.isfinite(HR_all_resam_1hr_2yr_rolling_diff_df["High Toss, 1hr Roll Mean Diff, Annual Window"])
z_HighTossroll_2yr = np.polyfit(mdates.date2num(HR_all_resam_1hr_2yr_rolling_diff_df.index)[idx_HighTossroll_2yr], HR_all_resam_1hr_2yr_rolling_diff_df["High Toss, 1hr Roll Mean Diff, Annual Window"].loc[idx_HighTossroll_2yr], 1)
p_HighTossroll_2yr = np.poly1d(z_HighTossroll_2yr)
polyX_HighTossroll_2yr = np.linspace(mdates.date2num(HR_all_resam_1hr_2yr_rolling_diff_df["High Toss, 1hr Roll Mean Diff, Annual Window"].first_valid_index()), mdates.date2num(HR_all_resam_1hr_2yr_rolling_diff_df["High Toss, 1hr Roll Mean Diff, Annual Window"].last_valid_index()), 100)
# Polynomial coefficients
print("Linear separation decrease is %.6f mm/yr " %(abs(z_HighTossroll_2yr[0])*1000*365.25))

plt.plot(polyX_HRsideroll_2yr,p_HRsideroll_2yr(polyX_HRsideroll_2yr))
plt.plot(polyX_CNRUSroll_2yr,p_CNRUSroll_2yr(polyX_CNRUSroll_2yr))
plt.plot(polyX_HighTossroll_2yr,p_HighTossroll_2yr(polyX_HighTossroll_2yr))

#%% Adjusting all levels to Boston trend (assumes rise in river directly related)

# Extend predicted levels from 1946 to 2100
Boston_t0 = df_elevs_Boston["datetime"].iloc[0].to_pydatetime()
Boston_to_2100_tdelta = datetime(2100,12,31,0,0) - Boston_t0
Boston_to_2100_hours = np.arange(Boston_to_2100_tdelta.total_seconds()/3600)
Boston_to_2100_times = Tide._times(Boston_t0, Boston_to_2100_hours)

"""
Adjusting Harborside Levels to Boston Trend
"""
date_oceanside_midpoint = oceanside_dates_nonans.iloc[int(len(oceanside_dates_nonans)/2)]
Boston_to_oceanside_mid_tdelta = date_oceanside_midpoint.to_pydatetime() - Boston_t0
Boston_to_oceanside_mid_hours = np.arange(Boston_to_oceanside_mid_tdelta.total_seconds()/3600)

# oceanside_prediction_mean = 0.115 m and Boston mean at that location is 0.025 m, so H0 needs
# to be adjusted by that difference.
Boston_mean_at_oceanside_mid = p_Boston_hourly(Boston_to_oceanside_mid_hours[-1])
Boston_to_oceanside_trend_offset = oceanside_prediction_mean - Boston_mean_at_oceanside_mid

# Predict the tides using the Pytides model from 1946 to 2100
# adjust Boston polyfit by oceanside mean
oceanside_mean_poly = Boston_to_oceanside_trend_offset + p_Boston_hourly(Boston_to_2100_hours)
# remove mean from predicted levels at oceanside, add adjusted polyfit.
oceanside_prediction_to_2100 = oceanside_mean_poly + oceanside_tide.at(Boston_to_2100_times) - oceanside_prediction_mean
oceanside_prediction_to_2100_df = pd.DataFrame(oceanside_prediction_to_2100, index=Boston_to_2100_times)

# Plot the 1945 to 2100 results
plt.figure()
plt.plot(HR_all_resam_1hr_df['datetime'], oceanside_levels)
plt.plot(Boston_to_2100_times, oceanside_prediction_to_2100, label="Pytides")
ylabel_elev = 'Elevation [m NAVD88]'
DateAxisFmt(ylabel_elev)
plt.legend()

"""
Adjusting HRside Levels to Boston Trend
"""
date_HRside_midpoint = HRside_dates_nonans.iloc[int(len(HRside_dates_nonans)/2)]
Boston_to_HRside_mid_tdelta = date_HRside_midpoint.to_pydatetime() - Boston_t0
Boston_to_HRside_mid_hours = np.arange(Boston_to_HRside_mid_tdelta.total_seconds()/3600)

# HRside_prediction_mean = 0.115 m and Boston mean at that location is 0.025 m, so H0 needs
# to be adjusted by that difference.
Boston_mean_at_HRside_mid = p_Boston_hourly(Boston_to_HRside_mid_hours[-1])
Boston_to_HRside_trend_offset = HRside_prediction_mean - Boston_mean_at_HRside_mid

# Predict the tides using the Pytides model from 1946 to 2100
# adjust Boston polyfit by HRside mean
HRside_mean_poly = Boston_to_HRside_trend_offset + p_Boston_hourly(Boston_to_2100_hours)
# remove mean from predicted levels at HRside, add adjusted polyfit.
HRside_prediction_to_2100 = HRside_mean_poly + HRside_tide.at(Boston_to_2100_times) - HRside_prediction_mean
HRside_prediction_to_2100_df = pd.DataFrame(HRside_prediction_to_2100, index=Boston_to_2100_times)

# Plot the 1945 to 2100 results
plt.figure()
plt.plot(HR_all_resam_1hr_df['datetime'], HRside_levels)
plt.plot(Boston_to_2100_times, HRside_prediction_to_2100, label="Pytides")
ylabel_elev = 'Elevation [m NAVD88]'
DateAxisFmt(ylabel_elev)
plt.legend()

"""
Adjusting CNR U/S Levels to Boston Trend
"""
date_CNRUS_midpoint = CNRUS_dates_nonans.iloc[int(len(CNRUS_dates_nonans)/2)]
Boston_to_CNRUS_mid_tdelta = date_CNRUS_midpoint.to_pydatetime() - Boston_t0
Boston_to_CNRUS_mid_hours = np.arange(Boston_to_CNRUS_mid_tdelta.total_seconds()/3600)

# CNRUS_prediction_mean = 0.115 m and Boston mean at that location is 0.025 m, so H0 needs
# to be adjusted by that difference.
Boston_mean_at_CNRUS_mid = p_Boston_hourly(Boston_to_CNRUS_mid_hours[-1])
Boston_to_CNRUS_trend_offset = CNRUS_prediction_mean - Boston_mean_at_CNRUS_mid

# Predict the tides using the Pytides model from 1946 to 2100
# adjust Boston polyfit by CNRUS mean
CNRUS_mean_poly = Boston_to_CNRUS_trend_offset + p_Boston_hourly(Boston_to_2100_hours)
# remove mean from predicted levels at CNRUS, add adjusted polyfit.
CNRUS_prediction_to_2100 = CNRUS_mean_poly + CNRUS_tide.at(Boston_to_2100_times) - CNRUS_prediction_mean
CNRUS_prediction_to_2100_df = pd.DataFrame(CNRUS_prediction_to_2100, index=Boston_to_2100_times)

# Plot the 1945 to 2100 results
plt.figure()
plt.plot(HR_all_resam_1hr_df['datetime'], CNRUS_levels)
plt.plot(Boston_to_2100_times, CNRUS_prediction_to_2100, label="Pytides")
ylabel_elev = 'Elevation [m NAVD88]'
DateAxisFmt(ylabel_elev)
plt.legend()

"""
Adjusting High Toss Levels to Boston Trend
"""
date_HighToss_midpoint = HighToss_dates_nonans.iloc[int(len(HighToss_dates_nonans)/2)]
Boston_to_HighToss_mid_tdelta = date_HighToss_midpoint.to_pydatetime() - Boston_t0
Boston_to_HighToss_mid_hours = np.arange(Boston_to_HighToss_mid_tdelta.total_seconds()/3600)

# HighToss_prediction_mean = 0.115 m and Boston mean at that location is 0.025 m, so H0 needs
# to be adjusted by that difference.
Boston_mean_at_HighToss_mid = p_Boston_hourly(Boston_to_HighToss_mid_hours[-1])
Boston_to_HighToss_trend_offset = HighToss_prediction_mean - Boston_mean_at_HighToss_mid

# Predict the tides using the Pytides model from 1946 to 2100
# adjust Boston polyfit by HighToss mean
HighToss_mean_poly = Boston_to_HighToss_trend_offset + p_Boston_hourly(Boston_to_2100_hours)
# remove mean from predicted levels at HighToss, add adjusted polyfit.
HighToss_prediction_to_2100 = HighToss_mean_poly + HighToss_tide.at(Boston_to_2100_times) - HighToss_prediction_mean
HighToss_prediction_to_2100_df = pd.DataFrame(HighToss_prediction_to_2100, index=Boston_to_2100_times)

# Plot the 1945 to 2100 results
plt.figure()
plt.plot(HR_all_resam_1hr_df['datetime'], HighToss_levels)
plt.plot(Boston_to_2100_times, HighToss_prediction_to_2100, label="PyTides Prediction")
ylabel_elev = 'Elevation [m NAVD88]'
DateAxisFmt(ylabel_elev)
plt.legend()

#%% Plot of Harborside Levels vs. HRside Levels, theoretical and measured.

HRside_prediction_to_2100_df = pd.DataFrame(HRside_prediction_to_2100, index=Boston_to_2100_times, columns=["Gage height, m, HR side, Predicted"])
HRside_prediction_2yr_df = HRside_prediction_to_2100_df.loc['2018-1-1':'2019-12-31']
oceanside_prediction_to_2100_df = pd.DataFrame(oceanside_prediction_to_2100, index=Boston_to_2100_times, columns=["Gage height, m, Ocean side, Predicted"])
oceanside_prediction_2yr_df = oceanside_prediction_to_2100_df.loc['2018-1-1':'2019-12-31']
# oceanHR_prediction_2yr_df = pd.merge(oceanside_prediction_2yr_df, HRside_prediction_2yr_df, col)

HR_all_resam_1hr_2yr_df.plot.scatter("Gage height, m, Ocean side", "Gage height, m, HR side", label='Measured')
plt.scatter(oceanside_prediction_2yr_df["Gage height, m, Ocean side, Predicted"], HRside_prediction_2yr_df["Gage height, m, HR side, Predicted"], color="Red", marker='.', label='Predicted')
plt.xlabel('WF Harbor Near-Dike Levels [m NAVD88]', fontsize=22)
plt.ylabel('HR Near-Dike Levels [m NAVD88]', fontsize=22)
plt.legend(loc='best', fontsize=22)

# oceanside_prediction_to_2100_df.reset_index(inplace=True)
# oceanside_prediction_to_2100_df.rename(columns={"index":"datetime"}, inplace=True)

#%% Cycle Volume Flux v. Predicted Ocean Side Levels

# Net Outflow from Dike
dike_outflow_arr = np.vstack([V_dike_in_dates, y_dike_res_outflows]).T
dike_outflow_df = pd.DataFrame(dike_outflow_arr, columns=["datetime","Proceeding Volume Outflux, m^3"])
dike_outflow_df["datetime"] = pd.to_datetime(dike_outflow_df["datetime"])
dike_outflow_df["Proceeding Volume Outflux, m^3"] = pd.to_numeric(dike_outflow_df["Proceeding Volume Outflux, m^3"])
outflow_dike_inflow_dike_df = pd.merge(dike_inflow_vol_df, dike_outflow_df)
net_outflow_inv = (outflow_dike_inflow_dike_df["Volume Through Dike, m^3, In"]+outflow_dike_inflow_dike_df["Proceeding Volume Outflux, m^3"])
net_outflow_inv_arr = np.vstack([V_dike_in_dates, net_outflow_inv]).T
net_outflow_inv_df = pd.DataFrame(net_outflow_inv_arr, columns=["Inflow Datetime","Cycle Volume Flux"])
net_outflow_inv_df["Inflow Datetime"] = pd.to_datetime(net_outflow_inv_df["Inflow Datetime"])
net_outflow_inv_df["Cycle Volume Flux"] = pd.to_numeric(net_outflow_inv_df["Cycle Volume Flux"])

# oceanside_pred_to_2100_merger_df = oceanside_prediction_to_2100_df.rename(columns={"datetime":"Inflow Datetime"})
# oceanside_minQ_dates_df = pd.merge(net_outflow_inv_df, oceanside_pred_to_2100_merger_df)

#%% Plot of Integrated Measured Cycle Volume Flux v. Predicted Max Oceanside Level that Cycle

# ax = oceanside_minQ_dates_df.plot.scatter(x="Gage height, m, Ocean side, Predicted", y="Cycle Volume Flux", marker='.')

# Fit trendline through min points
# oceanside_minQ_dates_df.dropna(inplace=True)
# z_oceanside_flux = np.polyfit(oceanside_minQ_dates_df["Gage height, m, Ocean side, Predicted"], oceanside_minQ_dates_df["Cycle Volume Flux"], 1)
# p_oceanside_flux = np.poly1d(z_oceanside_flux)
# polyX_oceanside_flux = np.linspace(oceanside_minQ_dates_df["Gage height, m, Ocean side, Predicted"].min(), oceanside_minQ_dates_df["Gage height, m, Ocean side, Predicted"].max(), 100)

# Using "Uncertainties" Module, Linear Regression

x = oceanside_minQ_dates_df["Gage height, m, Ocean side, Predicted"].values
y = oceanside_minQ_dates_df["Cycle Volume Flux"].values
n = len(y)

def f(x, a, b):
    return a * x + b

popt, pcov = curve_fit(f, x, y)

# retrieve parameter values
a = popt[0]
b = popt[1]
print('Optimal Values')
print('a: ' + str(a))
print('b: ' + str(b))

# compute r^2
r2 = 1.0-(sum((y-f(x,a,b))**2)/((n-1.0)*np.var(y,ddof=1)))
print('R^2: ' + str(r2))

# calculate parameter confidence interval
a,b = unc.correlated_values(popt, pcov)
print('Uncertainty')
print('a: ' + str(a))
print('b: ' + str(b))

# plot data
plt.figure()
plt.scatter(x, y, s=6, color='g', label='Measured Data')

# calculate regression confidence interval
px = np.linspace(np.nanmin(x), np.nanmax(x), 100)
py = a*px+b
nom = unp.nominal_values(py)
std = unp.std_devs(py)

def predband(x, xd, yd, p, func, conf=0.95):
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf    # significance
    N = xd.size          # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of t distribution for p=(1-alpha/2)
    q = scipy.stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb

lpb, upb = predband(px, x, y, popt, f, conf=0.95)

# plot the regression
linelabel = ("y = %.0fx + (%.0f)"%(a.nominal_value,b.nominal_value))
plt.plot(px, nom, c='black', label=linelabel)

# uncertainty lines (95% confidence)
plt.plot(px, nom - 1.96 * std, c='orange', label='95% Confidence Region')
plt.plot(px, nom + 1.96 * std, c='orange')
# prediction band (95% confidence)
plt.plot(px, lpb, 'k--',label='95% Prediction Band')
plt.plot(px, upb, 'k--')
plt.xlabel('WF Harbor Near-Dike Maximum Levels [m NAVD88]', fontsize=22)
plt.ylabel(r'In  <=  Volume Flux $\left[\frac{m^3}{cycle}\right]$  =>  Out', fontsize=22)
plt.legend(loc='upper right', fontsize=22)

# Slope is volume increase per tidal cycle per meter gained in HR.

# pylab.plot(polyX_oceanside_flux,p_oceanside_flux(polyX_oceanside_flux),color='Green',label=linelabel)

#%% Can plot NOAA verified and predicted levels with Boston data to show the reasonableness of PyTides.

# plt.plot(hours, noaa_predicted, label="NOAA Prediction")
# plt.plot(hours, noaa_verified, label="NOAA Verified")
# plt.title('Comparison of Pytides and NOAA predictions for Station: ' + str(station_id))
# plt.xlabel('Hours since ' + str(prediction_t0) + '(GMT)')

constituent = [c.name for c in oceanside_tide.model['constituent']]
constituent_df = pd.DataFrame(oceanside_tide.model, index=constituent).drop('constituent', axis=1)
constituent_df.sort_values('amplitude', ascending=False).head(30) # Sorts by constituent with largest amplitude

print('Form number %s, the tide is %s.' %(oceanside_tide.form_number()[0], oceanside_tide.classify()))

#%% Setting Hard minimum of oceanside levels based on HRside levels, using oceanside mins

oceanside_mins_df = pd.concat([HR_all_resam_1hr_mins_df["datetime"],HR_all_resam_1hr_mins_df["Min Gage height, m, Ocean side"]], axis=1)
HRside_levels_df = pd.concat([HR_all_resam_1hr_df["datetime"],HR_all_resam_1hr_df["Gage height, m, HR side"]], axis=1)
oceanside_mins_HRsametime_df = pd.merge(oceanside_mins_df, HRside_levels_df)
oceanside_mins_HRsametime_df.dropna(inplace=True)
oceanside_mins_HRsametime_df.set_index("datetime", inplace=True)

ax = oceanside_mins_HRsametime_df.plot.scatter("Min Gage height, m, Ocean side", "Gage height, m, HR side",color='Blue',label="All Data")

# Arbitrarily select bins from 0 to -0.8
oceanside_mins_HRsametime_df["HR side bins"] = pd.cut(oceanside_mins_HRsametime_df["Gage height, m, HR side"], bins=[-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0])
oceanside_mins_HRsametime_df.dropna(inplace=True)
oceanside_mins_HRbins = oceanside_mins_HRsametime_df.loc[oceanside_mins_HRsametime_df.groupby("HR side bins")["Min Gage height, m, Ocean side"].idxmin()]

oceanside_mins_HRsametime_df.plot.scatter("Min Gage height, m, Ocean side", "Gage height, m, HR side", marker='.', color='Red', label="Outliers Removed", ax=ax)
oceanside_mins_HRbins.plot.scatter("Min Gage height, m, Ocean side", "Gage height, m, HR side", marker='x', color='Green', label="Points for Setting Boundary", ax=ax)
plt.xlabel('WF Harbor Near-Dike Levels [m NAVD88]', fontsize=22)
plt.ylabel('HR Near-Dike Levels [m NAVD88]', fontsize=22)
plt.xlim(-1.2,-0.6)
plt.ylim(-1.1,0.2)

# Fit trendline through min points
idx_dike_lowlim = np.isfinite(oceanside_mins_HRbins["Min Gage height, m, Ocean side"]) & np.isfinite(oceanside_mins_HRbins["Gage height, m, HR side"])
z_dike_lowlim = np.polyfit(oceanside_mins_HRbins["Min Gage height, m, Ocean side"], oceanside_mins_HRbins["Gage height, m, HR side"], 1)
p_dike_lowlim = np.poly1d(z_dike_lowlim)
polyX_dike_lowlim = np.linspace(oceanside_mins_HRbins["Min Gage height, m, Ocean side"].min(), oceanside_mins_HRbins["Min Gage height, m, Ocean side"].max(), 100)
pylab.plot(polyX_dike_lowlim,p_dike_lowlim(polyX_dike_lowlim),color='Green',label='Lower Limit of HR Near-Dike \nLevels with Respect to WF \nHarbor Near-Dike Levels')
plt.legend(loc='lower right', bbox_to_anchor=(0.95,0.05), fontsize=22)

#%% Setting Hard minimum of oceanside levels based on HRside levels, using all data

oceanside_HRsametime_df = pd.merge(HR_all_resam_1hr_2yr_df["Gage height, m, Ocean side"], HR_all_resam_1hr_2yr_df["Gage height, m, HR side"], left_index=True, right_index=True)
oceanside_HRsametime_df.dropna(inplace=True)

ax = oceanside_HRsametime_df.plot.scatter("Gage height, m, Ocean side", "Gage height, m, HR side",color='Blue',label="All Data")

# Arbitrarily select bins from 0 to -0.8
min_bins = np.arange(-0.8,0.2,0.1)
oceanside_HRsametime_df["HR side bins"] = pd.cut(oceanside_HRsametime_df["Gage height, m, HR side"], bins=min_bins)
oceanside_HRsametime_df.dropna(inplace=True)
oceanside_mins_HRbins_all = oceanside_HRsametime_df.loc[oceanside_HRsametime_df.groupby("HR side bins")["Gage height, m, Ocean side"].idxmin()]

oceanside_HRsametime_df.plot.scatter("Gage height, m, Ocean side", "Gage height, m, HR side", marker='.', color='Red', label="Outliers Removed", ax=ax)
oceanside_mins_HRbins_all.plot.scatter("Gage height, m, Ocean side", "Gage height, m, HR side", marker='x', color='Green', label="Points for Setting Boundary", ax=ax)
plt.xlabel('WF Harbor Near-Dike Levels [m NAVD88]', fontsize=22)
plt.ylabel('HR Near-Dike Levels [m NAVD88]', fontsize=22)
plt.xlim(-1.2,-0.6)
plt.ylim(-1.1,0.2)

# Fit trendline through min points
idx_dike_lowlim_all = np.isfinite(oceanside_mins_HRbins_all["Gage height, m, Ocean side"]) & np.isfinite(oceanside_mins_HRbins_all["Gage height, m, HR side"])
z_dike_lowlim_all = np.polyfit(oceanside_mins_HRbins_all["Gage height, m, Ocean side"], oceanside_mins_HRbins_all["Gage height, m, HR side"], 1)
p_dike_lowlim_all = np.poly1d(z_dike_lowlim_all)
polyX_dike_lowlim_all = np.linspace(oceanside_mins_HRbins_all["Gage height, m, Ocean side"].min(), oceanside_mins_HRbins_all["Gage height, m, Ocean side"].max(), 100)
pylab.plot(polyX_dike_lowlim_all,p_dike_lowlim_all(polyX_dike_lowlim_all),color='Purple',label='Lower Limit of HR Near-Dike \nLevels with Respect to WF \nHarbor Near-Dike Levels')
plt.legend(loc='lower right', bbox_to_anchor=(0.95,0.05), fontsize=22)

#%% Adjust WF Harbor Side Minimum Levels

pred_to_2100_dike_df = HRside_prediction_to_2100_df.copy()
pred_to_2100_dike_df['Gage height, m, Ocean side, Predicted'] = oceanside_prediction_to_2100_df['Gage height, m, Ocean side, Predicted']
pred_to_2100_dike_df["Minimum Ocean Side Level"] = (pred_to_2100_dike_df["Gage height, m, HR side, Predicted"]-z_dike_lowlim_all[1])/z_dike_lowlim_all[0]
min_oceanside_level = (pred_to_2100_dike_df["Gage height, m, HR side, Predicted"]-z_dike_lowlim_all[1])/z_dike_lowlim_all[0]
pred_to_2100_dike_df["Minimum Ocean Side Condition"] = np.where((pred_to_2100_dike_df["Gage height, m, Ocean side, Predicted"]<pred_to_2100_dike_df["Minimum Ocean Side Level"]),True,False)
min_oceanside_cond = np.where((pred_to_2100_dike_df["Gage height, m, Ocean side, Predicted"]<pred_to_2100_dike_df["Minimum Ocean Side Level"]),True,False)

pred_to_2100_dike_df["Gage height, m, Ocean side, Predicted"].loc[min_oceanside_cond] = min_oceanside_level[min_oceanside_cond]

pred_dike_2yr_df = pred_to_2100_dike_df.loc['2018-1-1':'2019-12-31']

#%% Plot of Harborside Levels vs. HRside Levels, theoretical and measured, with hard limit on mins
# 2 year plots
HR_all_resam_1hr_2yr_df.plot.scatter("Gage height, m, Ocean side", "Gage height, m, HR side", label='Measured')
plt.scatter(pred_dike_2yr_df["Gage height, m, Ocean side, Predicted"], pred_dike_2yr_df["Gage height, m, HR side, Predicted"], color="Red", marker='.', label='Predicted')
plt.xlabel('WF Harbor Near-Dike Levels [m NAVD88]', fontsize=22)
plt.ylabel('HR Near-Dike Levels [m NAVD88]', fontsize=22)
plt.legend(loc='best', fontsize=22)

plt.figure()
plt.plot(HR_all_resam_1hr_2yr_df.index, HR_all_resam_1hr_2yr_df["Gage height, m, Ocean side"], label='Measured Levels')
plt.plot(pred_dike_2yr_df.index, pred_dike_2yr_df["Gage height, m, Ocean side, Predicted"], label='PyTides Levels')
DateAxisFmt(ylabel_elev)
plt.legend()

# 1946 to 2100 plots
pred_dike_1946_df = pred_to_2100_dike_df.loc['1946-1-1':'1946-12-31']
pred_dike_2100_df = pred_to_2100_dike_df.loc['2100-1-1':'2100-12-31']
pred_to_2100_dike_df.plot.scatter("Gage height, m, Ocean side, Predicted", "Gage height, m, HR side, Predicted", color="Gray", label='1946 to 2100')
plt.scatter(pred_dike_1946_df["Gage height, m, Ocean side, Predicted"], pred_dike_1946_df["Gage height, m, HR side, Predicted"], color="Red", marker='1', label='1946')
plt.scatter(pred_dike_2yr_df["Gage height, m, Ocean side, Predicted"], pred_dike_2yr_df["Gage height, m, HR side, Predicted"], color="Blue", marker='.', label='2018 to 2019')
plt.scatter(pred_dike_2100_df["Gage height, m, Ocean side, Predicted"], pred_dike_2100_df["Gage height, m, HR side, Predicted"], color="Green", marker='2', label='2100')
plt.xlabel('Predicted WF Harbor Near-Dike Levels [m NAVD88]', fontsize=22)
plt.ylabel('Predicted HR Near-Dike Levels [m NAVD88]', fontsize=22)
plt.legend(loc='best', bbox_to_anchor=(0.95,0.3), fontsize=22)

plt.figure()
plt.plot(HR_all_resam_1hr_df["datetime"], HR_all_resam_1hr_df["Gage height, m, Ocean side"], label="Measured Levels")
plt.plot(pred_to_2100_dike_df.index, pred_to_2100_dike_df["Gage height, m, Ocean side, Predicted"], label="PyTides Levels")
DateAxisFmt(ylabel_elev)
plt.legend()

#%% Kling-Gupta, RMSE, and Nash-Sutcliffe Efficiency
# It should also be noted that kge and kgeprime return four values for each simulated timeseries. 
# Indeed, it returns the KGE or KGE' value, as well as their three respective components 
# (r/$\alpha$/$\beta$, and r/$\gamma$/$\beta$, respectively). However, kge_c2m and kgeprime_c2m 
# only return one value, that is the corresponding bounded KGE value only.

# Ocean side
simulated_pytides_level_oceanside = np.array(pred_dike_2yr_df["Gage height, m, Ocean side, Predicted"])
observed_level_oceanside = np.array(HR_all_resam_1hr_2yr_df["Gage height, m, Ocean side"])

# use the evaluator with the Kling-Gupta Efficiency (objective function 1)
my_kge_oceanside = evaluator(kge, simulated_pytides_level_oceanside, observed_level_oceanside)

# use the evaluator with the Kling-Gupta Efficiency for inverted flow series (objective function 2)
my_kge_inv_oceanside = evaluator(kge, simulated_pytides_level_oceanside, observed_level_oceanside, transform='inv')

# use the evaluator with the Root Mean Square Error (objective function 3)
my_rmse_oceanside = evaluator(rmse, simulated_pytides_level_oceanside, observed_level_oceanside)

my_nse_oceanside = evaluator(nse, simulated_pytides_level_oceanside, observed_level_oceanside)

# HR side
simulated_pytides_level_HRside = np.array(pred_dike_2yr_df["Gage height, m, HR side, Predicted"])
observed_level_HRside = np.array(HR_all_resam_1hr_2yr_df["Gage height, m, HR side"])

# use the evaluator with the Kling-Gupta Efficiency (objective function 1)
my_kge_HRside = evaluator(kge, simulated_pytides_level_HRside, observed_level_HRside)

# use the evaluator with the Kling-Gupta Efficiency for inverted flow series (objective function 2)
my_kge_inv_HRside = evaluator(kge, simulated_pytides_level_HRside, observed_level_HRside, transform='inv')

# use the evaluator with the Root Mean Square Error (objective function 3)
my_rmse_HRside = evaluator(rmse, simulated_pytides_level_HRside, observed_level_HRside)

my_nse_HRside = evaluator(nse, simulated_pytides_level_HRside, observed_level_HRside)

# High Toss

#%% Plot of Residuals for 2018-2019
# obs - pred

# Ocean side
resid_oceanside = observed_level_oceanside - simulated_pytides_level_oceanside

plt.figure()
plt.scatter(observed_level_oceanside, resid_oceanside)

plt.figure()
sns.distplot(resid_oceanside, hist = False, kde = True,kde_kws = {'linewidth': 3})

plt.figure() # Why is the residuals plot opposite (simulated-observed?)
sns.residplot(observed_level_oceanside, simulated_pytides_level_oceanside, lowess=True, color="g")

#%% Save Predictions

# pred_to_2100_dtHRocean_df = pred_to_2100_dike_df.drop(columns=["Minimum Ocean Side Level","Minimum Ocean Side Condition"])
# pred_to_2100_dtHRocean_df.reset_index(inplace=True)
# pred_to_2100_dtHRocean_df.rename(columns={"index":"datetime"}, inplace=True)
# pred_to_2100_dtHRocean_df.to_csv(os.path.join(data_dir, 'General Dike Data', 'Dike_Data_HourlyPred_111946_12312100.csv'), index = False)

# CNRUS_prediction_to_2100_df.rename(columns={0:"Gage height, m, CNRUS, Predicted"}, inplace=True)
# HighToss_prediction_to_2100_df.rename(columns={0:"Gage height, m, High Toss, Predicted"}, inplace=True)
# pred_to_2100_CNRUS_HT_df = pd.concat([CNRUS_prediction_to_2100_df, HighToss_prediction_to_2100_df],axis=1)
# pred_to_2100_CNRUS_HT_df.reset_index(inplace=True)
# pred_to_2100_CNRUS_HT_df.rename(columns={"index":"datetime"}, inplace=True)
# pred_to_2100_CNRUS_HT_df.to_csv(os.path.join(data_dir, 'General Dike Data', 'CNRUS_HT_HourlyPred_111946_12312100.csv'), index = False)


#%% Load Predictions

pred_to_2100_dtHRocean_df = pd.read_csv(os.path.join(data_dir, 'General Dike Data', 'Dike_Data_HourlyPred_111946_12312100.csv'))
pred_to_2100_CNRUS_HT_df = pd.read_csv(os.path.join(data_dir, 'General Dike Data', 'CNRUS_HT_HourlyPred_111946_12312100.csv'))

pred_to_2100_dtHRocean_df["datetime"] = pd.to_datetime(pred_to_2100_dtHRocean_df["datetime"], infer_datetime_format=True)
pred_to_2100_dtHRocean_df.set_index('datetime',inplace=True)
pred_to_2100_CNRUS_HT_df["datetime"] = pd.to_datetime(pred_to_2100_CNRUS_HT_df["datetime"], infer_datetime_format=True)
pred_to_2100_CNRUS_HT_df.set_index('datetime',inplace=True)

pred_dtHRocean_2yr_df = pred_to_2100_dtHRocean_df.loc['2018-1-1':'2019-12-31']
pred_CNRUS_HT_2yr_df = pred_to_2100_CNRUS_HT_df.loc['2018-1-1':'2019-12-31']

# Plot 2yr Predictions
plt.figure()
plt.plot(HR_all_resam_1hr_2yr_df.index, HR_all_resam_1hr_2yr_df["Gage height, m, Ocean side"], label='WF Harbor Side of Dike, Measured Data')
plt.plot(pred_dtHRocean_2yr_df.index, pred_dtHRocean_2yr_df["Gage height, m, Ocean side, Predicted"], label='Harmonic Fit')
DateAxisFmt(ylabel_elev)
plt.xlabel('Date [YYYY-MM-DD]')
plt.legend()

#%% Histogram of residuals

oceanside_residuals = pd.DataFrame(HR_all_resam_1hr_2yr_df["Gage height, m, Ocean side"]-pred_dtHRocean_2yr_df["Gage height, m, Ocean side, Predicted"])
oceanside_residuals.rename(columns={0:'Residuals, m, Ocean side'}, inplace=True)
HRside_residuals = pd.DataFrame(HR_all_resam_1hr_2yr_df["Gage height, m, HR side"]-pred_dtHRocean_2yr_df["Gage height, m, HR side, Predicted"])
HRside_residuals.rename(columns={0:'Residuals, m, HR side'}, inplace=True)

# Plot Oceanside Residuals
ax = oceanside_residuals.plot.hist(bins=30,alpha=0.5)
plt.xlabel('WF Harbor Near-Dike Residuals, Observed-Modeled [m]')
plt.ylabel('Count')
plt.legend()
ax.get_legend().remove()

# Plot HRside Residuals
ax = HRside_residuals.plot.hist(bins=40,alpha=0.5)
plt.xlabel('HR Near-Dike Residuals, Observed-Modeled [m]')
plt.ylabel('Count')
plt.xlim(xmin=-0.5,xmax=0.5)
plt.legend()
ax.get_legend().remove()

#%% Kling-Gupta, RMSE, and Nash-Sutcliffe Efficiency
# It should also be noted that kge and kgeprime return four values for each simulated timeseries. 
# Indeed, it returns the KGE or KGE' value, as well as their three respective components 
# (r/$\alpha$/$\beta$, and r/$\gamma$/$\beta$, respectively). However, kge_c2m and kgeprime_c2m 
# only return one value, that is the corresponding bounded KGE value only.

# Ocean side
simulated_pytides_level_oceanside = np.array(pred_dtHRocean_2yr_df["Gage height, m, Ocean side, Predicted"])
observed_level_oceanside = np.array(HR_all_resam_1hr_2yr_df["Gage height, m, Ocean side"])

# use the evaluator with the Kling-Gupta Efficiency (objective function 1)
my_kge_oceanside = evaluator(kge, simulated_pytides_level_oceanside, observed_level_oceanside)

# use the evaluator with the Kling-Gupta Efficiency for inverted flow series (objective function 2)
my_kge_inv_oceanside = evaluator(kge, simulated_pytides_level_oceanside, observed_level_oceanside, transform='inv')

# use the evaluator with the Root Mean Square Error (objective function 3)
my_rmse_oceanside = evaluator(rmse, simulated_pytides_level_oceanside, observed_level_oceanside)

my_nse_oceanside = evaluator(nse, simulated_pytides_level_oceanside, observed_level_oceanside)

# HR side
simulated_pytides_level_HRside = np.array(pred_dtHRocean_2yr_df["Gage height, m, HR side, Predicted"])
observed_level_HRside = np.array(HR_all_resam_1hr_2yr_df["Gage height, m, HR side"])

# use the evaluator with the Kling-Gupta Efficiency (objective function 1)
my_kge_HRside = evaluator(kge, simulated_pytides_level_HRside, observed_level_HRside)

# use the evaluator with the Kling-Gupta Efficiency for inverted flow series (objective function 2)
my_kge_inv_HRside = evaluator(kge, simulated_pytides_level_HRside, observed_level_HRside, transform='inv')

# use the evaluator with the Root Mean Square Error (objective function 3)
my_rmse_HRside = evaluator(rmse, simulated_pytides_level_HRside, observed_level_HRside)

my_nse_HRside = evaluator(nse, simulated_pytides_level_HRside, observed_level_HRside)

# High Toss
my_rmse_HighToss = evaluator(rmse, np.array(pred_CNRUS_HT_2yr_df["Gage height, m, High Toss, Predicted"]),np.array(HR_all_resam_1hr_2yr_df["High Toss Water Level, NAVD88"]))

#%% Slice the predictions from the beginning to present. Trend. Get last value as present "mean". Build off surface scenarios.

pred_dtHRocean_hindcast_df = pred_to_2100_dtHRocean_df.loc['1946-1-1':'2019-12-31']

# harbor trend
dates_dtHRocean_hindcast = mdates.date2num(pred_dtHRocean_hindcast_df.index)
z_dtocean_hindcast = np.polyfit(dates_dtHRocean_hindcast, pred_dtHRocean_hindcast_df["Gage height, m, Ocean side, Predicted"], 2)
p_dtocean_hindcast = np.poly1d(z_dtocean_hindcast)
polyX_dtHRocean_hindcast = np.linspace(dates_dtHRocean_hindcast.min(), dates_dtHRocean_hindcast.max(), 100)

# hr trend
z_dtHR_hindcast = np.polyfit(dates_dtHRocean_hindcast, pred_dtHRocean_hindcast_df["Gage height, m, HR side, Predicted"], 2)
p_dtHR_hindcast = np.poly1d(z_dtHR_hindcast)

# Plot Hindcast Predictions, harbor side
plt.figure()
plt.plot(pred_dtHRocean_hindcast_df.index, pred_dtHRocean_hindcast_df["Gage height, m, Ocean side, Predicted"], label='Harmonic Fit')
plt.plot(polyX_dtHRocean_hindcast,p_dtocean_hindcast(polyX_dtHRocean_hindcast))
DateAxisFmt(ylabel_elev)
plt.xlabel('Year')
plt.legend()

# Plot Hindcast Predictions, hr side
plt.figure()
plt.plot(pred_dtHRocean_hindcast_df.index, pred_dtHRocean_hindcast_df["Gage height, m, HR side, Predicted"], label='Harmonic Fit')
plt.plot(polyX_dtHRocean_hindcast,p_dtHR_hindcast(polyX_dtHRocean_hindcast))
DateAxisFmt(ylabel_elev)
plt.xlabel('Year')
plt.legend()

# SLR 1, harbor
slr1_harbor_points_df = pd.DataFrame({'datenum':[polyX_dtHRocean_hindcast[-1],polyX_dtHRocean_hindcast[-1]+31*365.25,polyX_dtHRocean_hindcast[-1]+81*365.25],
                                      'water_level':[p_dtocean_hindcast(polyX_dtHRocean_hindcast)[-1],p_dtocean_hindcast(polyX_dtHRocean_hindcast)[-1]+0.19,p_dtocean_hindcast(polyX_dtHRocean_hindcast)[-1]+0.46]})

slr1_harbor_coeffs = np.polyfit(slr1_harbor_points_df['datenum'],slr1_harbor_points_df['water_level'],2)
slr1_harbor_eqn = np.poly1d(slr1_harbor_coeffs)
polyX_2019to2100 = np.linspace(slr1_harbor_points_df['datenum'].min(), slr1_harbor_points_df['datenum'].max(), 100)

# SLR 2, harbor
slr2_harbor_points_df = pd.DataFrame({'datenum':[polyX_dtHRocean_hindcast[-1],polyX_dtHRocean_hindcast[-1]+31*365.25,polyX_dtHRocean_hindcast[-1]+81*365.25],
                                      'water_level':[p_dtocean_hindcast(polyX_dtHRocean_hindcast)[-1],p_dtocean_hindcast(polyX_dtHRocean_hindcast)[-1]+0.33,p_dtocean_hindcast(polyX_dtHRocean_hindcast)[-1]+1.11]})

slr2_harbor_coeffs = np.polyfit(slr2_harbor_points_df['datenum'],slr2_harbor_points_df['water_level'],2)
slr2_harbor_eqn = np.poly1d(slr2_harbor_coeffs)

# SLR 3, harbor
slr3_harbor_points_df = pd.DataFrame({'datenum':[polyX_dtHRocean_hindcast[-1],polyX_dtHRocean_hindcast[-1]+31*365.25,polyX_dtHRocean_hindcast[-1]+81*365.25],
                                      'water_level':[p_dtocean_hindcast(polyX_dtHRocean_hindcast)[-1],p_dtocean_hindcast(polyX_dtHRocean_hindcast)[-1]+0.67,p_dtocean_hindcast(polyX_dtHRocean_hindcast)[-1]+2.5]})

slr3_harbor_coeffs = np.polyfit(slr3_harbor_points_df['datenum'],slr3_harbor_points_df['water_level'],2)
slr3_harbor_eqn = np.poly1d(slr3_harbor_coeffs)

# Plot Int-Low to High Preds for Local SLR on harbor side.
plt.figure()
plt.plot(polyX_dtHRocean_hindcast,p_dtocean_hindcast(polyX_dtHRocean_hindcast), label = 'Quadratic Trend, WF Harbor Harmonic Fit')
plt.plot(polyX_2019to2100,slr1_harbor_eqn(polyX_2019to2100), label = 'Quadratic Forecast, Int-Low SLR')
plt.plot(polyX_2019to2100,slr2_harbor_eqn(polyX_2019to2100), label = 'Quadratic Forecast, Int SLR')
plt.plot(polyX_2019to2100,slr3_harbor_eqn(polyX_2019to2100), label = 'Quadratic Forecast, High SLR')
DateAxisFmt(ylabel_elev)
plt.xlabel('Year')
plt.ylabel('WSE, WF Harbor Side of Dike [m NAVD88]')
plt.legend()

diff_in_means_dike = p_dtocean_hindcast(polyX_dtHRocean_hindcast)[-1]-p_dtHR_hindcast(polyX_dtHRocean_hindcast)[-1]

# SLR 1, hr
slr1_hr_points_df = pd.DataFrame({'datenum':[polyX_dtHRocean_hindcast[-1],polyX_dtHRocean_hindcast[-1]+31*365.25,polyX_dtHRocean_hindcast[-1]+81*365.25],
                                      'water_level':[p_dtHR_hindcast(polyX_dtHRocean_hindcast)[-1],p_dtHR_hindcast(polyX_dtHRocean_hindcast)[-1]+0.19,p_dtHR_hindcast(polyX_dtHRocean_hindcast)[-1]+0.46]})

slr1_hr_coeffs = np.polyfit(slr1_hr_points_df['datenum'],slr1_hr_points_df['water_level'],2)
slr1_hr_eqn = np.poly1d(slr1_hr_coeffs)
polyX_2019to2100 = np.linspace(slr1_hr_points_df['datenum'].min(), slr1_hr_points_df['datenum'].max(), 100)

# SLR 2, hr
slr2_hr_points_df = pd.DataFrame({'datenum':[polyX_dtHRocean_hindcast[-1],polyX_dtHRocean_hindcast[-1]+31*365.25,polyX_dtHRocean_hindcast[-1]+81*365.25],
                                      'water_level':[p_dtHR_hindcast(polyX_dtHRocean_hindcast)[-1],p_dtHR_hindcast(polyX_dtHRocean_hindcast)[-1]+0.33,p_dtHR_hindcast(polyX_dtHRocean_hindcast)[-1]+1.11]})

slr2_hr_coeffs = np.polyfit(slr2_hr_points_df['datenum'],slr2_hr_points_df['water_level'],2)
slr2_hr_eqn = np.poly1d(slr2_hr_coeffs)

# SLR 3, hr
slr3_hr_points_df = pd.DataFrame({'datenum':[polyX_dtHRocean_hindcast[-1],polyX_dtHRocean_hindcast[-1]+31*365.25,polyX_dtHRocean_hindcast[-1]+81*365.25],
                                      'water_level':[p_dtHR_hindcast(polyX_dtHRocean_hindcast)[-1],p_dtHR_hindcast(polyX_dtHRocean_hindcast)[-1]+0.67,p_dtHR_hindcast(polyX_dtHRocean_hindcast)[-1]+2.5]})

slr3_hr_coeffs = np.polyfit(slr3_hr_points_df['datenum'],slr3_hr_points_df['water_level'],2)
slr3_hr_eqn = np.poly1d(slr3_hr_coeffs)

# SLR 1, harbor, same rise hr
slr1_r1_hr_coeffs = slr1_harbor_coeffs - [0,0,diff_in_means_dike]
slr1_r1_hr_eqn = np.poly1d(slr1_r1_hr_coeffs)
# SLR 1, harbor, mid relative rise hr
slr1_r2_hr_polyvals = slr1_hr_eqn(polyX_2019to2100)+0.03*(polyX_2019to2100-polyX_2019to2100[0])/365.25
slr1_r2_hr_polyvals[slr1_r2_hr_polyvals>slr1_harbor_eqn(polyX_2019to2100)] = slr1_harbor_eqn(polyX_2019to2100)[slr1_r2_hr_polyvals>slr1_harbor_eqn(polyX_2019to2100)]
# SLR 1, harbor, high relative rise hr
slr1_r3_hr_polyvals = slr1_hr_eqn(polyX_2019to2100)+0.06*(polyX_2019to2100-polyX_2019to2100[0])/365.25
slr1_r3_hr_polyvals[slr1_r3_hr_polyvals>slr1_harbor_eqn(polyX_2019to2100)] = slr1_harbor_eqn(polyX_2019to2100)[slr1_r3_hr_polyvals>slr1_harbor_eqn(polyX_2019to2100)]
# SLR1 Plots
plt.figure()
plt.plot(polyX_dtHRocean_hindcast,p_dtocean_hindcast(polyX_dtHRocean_hindcast),label='Quadratic Trend, WF Harbor Harmonic Fit')
plt.plot(polyX_dtHRocean_hindcast,p_dtHR_hindcast(polyX_dtHRocean_hindcast),label='Quadratic Trend, HR Harmonic Fit')
plt.plot(polyX_2019to2100,slr1_harbor_eqn(polyX_2019to2100), label = 'WF, Int-Low SLR')
plt.plot(polyX_2019to2100,slr1_r1_hr_eqn(polyX_2019to2100), label = 'HR 1:1 Rise, Int-Low SLR')
plt.plot(polyX_2019to2100,slr1_r2_hr_polyvals, label = 'HR 30 mm/yr Relative Rise, Int-Low SLR')
plt.plot(polyX_2019to2100,slr1_r3_hr_polyvals, label = 'HR 60 mm/yr Relative Rise, Int-Low SLR')
DateAxisFmt(ylabel_elev)
plt.xlabel('Year')
plt.ylabel('WSE [m NAVD88]')
plt.legend()

# SLR 2, harbor, low relative rise hr
slr2_r1_hr_coeffs = slr2_harbor_coeffs - [0,0,diff_in_means_dike]
slr2_r1_hr_eqn = np.poly1d(slr2_r1_hr_coeffs)
# SLR 2, harbor, mid relative rise hr
slr2_r2_hr_polyvals = slr2_hr_eqn(polyX_2019to2100)+0.03*(polyX_2019to2100-polyX_2019to2100[0])/365.25
slr2_r2_hr_polyvals[slr2_r2_hr_polyvals>slr2_harbor_eqn(polyX_2019to2100)] = slr2_harbor_eqn(polyX_2019to2100)[slr2_r2_hr_polyvals>slr2_harbor_eqn(polyX_2019to2100)]
# SLR 2, harbor, high relative rise hr
slr2_r3_hr_polyvals = slr2_hr_eqn(polyX_2019to2100)+0.06*(polyX_2019to2100-polyX_2019to2100[0])/365.25
slr2_r3_hr_polyvals[slr2_r3_hr_polyvals>slr2_harbor_eqn(polyX_2019to2100)] = slr2_harbor_eqn(polyX_2019to2100)[slr2_r3_hr_polyvals>slr2_harbor_eqn(polyX_2019to2100)]
# SLR2 Plots
plt.figure()
plt.plot(polyX_dtHRocean_hindcast,p_dtocean_hindcast(polyX_dtHRocean_hindcast), label = 'Quadratic Trend, WF Harbor Harmonic Fit')
plt.plot(polyX_dtHRocean_hindcast,p_dtHR_hindcast(polyX_dtHRocean_hindcast),label='Quadratic Trend, HR Harmonic Fit')
plt.plot(polyX_2019to2100,slr2_harbor_eqn(polyX_2019to2100), label = 'WF, Int SLR')
plt.plot(polyX_2019to2100,slr2_r1_hr_eqn(polyX_2019to2100), label = 'HR 1:1 Rise, Int SLR')
plt.plot(polyX_2019to2100,slr2_r2_hr_polyvals, label = 'HR 30 mm/yr Relative Rise, Int SLR')
plt.plot(polyX_2019to2100,slr2_r3_hr_polyvals, label = 'HR 60 mm/yr Relative Rise, Int SLR')
DateAxisFmt(ylabel_elev)
plt.xlabel('Year')
plt.ylabel('WSE [m NAVD88]')
plt.legend()

# SLR 3, harbor, low relative rise hr
slr3_r1_hr_coeffs = slr3_harbor_coeffs - [0,0,diff_in_means_dike]
slr3_r1_hr_eqn = np.poly1d(slr3_r1_hr_coeffs)
# SLR 3, harbor, mid relative rise hr

# SLR 3, harbor, high relative rise hr

# SLR3 Plots
plt.figure()
plt.plot(polyX_dtHRocean_hindcast,p_dtocean_hindcast(polyX_dtHRocean_hindcast), label = 'Quadratic Trend, WF Harbor Harmonic Fit')
plt.plot(polyX_dtHRocean_hindcast,p_dtHR_hindcast(polyX_dtHRocean_hindcast),label='Quadratic Trend, HR Harmonic Fit')
plt.plot(polyX_2019to2100,slr3_harbor_eqn(polyX_2019to2100), label = 'WF, High SLR')
plt.plot(polyX_2019to2100,slr1_harbor_eqn(polyX_2019to2100), label = 'HR 1:1 Rise, High SLR')
plt.plot(polyX_2019to2100,slr2_harbor_eqn(polyX_2019to2100), label = 'HR 30 mm/yr Relative Rise, High SLR')
plt.plot(polyX_2019to2100,slr3_harbor_eqn(polyX_2019to2100), label = 'HR 60 mm/yr Relative Rise, High SLR')
DateAxisFmt(ylabel_elev)
plt.xlabel('Year')
plt.ylabel('WSE [m NAVD88]')
plt.legend()

