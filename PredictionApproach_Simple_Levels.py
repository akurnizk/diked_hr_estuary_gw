# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:54:55 2020

@author: akurnizk
"""

import json #convert web request response as a structured json
import sys, os
import requests #needed to make web requests
import hydroeval
import numpy as np
import pandas as pd #store the data we get as a dataframe
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime #parse the datetimes we get from NOAA

from matplotlib import pylab
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

from functools import reduce
from scipy.optimize import curve_fit

import uncertainties.unumpy as unp
import uncertainties as unc

from pycse import regress

import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import scipy
from pytides import astro
from pytides.tide import Tide

import matplotlib as mpl
mpl.rc('xtick', labelsize=22)     
mpl.rc('ytick', labelsize=22)
mpl.rcParams['pdf.fonttype'] = 42

map_dir = r'E:\Maps' # retrieved files from https://viewer.nationalmap.gov/basic/
data_dir = os.path.join('E:\Data')

%matplotlib qt

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

#%% Load Dataframe of all HR data & Geometry

HR_all_resam_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","HR_All_Data_Resampled_5min_8272017-1212020.csv")) # Calculated
data_cols = HR_all_resam_df.columns.drop("datetime")
HR_all_resam_df[data_cols] = HR_all_resam_df[data_cols].apply(pd.to_numeric, errors='coerce')
HR_all_resam_df["datetime"] = pd.to_datetime(HR_all_resam_df["datetime"])

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

#%% Plot of everything

ax = HR_all_resam_df.plot.scatter(x="datetime", y="Gage height, m, Ocean side", color='LightBlue', label = 'Gage height, m , Ocean side')
HR_all_resam_df.plot.scatter(x="datetime", y="Gage height, m, HR side", color='LightGreen', label = 'Gage height, m , HR side', ax=ax)
HR_all_resam_df.plot.scatter(x="datetime", y="Discharge, cms", color='Turquoise', label = 'Discharge, cms', ax=ax)
HR_all_resam_df.plot.scatter(x="datetime", y="CNR U/S Water Level, NAVD88", color='DarkGreen', label = 'Water Level, m, CNR U/S', ax=ax)
HR_all_resam_df.plot.scatter(x="datetime", y="Dog Leg Water Level, NAVD88", color='DarkRed', label = 'Water Level, m, Dog Leg', ax=ax)
HR_all_resam_df.plot.scatter(x="datetime", y="High Toss Water Level, NAVD88", color='DarkOrange', label = 'Water Level, m, High Toss', ax=ax)
HR_all_resam_df.plot.scatter(x="datetime", y="Discharge, Dike Calc, cms", color='DarkBlue', label = 'Dike Calculated Discharge, cms', ax=ax)

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

#%% Downscaling to 30 minutes and hourly

# Take only 30 minute interval measurements
del_30min_range = pd.date_range(start=HR_all_resam_df["datetime"].iloc[0], end=HR_all_resam_df["datetime"].iloc[-1], freq='30min')
del_30min_range = pd.DataFrame(del_30min_range,columns=['datetime'])
HR_all_resam_df_di = HR_all_resam_df.set_index('datetime')
HR_all_resam30min_df_di = HR_all_resam_df_di.reindex(del_30min_range["datetime"])
HR_all_resam30min_df = HR_all_resam30min_df_di.reset_index()

# average by merging 30min resample with 5min and averaging between?
#%% Save
# HR_all_resam30min_df.to_csv(os.path.join(data_dir,"General Dike Data","HR_All_Data_Resampled_30min_8272017-1212020.csv"), index = False)

#%% Resample to hourly averages of data
hourly_summary_onhalfhr_di = HR_all_resam_df_di.resample('H', loffset='27.5min').mean() # on half of 55 minutes.
hourly_summary_onhalfhr = hourly_summary_onhalfhr_di.reset_index()

# shift and average (interpolate doesn't matter - just selecting the values and shifting them 27.5 minutes)
# this makes maxes and mins lower and higher, respectively, than realistic, but also minimizes anomalous effects of waves

hourly_summary_onhr = [] # faster to append to list then make frame
bin_old=HR_all_resam_df_di.index[0]
for row in range(len(HR_all_resam_df_di)):
    if (HR_all_resam_df_di.index.minute[row]==30): 
        bin_new = HR_all_resam_df_di.index[row]
        hourly_summary_onhr.append(HR_all_resam_df_di[bin_old:bin_new].mean())
        bin_old = HR_all_resam_df_di.index[row]
                               
hourly_summary_onhr_df = pd.DataFrame(hourly_summary_onhr)
hourly_summary_onhr_df = hourly_summary_onhr_df[1:]

# this line is just for the datetime indexes
hourly_summary_onhr_di = HR_all_resam_df_di.resample('H').mean()[1:] # first isn't a representative mean.

hourly_summary_onhr_df.index = hourly_summary_onhr_di.index
hourly_summary_onhr_df.reset_index(inplace=True)

#%% Write
# hourly_summary_onhr_df.to_csv(os.path.join(data_dir,"General Dike Data","HR_All_Data_Resampled_HourlyMeans_8272017-1212020.csv"), index = False)

#%% Read
hourly_summary_onhr_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","HR_All_Data_Resampled_HourlyMeans_8272017-1212020.csv")) # Calculated
data_cols = hourly_summary_onhr_df.columns.drop("datetime")
hourly_summary_onhr_df[data_cols] = hourly_summary_onhr_df[data_cols].apply(pd.to_numeric, errors='coerce')
hourly_summary_onhr_df["datetime"] = pd.to_datetime(hourly_summary_onhr_df["datetime"])

#%% Plot to compare reindexing from 5 min to 30 min to hourly means

plt.figure()
ax = HR_all_resam_df.plot.scatter(x="datetime", y="Gage height, m, Ocean side", color='LightBlue', label = 'Gage height, m , Ocean side, 5min')
HR_all_resam30min_df.plot.scatter(x="datetime", y="Gage height, m, Ocean side", color='LightGreen', label = 'Gage height, m , Ocean side, 30min', ax=ax)
hourly_summary_onhalfhr.plot.scatter(x="datetime", y="Gage height, m, Ocean side", color='LightSalmon', label = 'Gage height, m , Ocean side, on27.5, 1hr_avg', ax=ax)
hourly_summary_onhr_df.plot.scatter(x="datetime", y="Gage height, m, Ocean side", color='Green', label = 'Gage height, m , Ocean side, on27.5, 1hr_avg', ax=ax)

HR_all_resam_df.plot.scatter(x="datetime", y="Gage height, m, HR side", color='Red', label = 'Gage height, m , HR side, 5min', ax=ax)
HR_all_resam30min_df.plot.scatter(x="datetime", y="Gage height, m, HR side", color='Black', label = 'Gage height, m , HR side, 30min', ax=ax)
hourly_summary_onhalfhr.plot.scatter(x="datetime", y="Gage height, m, HR side", color='Salmon', label = 'Gage height, m , HR side, 1hr_avg', ax=ax)
hourly_summary_onhr_df.plot.scatter(x="datetime", y="Gage height, m, HR side", color='Grey', label = 'Gage height, m , HR side, 1hr_avg', ax=ax)

# Show X-axis major tick marks as dates
ylabel_elev = 'Elevation (m)'
DateAxisFmt(ylabel_elev)
plt.legend(loc='upper right')

#%% Moving Average of Hourly WF Harbor and HR Levels, compared

WF_HR_1hr_df = hourly_summary_onhr_df.filter(["datetime", "Gage height, m, Ocean side", "Gage height, m, HR side"], axis=1)
WF_HR_meas_istart = WF_HR_1hr_df["Gage height, m, Ocean side"].first_valid_index()
WF_HR_meas_iend = WF_HR_1hr_df["Gage height, m, HR side"].last_valid_index()
WF_HR_1hr_df = WF_HR_1hr_df.iloc[WF_HR_meas_istart:WF_HR_meas_iend]
WF_HR_1hr_df.reset_index(drop=True, inplace=True)

# Rolling means give more stable differences: Try Daily, Monthly, Yearly

# Interpolate some of the missing HR values
WF_HR_1hr_df["Gage height, m, HR side"].interpolate(method='cubic', limit=5, limit_direction='both', limit_area='inside', inplace=True)

WF_HR_1hr_df["Residuals"] = WF_HR_1hr_df["Gage height, m, Ocean side"] - WF_HR_1hr_df["Gage height, m, HR side"]
plt.scatter(WF_HR_1hr_df["datetime"], WF_HR_1hr_df["Residuals"])


#%% Tidal Day Window

WF_HR_1hr_df["Daily Tidal Rolling Residuals"] = WF_HR_1hr_df["Residuals"].rolling(window=25, min_periods=24, center=True).mean()

WF_HR_1hr_df["Ocean side, 1hr Roll Mean, Daily Tidal Window"] = WF_HR_1hr_df["Gage height, m, Ocean side"].rolling(window=25, min_periods=24, center=True).mean()
WF_HR_1hr_df["HR side, 1hr Roll Mean, Daily Tidal Window"] = WF_HR_1hr_df["Gage height, m, HR side"].rolling(window=25, min_periods=24, center=True).mean()

plt.figure()
plt.scatter(WF_HR_1hr_df["datetime"], WF_HR_1hr_df["Ocean side, 1hr Roll Mean, Daily Tidal Window"], s=1, label='WF Harbor Near-Dike Levels')
plt.scatter(WF_HR_1hr_df["datetime"], WF_HR_1hr_df["HR side, 1hr Roll Mean, Daily Tidal Window"], s=1, label='Herring River Near-Dike Levels')

# Using "Uncertainties" Module, Linear Regression
WF_HR_1hr_df_nona = WF_HR_1hr_df.dropna()
x_day = mdates.date2num(WF_HR_1hr_df_nona["datetime"])
y_day = WF_HR_1hr_df_nona["Daily Tidal Rolling Residuals"].values
n = len(y_day)

def f_line(x, a, b):
    return a * x + b

popt, pcov = curve_fit(f_line, x_day, y_day)

# retrieve parameter values
a = popt[0]
b = popt[1]
print('Optimal Values')
print('a: ' + str(a))
print('b: ' + str(b))

# compute r^2
r2 = 1.0-(sum((y_day-f_line(x_day,a,b))**2)/((n-1.0)*np.var(y_day,ddof=1)))
print('R^2: ' + str(r2))

# calculate parameter confidence interval
a,b = unc.correlated_values(popt, pcov)
print('Uncertainty')
print('a: ' + str(a))
print('b: ' + str(b))

# plot data
plt.scatter(x_day, y_day, s=1, color='g', label="Separation in Water Levels")

# calculate regression confidence interval
px_day = np.linspace(np.nanmin(x_day), np.nanmax(x_day), 100)
py_day = a*px_day+b
nom_day = unp.nominal_values(py_day)
std = unp.std_devs(py_day)

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

lpb, upb = predband(px_day, x_day, y_day, popt, f_line, conf=0.95)

# plot the regression
plt.plot(px_day, nom_day, c='black')

# uncertainty lines (95% confidence)
# plt.plot(px, nom_day - 1.96 * std, c='orange', label='95% Confidence Region')
# plt.plot(px, nom_day + 1.96 * std, c='orange')
# prediction band (95% confidence)
# plt.plot(px, lpb, 'k--',label='95% Prediction Band')
# plt.plot(px, upb, 'k--')
ylabel_delelev = 'Rolling Mean, Daily Tidal Window \n Water Surface Elevation [m NAVD88]'
DateAxisFmt(ylabel_delelev)
plt.legend(loc='lower right', bbox_to_anchor=(0.9,0), fontsize=22)

#%% Month Window

WF_HR_1hr_df["Monthly Rolling Residuals"] = WF_HR_1hr_df["Residuals"].rolling(window=730, min_periods=700, center=True).mean()

WF_HR_1hr_df["Ocean side, 1hr Roll Mean, Month Window"] = WF_HR_1hr_df["Gage height, m, Ocean side"].rolling(window=730, min_periods=700, center=True).mean()
WF_HR_1hr_df["HR side, 1hr Roll Mean, Month Window"] = WF_HR_1hr_df["Gage height, m, HR side"].rolling(window=730, min_periods=700, center=True).mean()

plt.figure()
plt.scatter(WF_HR_1hr_df["datetime"], WF_HR_1hr_df["Ocean side, 1hr Roll Mean, Month Window"], s=1, label='WF Harbor Near-Dike Levels')
plt.scatter(WF_HR_1hr_df["datetime"], WF_HR_1hr_df["HR side, 1hr Roll Mean, Month Window"], s=1, label='Herring River Near-Dike Levels')

# Get max difference, divide all differences by that max to find percent separation, plot percent separation, over time,
# fit line? find percent divergence per year or per second with error, apply to SLR rate to find HR rise rate.

# this should estimate the linear rate that the gap between the HR levels and WF levels is changing, percent per day
# sns.regplot(x=mdates.date2num(WF_HR_1hr_df["datetime"]), y="Fraction of Max Difference", data=WF_HR_1hr_df, color='g')

# Using "Uncertainties" Module, Linear Regression
WF_HR_1hr_df_nona = WF_HR_1hr_df.dropna()
x_month = mdates.date2num(WF_HR_1hr_df_nona["datetime"])
y_month = WF_HR_1hr_df_nona["Monthly Rolling Residuals"].values
n = len(y_month)

popt, pcov = curve_fit(f_line, x_month, y_month)

# retrieve parameter values
a = popt[0]
b = popt[1]
print('Optimal Values')
print('a: ' + str(a))
print('b: ' + str(b))

# compute r^2
r2 = 1.0-(sum((y_month-f_line(x_month,a,b))**2)/((n-1.0)*np.var(y_month,ddof=1)))
print('R^2: ' + str(r2))

# calculate parameter confidence interval
a,b = unc.correlated_values(popt, pcov)
print('Uncertainty_month')
print('a: ' + str(a))
print('b: ' + str(b))

# plot data
plt.scatter(x_month, y_month, s=1, color='g', label="Separation in Water Levels")

# calculate regression confidence interval
px_month = np.linspace(np.nanmin(x_month), np.nanmax(x_month), 100)
py_month = a*px_month+b
nom_month = unp.nominal_values(py_month)
std = unp.std_devs(py_month)

lpb, upb = predband(px_month, x_month, y_month, popt, f_line, conf=0.95)

# plot the regression
plt.plot(px_month, nom_month, c='black')

# uncertainty_month lines (95% confidence)
# plt.plot(px_month, nom_month - 1.96 * std, c='orange', label='95% Confidence Region')
# plt.plot(px_month, nom_month + 1.96 * std, c='orange')
# prediction band (95% confidence)
# plt.plot(px_month, lpb, 'k--',label='95% Prediction Band')
# plt.plot(px_month, upb, 'k--')
y_label_delelev = 'Rolling Mean, Monthly Window \n Water Surface Elevation [m NAVD88]'
DateAxisFmt(y_label_delelev)
plt.legend(loc='best', bbox_to_anchor=(0.43,0.57), fontsize=22)

#%% Annual Window

WF_HR_1hr_df["Annual Rolling Residuals"] = WF_HR_1hr_df["Residuals"].rolling(window=8766, min_periods=8000, center=True).mean()

WF_HR_1hr_df["Ocean side, 1hr Roll Mean, Annual Window"] = WF_HR_1hr_df["Gage height, m, Ocean side"].rolling(window=8766, min_periods=8000, center=True).mean()
WF_HR_1hr_df["HR side, 1hr Roll Mean, Annual Window"] = WF_HR_1hr_df["Gage height, m, HR side"].rolling(window=8766, min_periods=8000, center=True).mean()

plt.figure()
plt.scatter(WF_HR_1hr_df["datetime"], WF_HR_1hr_df["Ocean side, 1hr Roll Mean, Annual Window"], s=1, label='WF Harbor Near-Dike Levels')
plt.scatter(WF_HR_1hr_df["datetime"], WF_HR_1hr_df["HR side, 1hr Roll Mean, Annual Window"], s=1, label='Herring River Near-Dike Levels')

# Get max difference, divide all differences by that max to find percent separation, plot percent separation, over time,
# fit line? find percent divergence per year or per second with error, apply to SLR rate to find HR rise rate.

# this should estimate the linear rate that the gap between the HR levels and WF levels is changing, percent per day
# sns.regplot(x=mdates.date2num(WF_HR_1hr_df["datetime"]), y="Fraction of Max Difference", data=WF_HR_1hr_df, color='g')

# Using "Uncertainties" Module, Linear Regression
WF_HR_1hr_df_nona = WF_HR_1hr_df.dropna()
x_year = mdates.date2num(WF_HR_1hr_df_nona["datetime"])
y_year = WF_HR_1hr_df_nona["Annual Rolling Residuals"].values
n = len(y_year)

popt, pcov = curve_fit(f_line, x_year, y_year)

# retrieve parameter values
a = popt[0]
b = popt[1]
print('Optimal Values')
print('a: ' + str(a))
print('b: ' + str(b))

# compute r^2
r2 = 1.0-(sum((y_year-f_line(x_year,a,b))**2)/((n-1.0)*np.var(y_year,ddof=1)))
print('R^2: ' + str(r2))

# calculate parameter confidence interval
a,b = unc.correlated_values(popt, pcov)
print('Uncertainty_year')
print('a: ' + str(a))
print('b: ' + str(b))

print('Standard Deviation')
print('std y_year: ' + str(np.nanstd(y_year)))

# plot data
plt.scatter(x_year, y_year, s=1, color='g', label="Separation in Water Levels")

# calculate regression confidence interval
px_year = np.linspace(np.nanmin(x_year), np.nanmax(x_year), 100)
py_year = a*px_year+b
nom_year = unp.nominal_values(py_year)
std = unp.std_devs(py_year)

lpb, upb = predband(px_year, x_year, y_year, popt, f_line, conf=0.95)

# plot the regression
plt.plot(px_year, nom_year, c='black', label='Separation Trend')

# uncertainty lines (95% confidence)
# plt.plot(px, nom_year - 1.96 * std, c='orange', label='95% Confidence Region')
# plt.plot(px, nom_year + 1.96 * std, c='orange')
# prediction band (95% confidence)
# plt.plot(px, lpb, 'k--',label='95% Prediction Band')
# plt.plot(px, upb, 'k--')
ylabel_delelev = 'Rolling Mean, Annual Window \n Water Surface Elevation [m NAVD88]'
DateAxisFmt(ylabel_delelev)
plt.legend(loc='lower right', bbox_to_anchor=(0.7,0.2), fontsize=22)

# Not perfect, but provides potential scenarios for resulting levels in Herring River

#%% Thesis Figure

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

# Tidal Day
ax1.scatter(WF_HR_1hr_df["datetime"], WF_HR_1hr_df["Ocean side, 1hr Roll Mean, Daily Tidal Window"], s=1, label='WF Harbor Near-Dike Levels')
ax1.scatter(WF_HR_1hr_df["datetime"], WF_HR_1hr_df["HR side, 1hr Roll Mean, Daily Tidal Window"], s=1, label='Herring River Near-Dike Levels')
ax1.scatter(x_day, y_day, s=1, color='g', label="Separation in Water Levels")
ax1.plot(px_day, nom_day, c='black')
ax1.text(0.72,0.2,"Tidal Day Window",transform=ax1.transAxes,fontsize=22)

# Month
ax2.scatter(WF_HR_1hr_df["datetime"], WF_HR_1hr_df["Ocean side, 1hr Roll Mean, Month Window"], s=1, label='WF Harbor Near-Dike Levels')
ax2.scatter(WF_HR_1hr_df["datetime"], WF_HR_1hr_df["HR side, 1hr Roll Mean, Month Window"], s=1, label='Herring River Near-Dike Levels')
ax2.scatter(x_month, y_month, s=1, color='g', label="Separation in Water Levels")
ax2.plot(px_month, nom_month, c='black')
ax2.text(0.77,0.4,"Month Window",transform=ax2.transAxes,fontsize=22)

# Year
ax3.scatter(WF_HR_1hr_df["datetime"], WF_HR_1hr_df["Ocean side, 1hr Roll Mean, Annual Window"], s=1, label='WF Harbor \n Near-Dike \n Levels')
ax3.scatter(WF_HR_1hr_df["datetime"], WF_HR_1hr_df["HR side, 1hr Roll Mean, Annual Window"], s=1, label='Herring \n River \n Near-Dike \n Levels')
ax3.scatter(x_year, y_year, s=1, color='g', label="Separation \n in Levels")
ax3.plot(px_year, nom_year, c='black', label='Separation \n Trend')
ax3.text(0.8,0.2,"Year Window",transform=ax3.transAxes,fontsize=22)

plt.tight_layout()

ax2.set_ylabel('Hourly Rolling Average of \n Water Surface Elevation [m NAVD88]', fontsize=22)
ax3.set_xlabel('Date [Year-Month]', fontsize=22, labelpad=15)

ax3.legend(bbox_to_anchor=(1.0, 3.0), loc='upper left', markerscale=6, fontsize=22)

loc = mdates.AutoDateLocator()
ax3.xaxis.set_major_locator(loc)
ax3.xaxis.set_minor_locator(loc)
ax3.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
ax1.tick_params(labelrotation=45)
ax2.tick_params(labelrotation=45)
ax3.tick_params(labelrotation=45)

plt.gcf().subplots_adjust(bottom=0.25)

# More data required to make a better estimate of the separation trend

#%% Resid Compare with Monthly Precipitation Data, Harwich

# Annual Window, Group by Month First
WF_HR_1hr_df_di = WF_HR_1hr_df.set_index(WF_HR_1hr_df['datetime'],drop=True)
WF_HR_mo_df_di = WF_HR_1hr_df_di.resample('MS').mean()
WF_HR_mo_df = WF_HR_mo_df_di.reset_index()

WF_HR_mo_df["Annual Rolling Residuals"] = WF_HR_mo_df["Residuals"].rolling(window=12, min_periods=12, center=True).mean()
WF_HR_mo_df["Ocean side, 1hr Roll Mean, Annual Window"] = WF_HR_mo_df["Gage height, m, Ocean side"].rolling(window=12, min_periods=12, center=True).mean()
WF_HR_mo_df["HR side, 1hr Roll Mean, Annual Window"] = WF_HR_mo_df["Gage height, m, HR side"].rolling(window=12, min_periods=12, center=True).mean()

plt.figure()
plt.scatter(WF_HR_mo_df["datetime"], WF_HR_mo_df["Ocean side, 1hr Roll Mean, Annual Window"], s=1, label='WF Harbor Near-Dike Levels')
plt.scatter(WF_HR_mo_df["datetime"], WF_HR_mo_df["HR side, 1hr Roll Mean, Annual Window"], s=1, label='Herring River Near-Dike Levels')

WF_HR_mo_df_nona = WF_HR_mo_df.dropna()

Harwich_mo_precip = pd.read_csv(os.path.join(data_dir, 'Precipitation Data', 'PrecipitationDataHarwich.csv'))
Harwich_mo_precip["datetime"] = pd.to_datetime(Harwich_mo_precip["datetime"])
Harwich_mo_precip["Annual Rolling Average"] = Harwich_mo_precip["Monthly Precipitation, m"].rolling(window=12, min_periods=12, center=True).mean()
Harwich_mo_precip.set_index("datetime", inplace=True)
Harwich_mo_precip_matching = Harwich_mo_precip["2018-03-01":Harwich_mo_precip["Annual Rolling Average"].last_valid_index()]
Harwich_mo_precip_matching.reset_index(inplace=True)

Harwich_precip_dike_resid = pd.merge(Harwich_mo_precip_matching, WF_HR_mo_df_nona)
Harwich_precip_dike_resid = Harwich_precip_dike_resid.filter(["datetime","Annual Rolling Average","Annual Rolling Residuals","Ocean side, 1hr Roll Mean, Annual Window","HR side, 1hr Roll Mean, Annual Window"], axis=1)

x_precip_resid = mdates.date2num(Harwich_precip_dike_resid["datetime"])

# Precipitation
y_precip = Harwich_precip_dike_resid["Annual Rolling Average"].values
n = len(y_year)

popt, pcov = curve_fit(f_line, x_precip_resid, y_precip)

# retrieve parameter values
a = popt[0]
b = popt[1]
print('Optimal Values')
print('a: ' + str(a), 'b: ' + str(b))

# compute r^2
r2 = 1.0-(sum((y_precip-f_line(x_precip_resid,a,b))**2)/((n-1.0)*np.var(y_precip,ddof=1)))
print('R^2: ' + str(r2))

# calculate parameter confidence interval
a,b = unc.correlated_values(popt, pcov)
print('Uncertainty_precip')
print('a: ' + str(a), 'b: ' + str(b))

# plot data
plt.scatter(x_precip_resid, y_precip, label="Precipitation")

# calculate regression confidence interval
px_precip_resid = np.linspace(np.nanmin(x_precip_resid), np.nanmax(x_precip_resid), 100)
py_precip = a*px_precip_resid+b
nom_precip = unp.nominal_values(py_precip)
std = unp.std_devs(py_precip)

lpb, upb = predband(px_precip_resid, x_precip_resid, y_precip, popt, f_line, conf=0.95)

# plot the regression
plt.plot(px_precip_resid, nom_precip, c='black', label='Precipitation Trend')

"""
Difference in Levels Around Dike
""" 
y_resid = Harwich_precip_dike_resid["Annual Rolling Residuals"].values
n = len(y_year)

popt, pcov = curve_fit(f_line, x_precip_resid, y_resid)

# retrieve parameter values
a = popt[0]
b = popt[1]
print('Optimal Values')
print('a: ' + str(a), 'b: ' + str(b))

# compute r^2
r2 = 1.0-(sum((y_resid-f_line(x_precip_resid,a,b))**2)/((n-1.0)*np.var(y_resid,ddof=1)))
print('R^2: ' + str(r2))

# calculate parameter confidence interval
a,b = unc.correlated_values(popt, pcov)
print('Uncertainty_resid')
print('a: ' + str(a), 'b: ' + str(b))

# plot data
plt.scatter(x_precip_resid, y_resid, label="Separation in Water Levels")

# calculate regression confidence interval
px_precip_resid = np.linspace(np.nanmin(x_precip_resid), np.nanmax(x_precip_resid), 100)
py_resid = a*px_precip_resid+b
nom_resid = unp.nominal_values(py_resid)
std = unp.std_devs(py_resid)

lpb, upb = predband(px_precip_resid, x_precip_resid, y_resid, popt, f_line, conf=0.95)

# plot the regression
plt.plot(px_precip_resid, nom_resid, c='black', linestyle='--', label='Separation Trend')

"""
WF Harbor Levels 
""" 
y_ocean = Harwich_precip_dike_resid["Ocean side, 1hr Roll Mean, Annual Window"].values
n = len(y_year)

popt, pcov = curve_fit(f_line, x_precip_resid, y_ocean)

# retrieve parameter values
a = popt[0]
b = popt[1]
print('Optimal Values')
print('a: ' + str(a), 'b: ' + str(b))

# compute r^2
r2 = 1.0-(sum((y_ocean-f_line(x_precip_resid,a,b))**2)/((n-1.0)*np.var(y_ocean,ddof=1)))
print('R^2: ' + str(r2))

# calculate parameter confidence interval
a,b = unc.correlated_values(popt, pcov)
print('Uncertainty_ocean')
print('a: ' + str(a), 'b: ' + str(b))

# plot data
plt.scatter(x_precip_resid, y_ocean, label="WF Harbor Near Dike Water Levels")

# calculate regression confidence interval
px_precip_resid = np.linspace(np.nanmin(x_precip_resid), np.nanmax(x_precip_resid), 100)
py_ocean = a*px_precip_resid+b
nom_ocean = unp.nominal_values(py_ocean)
std = unp.std_devs(py_ocean)

lpb, upb = predband(px_precip_resid, x_precip_resid, y_ocean, popt, f_line, conf=0.95)

# plot the regression
plt.plot(px_precip_resid, nom_ocean, c='black', linestyle='-.', label='WF Harbor Near Dike Water Levels Trend')
ylabel_delelev = 'Rolling Mean, Annual Window \n Water Level [m NAVD88], Precipitation [m]'
DateAxisFmt(ylabel_delelev)
plt.xlabel('Date [Year-Month]', fontsize=22, labelpad=15)
plt.legend(loc='lower right', bbox_to_anchor=(0.7,0.3), fontsize=22)

#%% Thesis Figure

fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':[1,3]})
fig.text(0.04,0.6,'Annual Window Rolling Averages', ha='center', va='center', rotation='vertical', fontsize=22)

# Tidal Day
ax1.scatter(x_precip_resid, y_precip, label='Monthly Means')
ax1.plot(px_precip_resid, nom_precip, c='black')
# ax1.text(0.72,0.2,"Tidal Day Window",transform=ax1.transAxes,fontsize=22)
ax1.legend(loc='lower right', bbox_to_anchor=(0.95,-0.05),fontsize=22)

# Month
ax2.scatter(x_precip_resid, y_resid, c='magenta', label="Separation in Near-Dike Water Levels, Monthly Means")
ax2.scatter(x_precip_resid, y_ocean, c='green', label="WF Harbor Near-Dike Levels, Monthly Means")
ax2.plot(px_precip_resid, nom_resid, c='black', linestyle='--', label='Separation Trend')
ax2.plot(px_precip_resid, nom_ocean, c='black', linestyle='-.', label='WF Harbor Trend')
# ax2.text(0.77,0.4,"Month Window",transform=ax2.transAxes,fontsize=22)
# ax2.text(0.8,0.2,"Year Window",transform=ax3.transAxes,fontsize=22)
ax2.legend(loc='center left', bbox_to_anchor=(0.05,0.5), fontsize=22)

ax1.set_ylabel('Precip. [m]', fontsize=22)
ax2.set_ylabel('Water Level [m NAVD88]', fontsize=22)

ax2.set_xlabel('Date [Year-Month]', fontsize=22, labelpad=15)

loc = mdates.AutoDateLocator()
ax2.xaxis.set_major_locator(loc)
ax2.xaxis.set_minor_locator(loc)
ax2.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
ax1.tick_params(labelrotation=45)
ax2.tick_params(labelrotation=45)

plt.gcf().subplots_adjust(bottom=0.25)

# More data required to make a better estimate of the separation trend

# Correlations
y_river = Harwich_precip_dike_resid["HR side, 1hr Roll Mean, Annual Window"].values
r_riverocean = np.corrcoef(y_river,y_ocean)[0,1]
r_riverprecip = np.corrcoef(y_river,y_precip)[0,1]
r_oceanprecip = np.corrcoef(y_ocean,y_precip)[0,1]
r_residocean = np.corrcoef(y_resid,y_ocean)[0,1]
r_residriver = np.corrcoef(y_resid,y_river)[0,1]
r_residprecip = np.corrcoef(y_resid,y_precip)[0,1]

#%% Slice to remove nan values and separate df columns, 30 min.

HR_all_slice_start_i = HR_all_resam30min_df["Gage height, m, Ocean side"].first_valid_index()
HR_all_slice_end_i = HR_all_resam30min_df["Gage height, m, HR side"].last_valid_index()

HR_all_resam30min_df_slice = HR_all_resam30min_df.iloc[HR_all_slice_start_i:HR_all_slice_end_i]

HR_all_resam30min_df_slice.to_csv(os.path.join(data_dir,"General Dike Data","HR_All_Data_Resampled_30min_9272017-11132019.csv"), index = False)

# Interpolate between points or change time delta?
# HR_all_resam30min_df_slice.interpolate(method='index', limit=5, inplace=True)

# Need to remove nan vals and reduce measurement frequency
nan_indices_oceanside = np.isnan(HR_all_resam30min_df_slice["Gage height, m, Ocean side"])
nan_indices_HRside = np.isnan(HR_all_resam30min_df_slice["Gage height, m, HR side"])
# These should be zero
print(len(nan_indices_HRside[nan_indices_oceanside==True]))
print(len(nan_indices_HRside[nan_indices_HRside==True]))

# Linear fits to half hour time series levels    
idx_oceanside_slice = np.isfinite(HR_all_resam30min_df_slice["datetime"]) & np.isfinite(HR_all_resam30min_df_slice["Gage height, m, Ocean side"])
idx_HRside_slice = np.isfinite(HR_all_resam30min_df_slice["datetime"]) & np.isfinite(HR_all_resam30min_df_slice["Gage height, m, HR side"])
z_oceanside_slice = np.polyfit(mdates.date2num(HR_all_resam30min_df_slice["datetime"].loc[idx_oceanside_slice]), HR_all_resam30min_df_slice["Gage height, m, Ocean side"].loc[idx_oceanside_slice], 1)
z_HRside_slice = np.polyfit(mdates.date2num(HR_all_resam30min_df_slice["datetime"].loc[idx_HRside_slice]), HR_all_resam30min_df_slice["Gage height, m, HR side"].loc[idx_HRside_slice], 1)
p_oceanside_slice = np.poly1d(z_oceanside_slice)
p_HRside_slice = np.poly1d(z_HRside_slice)
polyX_slice = np.linspace(mdates.date2num(HR_all_resam30min_df_slice["datetime"].loc[idx_oceanside_slice]).min(), mdates.date2num(HR_all_resam30min_df_slice["datetime"].loc[idx_oceanside_slice]).max(), 100)
    
# Max and min vals for tides, 30 min.
oceanside_levels = HR_all_resam30min_df_slice["Gage height, m, Ocean side"]
HRside_levels = HR_all_resam30min_df_slice["Gage height, m, HR side"]
CNRUS_levels = HR_all_resam30min_df_slice["CNR U/S Water Level, NAVD88"]
DogLeg_levels = HR_all_resam30min_df_slice["Dog Leg Water Level, NAVD88"]
HighToss_levels = HR_all_resam30min_df_slice["High Toss Water Level, NAVD88"]
dates = HR_all_resam30min_df_slice["datetime"]
oceanside_levels.reset_index(drop=True,inplace=True)
HRside_levels.reset_index(drop=True,inplace=True)
CNRUS_levels.reset_index(drop=True,inplace=True)
DogLeg_levels.reset_index(drop=True,inplace=True)
HighToss_levels.reset_index(drop=True,inplace=True)
dates.reset_index(drop=True,inplace=True)

#%% Max and min vals for tides, hourly

oceanside_levels_hrly = hourly_summary_onhr_df["Gage height, m, Ocean side"]
HRside_levels_hrly = hourly_summary_onhr_df["Gage height, m, HR side"]
CNRUS_levels_hrly = hourly_summary_onhr_df["CNR U/S Water Level, NAVD88"]
DogLeg_levels_hrly = hourly_summary_onhr_df["Dog Leg Water Level, NAVD88"]
HighToss_levels_hrly = hourly_summary_onhr_df["High Toss Water Level, NAVD88"]
dates_hrly = hourly_summary_onhr_df["datetime"]

#%% Max and Min Extractor Function

def MaxMinLevels(dates, levels):
    """
    Function for extracting the max and min levels of a time series
    of tidally affected water elevation data. Maximum and minimum
    levels must stay on their respective sides of the mean to be
    considered.

    Parameters
    ----------
    dates : Series
        Series object of pandas.core.series module
    levels : Series
        Series object of pandas.core.series module

    Returns
    -------
    min_dates : , Array of object
        ndarray object of numpy module
    min_levels : , Array of float64
        same shape as min_dates
    max_dates : , Array of object
        ndarray object of numpy module
    max_levels : , Array of float64
        same shape as max_dates

    """
    datestart_neg = 0
    datestart_pos = 0
    date_interval_neg = 0
    date_interval_pos = 0
    bin_start_neg = 0
    bin_start_pos = 0
    max_dates = []
    min_dates = []
    y_mins = []
    y_maxes = []
    for bin_index in range(len(dates)-1):
        elev_start = levels[bin_index]
        elev_end = levels[bin_index+1]
        trans_cond = (elev_start-np.nanmean(levels))*(elev_end-np.nanmean(levels)) # subtract the means for a good crossover point
        if (trans_cond<=0)&(elev_start<elev_end):
            datestart_pos = dates.iloc[bin_index]
            bin_start_pos = bin_index
            dateend_neg = dates.iloc[bin_index+1]
            if (datestart_neg!=0):
                date_interval_neg = (dateend_neg - datestart_neg).seconds # date interval in seconds
                if (date_interval_neg > 6000): # Make sure small fluctuations aren't being counted
                    temp_interval = levels.iloc[bin_start_neg:bin_index]
                    min_index = temp_interval.loc[temp_interval==np.nanmin(temp_interval)].index.values[0]
                    if (len(min_dates) == 0):
                        y_mins.append(np.nanmin(temp_interval))
                        min_dates.append(dates.iloc[min_index])
                    if (dates.iloc[min_index] != min_dates[-1]): # makes sure duplicates aren't being printed
                        y_mins.append(np.nanmin(temp_interval)) # duplicates are somehow the result of nans
                        min_dates.append(dates.iloc[min_index])
        if (trans_cond<=0)&(elev_start>elev_end):
            datestart_neg = dates.iloc[bin_index]
            bin_start_neg = bin_index
            dateend_pos = dates.iloc[bin_index+1]
            if (datestart_pos!=0):
                date_interval_pos = (dateend_pos - datestart_pos).seconds # date interval in seconds
                if (date_interval_pos > 6000): # Make sure small fluctuations aren't being counted
                    temp_interval = levels.iloc[bin_start_pos:bin_index]    
                    max_index = temp_interval.loc[temp_interval==np.nanmax(temp_interval)].index.values[0]    
                    if (len(max_dates) == 0):
                        y_maxes.append(np.nanmax(temp_interval))
                        max_dates.append(dates.iloc[max_index])
                    if (dates.iloc[max_index] != max_dates[-1]):    
                        y_maxes.append(np.nanmax(temp_interval)) # makes sure duplicates aren't being printed
                        max_dates.append(dates.iloc[max_index]) # duplicates are somehow the result of nans
    min_dates = np.array(min_dates)
    max_dates = np.array(max_dates)
    y_mins = np.array(y_mins)
    y_maxes = np.array(y_maxes)
    return min_dates, y_mins, max_dates, y_maxes

#%% Max and Mins, 30 min.

# Wellfleet Harbor
min_dates_ocean, y_oceanside_mins, max_dates_ocean, y_oceanside_maxes = MaxMinLevels(dates, oceanside_levels)
# HR Side
min_dates_HR, y_HRside_mins, max_dates_HR, y_HRside_maxes = MaxMinLevels(dates, HRside_levels)
# CNR U/S
min_dates_CNR, y_CNR_mins, max_dates_CNR, y_CNR_maxes = MaxMinLevels(dates, CNRUS_levels)
# Dog Leg
min_dates_DL, y_DL_mins, max_dates_DL, y_DL_maxes = MaxMinLevels(dates, DogLeg_levels)
# High Toss
min_dates_HT, y_HT_mins, max_dates_HT, y_HT_maxes = MaxMinLevels(dates, HighToss_levels)

#%% Max and Mins, hrly

min_dates_hrly_ocean, y_oceanside_hrly_mins, max_dates_hrly_ocean, y_oceanside_hrly_maxes = MaxMinLevels(dates_hrly, oceanside_levels_hrly)
min_dates_hrly_HR, y_HRside_hrly_mins, max_dates_hrly_HR, y_HRside_hrly_maxes = MaxMinLevels(dates_hrly, HRside_levels_hrly)
min_dates_hrly_CNR, y_CNR_hrly_mins, max_dates_hrly_CNR, y_CNR_hrly_maxes = MaxMinLevels(dates_hrly, CNRUS_levels_hrly)
min_dates_hrly_DL, y_DL_hrly_mins, max_dates_hrly_DL, y_DL_hrly_maxes = MaxMinLevels(dates_hrly, DogLeg_levels_hrly)
min_dates_hrly_HT, y_HT_hrly_mins, max_dates_hrly_HT, y_HT_hrly_maxes = MaxMinLevels(dates_hrly, HighToss_levels_hrly)

#%% Max and Min plots, hrly

plt.scatter(mdates.date2num(max_dates_hrly_ocean), y_oceanside_hrly_maxes)
plt.scatter(mdates.date2num(min_dates_hrly_ocean), y_oceanside_hrly_mins)
plt.scatter(mdates.date2num(max_dates_hrly_HR), y_HRside_hrly_maxes)
plt.scatter(mdates.date2num(min_dates_hrly_HR), y_HRside_hrly_mins)
plt.scatter(mdates.date2num(max_dates_hrly_CNR), y_CNR_hrly_maxes)
plt.scatter(mdates.date2num(min_dates_hrly_CNR), y_CNR_hrly_mins)
plt.scatter(mdates.date2num(max_dates_hrly_DL), y_DL_hrly_maxes)
plt.scatter(mdates.date2num(min_dates_hrly_DL), y_DL_hrly_mins)
plt.scatter(mdates.date2num(max_dates_hrly_HT), y_HT_hrly_maxes)
plt.scatter(mdates.date2num(min_dates_hrly_HT), y_HT_hrly_mins)
DateAxisFmt(ylabel_elev)

#%% Plot oceanside slice with max/min

ax = HR_all_resam30min_df_slice.plot.scatter(x="datetime", y="Gage height, m, Ocean side", color='LightGreen', label = 'Gage height, m , Ocean side, 30min')
plt.scatter(mdates.date2num(max_dates_ocean), y_oceanside_maxes)
plt.scatter(mdates.date2num(min_dates_ocean), y_oceanside_mins)

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_elev)

#%% Plot HRside slice with max/min

ax = HR_all_resam30min_df_slice.plot.scatter(x="datetime", y="Gage height, m, HR side", color='LightGreen', label = 'Gage height, m , HR side, 30min')
plt.scatter(mdates.date2num(max_dates_HR), y_HRside_maxes)
plt.scatter(mdates.date2num(min_dates_HR), y_HRside_mins)

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_elev)

#%% Plot CNR U/S slice with max/min

ax = HR_all_resam30min_df_slice.plot.scatter(x="datetime", y="CNR U/S Water Level, NAVD88", color='LightGreen', label = 'CNR U/S Water Level, NAVD88, 30min')
plt.scatter(mdates.date2num(max_dates_CNR), y_CNR_maxes)
plt.scatter(mdates.date2num(min_dates_CNR), y_CNR_mins)

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_elev)

#%% Plot Dog Leg slice with max/min

ax = HR_all_resam30min_df_slice.plot.scatter(x="datetime", y="Dog Leg Water Level, NAVD88", color='LightGreen', label = 'Dog Leg Water Level, NAVD88, 30min')
plt.scatter(mdates.date2num(max_dates_DL), y_DL_maxes)
plt.scatter(mdates.date2num(min_dates_DL), y_DL_mins)

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_elev)

#%% Plot High Toss slice with max/min

ax = HR_all_resam30min_df_slice.plot.scatter(x="datetime", y="High Toss Water Level, NAVD88", color='LightGreen', label = 'High Toss Water Level, NAVD88, 30min')
plt.scatter(mdates.date2num(max_dates_HT), y_HT_maxes)
plt.scatter(mdates.date2num(min_dates_HT), y_HT_mins)

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_elev)

#%% Combine levels and associated dates

# 30 minute levels
# Oceanside
oceanside_maxlevels_arr = np.vstack([max_dates_ocean,y_oceanside_maxes]).T
oceanside_maxlevels_df = pd.DataFrame(oceanside_maxlevels_arr, columns=["datetime", "Max Gage height, m, Ocean side"])
oceanside_maxlevels_df["datetime"] = pd.to_datetime(oceanside_maxlevels_df["datetime"])
oceanside_maxlevels_df["Max Gage height, m, Ocean side"] = pd.to_numeric(oceanside_maxlevels_df["Max Gage height, m, Ocean side"])

oceanside_minlevels_arr = np.vstack([min_dates_ocean,y_oceanside_mins]).T
oceanside_minlevels_df = pd.DataFrame(oceanside_minlevels_arr, columns=["datetime", "Min Gage height, m, Ocean side"])
oceanside_minlevels_df["datetime"] = pd.to_datetime(oceanside_minlevels_df["datetime"])
oceanside_minlevels_df["Min Gage height, m, Ocean side"] = pd.to_numeric(oceanside_minlevels_df["Min Gage height, m, Ocean side"])

# HRside
HRside_maxlevels_arr = np.vstack([max_dates_HR,y_HRside_maxes]).T
HRside_maxlevels_df = pd.DataFrame(HRside_maxlevels_arr, columns=["datetime","Max Gage height, m, HR side"])
HRside_maxlevels_df["datetime"] = pd.to_datetime(HRside_maxlevels_df["datetime"])
HRside_maxlevels_df["Max Gage height, m, HR side"] = pd.to_numeric(HRside_maxlevels_df["Max Gage height, m, HR side"])

HRside_minlevels_arr = np.vstack([min_dates_HR,y_HRside_mins]).T
HRside_minlevels_df = pd.DataFrame(HRside_minlevels_arr, columns=["datetime","Min Gage height, m, HR side"])
HRside_minlevels_df["datetime"] = pd.to_datetime(HRside_minlevels_df["datetime"])
HRside_minlevels_df["Min Gage height, m, HR side"] = pd.to_numeric(HRside_minlevels_df["Min Gage height, m, HR side"])

# CNR U/S
CNR_maxlevels_arr = np.vstack([max_dates_CNR,y_CNR_maxes]).T
CNR_maxlevels_df = pd.DataFrame(CNR_maxlevels_arr, columns=["datetime","Max Gage height, m, CNR U/S"])
CNR_maxlevels_df["datetime"] = pd.to_datetime(CNR_maxlevels_df["datetime"])
CNR_maxlevels_df["Max Gage height, m, CNR U/S"] = pd.to_numeric(CNR_maxlevels_df["Max Gage height, m, CNR U/S"])

CNR_minlevels_arr = np.vstack([min_dates_CNR,y_CNR_mins]).T
CNR_minlevels_df = pd.DataFrame(CNR_minlevels_arr, columns=["datetime","Min Gage height, m, CNR U/S"])
CNR_minlevels_df["datetime"] = pd.to_datetime(CNR_minlevels_df["datetime"])
CNR_minlevels_df["Min Gage height, m, CNR U/S"] = pd.to_numeric(CNR_minlevels_df["Min Gage height, m, CNR U/S"])

# Dog Leg
DL_maxlevels_arr = np.vstack([max_dates_DL,y_DL_maxes]).T
DL_maxlevels_df = pd.DataFrame(DL_maxlevels_arr, columns=["datetime","Max Gage height, m, Dog Leg"])
DL_maxlevels_df["datetime"] = pd.to_datetime(DL_maxlevels_df["datetime"])
DL_maxlevels_df["Max Gage height, m, Dog Leg"] = pd.to_numeric(DL_maxlevels_df["Max Gage height, m, Dog Leg"])

DL_minlevels_arr = np.vstack([min_dates_DL,y_DL_mins]).T
DL_minlevels_df = pd.DataFrame(DL_minlevels_arr, columns=["datetime","Min Gage height, m, Dog Leg"])
DL_minlevels_df["datetime"] = pd.to_datetime(DL_minlevels_df["datetime"])
DL_minlevels_df["Min Gage height, m, Dog Leg"] = pd.to_numeric(DL_minlevels_df["Min Gage height, m, Dog Leg"])

# High Toss
HT_maxlevels_arr = np.vstack([max_dates_HT,y_HT_maxes]).T
HT_maxlevels_df = pd.DataFrame(HT_maxlevels_arr, columns=["datetime","Max Gage height, m, High Toss"])
HT_maxlevels_df["datetime"] = pd.to_datetime(HT_maxlevels_df["datetime"])
HT_maxlevels_df["Max Gage height, m, High Toss"] = pd.to_numeric(HT_maxlevels_df["Max Gage height, m, High Toss"])

HT_minlevels_arr = np.vstack([min_dates_HT,y_HT_mins]).T
HT_minlevels_df = pd.DataFrame(HT_minlevels_arr, columns=["datetime","Min Gage height, m, High Toss"])
HT_minlevels_df["datetime"] = pd.to_datetime(HT_minlevels_df["datetime"])
HT_minlevels_df["Min Gage height, m, High Toss"] = pd.to_numeric(HT_minlevels_df["Min Gage height, m, High Toss"])

# Hourly levels
# Oceanside
oceanside_hrly_maxlevels_arr = np.vstack([max_dates_hrly_ocean,y_oceanside_hrly_maxes]).T
oceanside_hrly_maxlevels_df = pd.DataFrame(oceanside_hrly_maxlevels_arr, columns=["datetime", "Max Gage height, m, Ocean side"])
oceanside_hrly_maxlevels_df["datetime"] = pd.to_datetime(oceanside_hrly_maxlevels_df["datetime"])

oceanside_hrly_minlevels_arr = np.vstack([min_dates_hrly_ocean,y_oceanside_hrly_mins]).T
oceanside_hrly_minlevels_df = pd.DataFrame(oceanside_hrly_minlevels_arr, columns=["datetime", "Min Gage height, m, Ocean side"])
oceanside_hrly_minlevels_df["datetime"] = pd.to_datetime(oceanside_hrly_minlevels_df["datetime"])

# HRside_hrly
HRside_hrly_maxlevels_arr = np.vstack([max_dates_hrly_HR,y_HRside_hrly_maxes]).T
HRside_hrly_maxlevels_df = pd.DataFrame(HRside_hrly_maxlevels_arr, columns=["datetime","Max Gage height, m, HR side"])
HRside_hrly_maxlevels_df["datetime"] = pd.to_datetime(HRside_hrly_maxlevels_df["datetime"])

HRside_hrly_minlevels_arr = np.vstack([min_dates_hrly_HR,y_HRside_hrly_mins]).T
HRside_hrly_minlevels_df = pd.DataFrame(HRside_hrly_minlevels_arr, columns=["datetime","Min Gage height, m, HR side"])
HRside_hrly_minlevels_df["datetime"] = pd.to_datetime(HRside_hrly_minlevels_df["datetime"])

# CNR_hrly U/S
CNR_hrly_maxlevels_arr = np.vstack([max_dates_hrly_CNR,y_CNR_hrly_maxes]).T
CNR_hrly_maxlevels_df = pd.DataFrame(CNR_hrly_maxlevels_arr, columns=["datetime","Max Gage height, m, CNR U/S"])
CNR_hrly_maxlevels_df["datetime"] = pd.to_datetime(CNR_hrly_maxlevels_df["datetime"])

CNR_hrly_minlevels_arr = np.vstack([min_dates_hrly_CNR,y_CNR_hrly_mins]).T
CNR_hrly_minlevels_df = pd.DataFrame(CNR_hrly_minlevels_arr, columns=["datetime","Min Gage height, m, CNR U/S"])
CNR_hrly_minlevels_df["datetime"] = pd.to_datetime(CNR_hrly_minlevels_df["datetime"])

# Dog Leg
DL_hrly_maxlevels_arr = np.vstack([max_dates_hrly_DL,y_DL_hrly_maxes]).T
DL_hrly_maxlevels_df = pd.DataFrame(DL_hrly_maxlevels_arr, columns=["datetime","Max Gage height, m, Dog Leg"])
DL_hrly_maxlevels_df["datetime"] = pd.to_datetime(DL_hrly_maxlevels_df["datetime"])

DL_hrly_minlevels_arr = np.vstack([min_dates_hrly_DL,y_DL_hrly_mins]).T
DL_hrly_minlevels_df = pd.DataFrame(DL_hrly_minlevels_arr, columns=["datetime","Min Gage height, m, Dog Leg"])
DL_hrly_minlevels_df["datetime"] = pd.to_datetime(DL_hrly_minlevels_df["datetime"])

# High Toss
HT_hrly_maxlevels_arr = np.vstack([max_dates_hrly_HT,y_HT_hrly_maxes]).T
HT_hrly_maxlevels_df = pd.DataFrame(HT_hrly_maxlevels_arr, columns=["datetime","Max Gage height, m, High Toss"])
HT_hrly_maxlevels_df["datetime"] = pd.to_datetime(HT_hrly_maxlevels_df["datetime"])

HT_hrly_minlevels_arr = np.vstack([min_dates_hrly_HT,y_HT_hrly_mins]).T
HT_hrly_minlevels_df = pd.DataFrame(HT_hrly_minlevels_arr, columns=["datetime","Min Gage height, m, High Toss"])
HT_hrly_minlevels_df["datetime"] = pd.to_datetime(HT_hrly_minlevels_df["datetime"])

#%% Merge Max and Min Dataframes

max_frames_hrly = [oceanside_hrly_maxlevels_df, HRside_hrly_maxlevels_df, CNR_hrly_maxlevels_df, DL_hrly_maxlevels_df, HT_hrly_maxlevels_df]
min_frames_hrly = [oceanside_hrly_minlevels_df, HRside_hrly_minlevels_df, CNR_hrly_minlevels_df, DL_hrly_minlevels_df, HT_hrly_minlevels_df]
HR_hrly_max_df = reduce(lambda left, right: pd.merge_ordered(left, right, on=["datetime"]), max_frames_hrly)
HR_hrly_min_df = reduce(lambda left, right: pd.merge_ordered(left, right, on=["datetime"]), min_frames_hrly)  

#%% Save
HR_hrly_max_df.to_csv(os.path.join(data_dir, 'General Dike Data', 'HR_All_Data_Resampled_HourlyMaxes_8272017-1212020.csv'), index = False)
HR_hrly_min_df.to_csv(os.path.join(data_dir, 'General Dike Data', 'HR_All_Data_Resampled_HourlyMins_8272017-1212020.csv'), index = False)

#%% Load
HR_hrly_max_df = pd.read_csv(os.path.join(data_dir, 'General Dike Data', 'HR_All_Data_Resampled_HourlyMaxes_8272017-1212020.csv'))
HR_hrly_min_df = pd.read_csv(os.path.join(data_dir, 'General Dike Data', 'HR_All_Data_Resampled_HourlyMins_8272017-1212020.csv'))

#%% Function for determining maxes reached in the HR resulting (res) from the maxes in the harbor

def ResMax(max_lev_upstream, oceanside_maxes, datetime_maxes):
    """
    Function for determing maximum levels reached upstream of a dike resulting
    from maximum levels outside of the dike in each tidal cycle.

    Parameters
    ----------
    max_lev_upstream : , Series
        Series object of pandas.core.series module, single column from 
        datetime ordered dataframe of oceanside and river gage max levels
    oceanside_maxes : , Series
        Series object of pandas.core.series module, single column from 
        datetime ordered dataframe of oceanside and river gage max levels
    datetime_maxes : , Series
        Series object of pandas.core.series module, single column from 
        datetime ordered dataframe of oceanside and river gage max levels

    Returns
    -------
    deltaT_maxes : Array of object
        ndarray object of numpy module, the elapsed time from oceanside peaks
        to peaks at gage inside the dike
    y_res_maxes : Array of float64
        the maximum levels at the gage inside the dike, the same size as the
        oceanside max levels (possibly including nans)

    """
    y_res_maxes = []
    deltaT_maxes = []
    for row in range(len(oceanside_maxes)-1):
        peak_lag = (datetime_maxes.iloc[row+1]-datetime_maxes.iloc[row]).seconds/3600
        if ~np.isnan(oceanside_maxes.iloc[row]) & (peak_lag>=2.0) & (peak_lag<=4.0):
            y_res_maxes.append(max_lev_upstream.iloc[row+1])
            deltaT_maxes.append(datetime_maxes.iloc[row+1]-datetime_maxes.iloc[row])
        elif ~np.isnan(oceanside_maxes.iloc[row]): # if peak lag is more than 4 hours or less than 1, erroneous data
            y_res_maxes.append(np.nan)
            deltaT_maxes.append(np.nan)
    # if last value in oceanside array is not nan, append another nan on the return arrays
    if (datetime_maxes.iloc[row+1]==datetime_maxes.iloc[-1]) & ~np.isnan(oceanside_maxes.iloc[row+1]): 
        y_res_maxes.append(np.nan)
        deltaT_maxes.append(np.nan)
    y_res_maxes = np.array(y_res_maxes)
    deltaT_maxes = np.array(deltaT_maxes)
    return deltaT_maxes, y_res_maxes

#%% Compare max values on ocean side with levels at same time and resulting maxes on HR side (30 min interpolated)

HRside_alllevels_df = pd.concat([dates, HRside_levels], axis=1)
CNRUS_alllevels_df = pd.concat([dates, CNRUS_levels], axis=1)
DogLeg_alllevels_df = pd.concat([dates, DogLeg_levels], axis=1)
HighToss_alllevels_df = pd.concat([dates, HighToss_levels], axis=1)

# Dataframe of Oceanside maxes and HR levels at same time.
oceanside_max_HRsametime_df = pd.merge(oceanside_maxlevels_df, HRside_alllevels_df)
oceanside_max_CNRsametime_df = pd.merge(oceanside_maxlevels_df, CNRUS_alllevels_df)
oceanside_max_DLsametime_df = pd.merge(oceanside_maxlevels_df, DogLeg_alllevels_df)
oceanside_max_HTsametime_df = pd.merge(oceanside_maxlevels_df, HighToss_alllevels_df)

# Dataframe of Oceanside mins and HR levels at same time.
oceanside_min_HRsametime_df = pd.merge(oceanside_minlevels_df, HRside_alllevels_df)
oceanside_min_CNRsametime_df = pd.merge(oceanside_minlevels_df, CNRUS_alllevels_df)
oceanside_min_DLsametime_df = pd.merge(oceanside_minlevels_df, DogLeg_alllevels_df)
oceanside_min_HTsametime_df = pd.merge(oceanside_minlevels_df, HighToss_alllevels_df)

# Combine max level arrays, keeping and ordering dates.

# HR Side of Dike
maxlevels_oceanHR_ordered_df = pd.merge_ordered(oceanside_maxlevels_df, HRside_maxlevels_df, on="datetime")
oceanHR_maxes = maxlevels_oceanHR_ordered_df["Max Gage height, m, Ocean side"]
HRside_maxes = maxlevels_oceanHR_ordered_df["Max Gage height, m, HR side"]
datetime_oceanHR_maxes = maxlevels_oceanHR_ordered_df["datetime"]

# CNR U/S
maxlevels_oceanCNR_ordered_df = pd.merge_ordered(oceanside_maxlevels_df, CNR_maxlevels_df, on="datetime")
oceanCNR_maxes = maxlevels_oceanCNR_ordered_df["Max Gage height, m, Ocean side"]
CNRUS_maxes = maxlevels_oceanCNR_ordered_df["Max Gage height, m, CNR U/S"]
datetime_oceanCNR_maxes = maxlevels_oceanCNR_ordered_df["datetime"]

# Dog Leg
maxlevels_oceanDL_ordered_df = pd.merge_ordered(oceanside_maxlevels_df, DL_maxlevels_df, on="datetime")
oceanDL_maxes = maxlevels_oceanDL_ordered_df["Max Gage height, m, Ocean side"]
DogLeg_maxes = maxlevels_oceanDL_ordered_df["Max Gage height, m, Dog Leg"]
datetime_oceanDL_maxes = maxlevels_oceanDL_ordered_df["datetime"]

# High Toss
maxlevels_oceanHT_ordered_df = pd.merge_ordered(oceanside_maxlevels_df, HT_maxlevels_df, on="datetime")
oceanHT_maxes = maxlevels_oceanHT_ordered_df["Max Gage height, m, Ocean side"]
HighToss_maxes = maxlevels_oceanHT_ordered_df["Max Gage height, m, High Toss"]
datetime_oceanHT_maxes = maxlevels_oceanHT_ordered_df["datetime"]

# Get the maxes reached in the HR resulting (res) from the maxes in the harbor
deltaT_HRmaxes, y_HRside_res_maxes = ResMax(HRside_maxes, oceanHR_maxes, datetime_oceanHR_maxes)
deltaT_CNRmaxes, y_CNR_res_maxes = ResMax(CNRUS_maxes, oceanCNR_maxes, datetime_oceanCNR_maxes)
deltaT_DLmaxes, y_DL_res_maxes = ResMax(DogLeg_maxes, oceanDL_maxes, datetime_oceanDL_maxes)
deltaT_HTmaxes, y_HT_res_maxes = ResMax(HighToss_maxes, oceanHT_maxes, datetime_oceanHT_maxes)

# Combine datetimes of maxima in harbor with resulting maxima in river, merge with levels at the same time

# HR Side of Dike
HRside_resmax_arr = np.vstack([max_dates_ocean, y_HRside_res_maxes]).T
HRside_resmax_df = pd.DataFrame(HRside_resmax_arr, columns=["datetime","Resulting Max Gage height, m, HR side"])
HRside_resmax_df["datetime"] = pd.to_datetime(HRside_resmax_df["datetime"])
HRside_resmax_df["Resulting Max Gage height, m, HR side"] = pd.to_numeric(HRside_resmax_df["Resulting Max Gage height, m, HR side"])
oceanside_max_HR_match_res_df = pd.merge(oceanside_max_HRsametime_df, HRside_resmax_df)

# CNR U/S
CNRUS_resmax_arr = np.vstack([max_dates_ocean, y_CNR_res_maxes]).T
CNRUS_resmax_df = pd.DataFrame(CNRUS_resmax_arr, columns=["datetime","Resulting Max Gage height, m, CNR U/S"])
CNRUS_resmax_df["datetime"] = pd.to_datetime(CNRUS_resmax_df["datetime"])
CNRUS_resmax_df["Resulting Max Gage height, m, CNR U/S"] = pd.to_numeric(CNRUS_resmax_df["Resulting Max Gage height, m, CNR U/S"])
oceanside_max_CNR_match_res_df = pd.merge(oceanside_max_CNRsametime_df, CNRUS_resmax_df)

# Dog Leg
DogLeg_resmax_arr = np.vstack([max_dates_ocean, y_DL_res_maxes]).T
DogLeg_resmax_df = pd.DataFrame(DogLeg_resmax_arr, columns=["datetime","Resulting Max Gage height, m, Dog Leg"])
DogLeg_resmax_df["datetime"] = pd.to_datetime(DogLeg_resmax_df["datetime"])
DogLeg_resmax_df["Resulting Max Gage height, m, Dog Leg"] = pd.to_numeric(DogLeg_resmax_df["Resulting Max Gage height, m, Dog Leg"])
oceanside_max_DL_match_res_df = pd.merge(oceanside_max_DLsametime_df, DogLeg_resmax_df)

# High Toss
HighToss_resmax_arr = np.vstack([max_dates_ocean, y_HT_res_maxes]).T
HighToss_resmax_df = pd.DataFrame(HighToss_resmax_arr, columns=["datetime","Resulting Max Gage height, m, High Toss"])
HighToss_resmax_df["datetime"] = pd.to_datetime(HighToss_resmax_df["datetime"])
HighToss_resmax_df["Resulting Max Gage height, m, High Toss"] = pd.to_numeric(HighToss_resmax_df["Resulting Max Gage height, m, High Toss"])
oceanside_max_HT_match_res_df = pd.merge(oceanside_max_HTsametime_df, HighToss_resmax_df)

#%% Plot of Ocean side vs. HR side

datapoints_tidalcycle = int(89400/60/30/2)
HRvOcean_test = HR_all_resam30min_df_slice.iloc[0:datapoints_tidalcycle]

ax = HR_all_resam_df.plot.scatter(x="Gage height, m, Ocean side", y="Gage height, m, HR side", color='Black')

# oceanside_max_HRsametime_df.plot.scatter(x="Max Gage height, m, Ocean side", y="Gage height, m, HR side", color='LightGreen', label = 'Max Gage height, m, Ocean side', ax=ax)
# oceanside_min_HRsametime_df.plot.scatter(x="Min Gage height, m, Ocean side", y="Gage height, m, HR side", color='LightBlue', label = 'Min Gage height, m, Ocean side', ax=ax)

# First cycle
# HRvOcean_test.plot.scatter(x='Gage height, m, Ocean side', y='Gage height, m, HR side', color='Red', ax=ax)

ax.set_xlabel('Gage height, m, Ocean side')
ax.set_ylabel('Gage height, m, HR side')

# For plotting the seasons separately
# HR_all_resam_df['Season, Oct-Apr=0, May-Sept=1'] = [0 if x.month in [10,11,12,1,2,3,4] else 1 for x in HR_all_resam_df["datetime"]]

oceanside_max_HRsametime_df['Season, Oct-Apr=0, May-Sept=1'] = [0 if x.month in [10,11,12,1,2,3,4] else 1 for x in oceanside_max_HRsametime_df["datetime"]]
oceanside_min_HRsametime_df['Season, Oct-Apr=0, May-Sept=1'] = [0 if x.month in [10,11,12,1,2,3,4] else 1 for x in oceanside_min_HRsametime_df["datetime"]]

hue = 'Season, Oct-Apr=0, May-Sept=1'
style = 'Season, Oct-Apr=0, May-Sept=1'
ax = sns.scatterplot(x="Max Gage height, m, Ocean side", y="Gage height, m, HR side", hue=hue, style=style, data=oceanside_max_HRsametime_df)
ax = sns.scatterplot(x="Min Gage height, m, Ocean side", y="Gage height, m, HR side", hue=hue, style=style, data=oceanside_min_HRsametime_df)

# legend adjustment
custom = [Line2D([], [], marker='o', color='b', linestyle='None'),
          Line2D([], [], marker='x', color='orange', linestyle='None')]

plt.legend(custom, ['Oct-Apr', 'May-Sept'], loc='lower right', bbox_to_anchor=(0.95,0.15), fontsize=22)

# Need to figure out issues with the colorbar.
# g = sns.jointplot("Gage height, m, Ocean side", "Gage height, m, HR side", data=HR_all_resam_df, kind='kde', space=0, color='g')

# idx_oceanHR_dates = np.isfinite(HR_all_resam_df["Gage height, m, Ocean side"]) & np.isfinite(HR_all_resam_df["Gage height, m, HR side"])
# ax = sns.kdeplot(HR_all_resam_df["Gage height, m, Ocean side"][idx_oceanHR_dates], HR_all_resam_df["Gage height, m, HR side"][idx_oceanHR_dates], cmap="Blues", shade=True, shade_lowest=False, cbar=True)

#%% Lag Times, HR side

deltaT_HRmaxes_df = pd.DataFrame(deltaT_HRmaxes, columns=["delta datetime"])
deltaT_HRmaxes_df["delta datetime"] = deltaT_HRmaxes_df["delta datetime"].dt.seconds/3600
mean_HRlag_time = np.nanmean(deltaT_HRmaxes_df["delta datetime"])
stdev_HRlag_time = np.std(deltaT_HRmaxes_df["delta datetime"])
print("Mean lag time oceanside max to HRside max =", int(round(mean_HRlag_time)), "hours and", int(round(abs(mean_HRlag_time-round(mean_HRlag_time))*60)), "minutes.")
# This makes sense, as the time from the peak to the mean level outside is 1/4 tidal cycle, or 3 hrs 13 minutes.
# The head in the HR is increasing fastest when the ocean is at its peak, and maxes when ocean hits mean.
print("Standard Deviation of lag time oceanside max to HRside max =", int(round(abs(stdev_HRlag_time-round(stdev_HRlag_time))*60)), "minutes.")
plt.figure()
plt.scatter(mdates.date2num(max_dates_ocean), deltaT_HRmaxes_df["delta datetime"], label="Time from Oceanside Peak to HRside Peak")

# Show X-axis major tick marks as dates
ylabel_time_hrs = 'Elapsed Time [hours]'
DateAxisFmt(ylabel_time_hrs)
plt.legend(loc='lower center', bbox_to_anchor = (0.5,0.55), fontsize=18)

#%% Lag Times, CNR U/S

deltaT_CNRmaxes_df = pd.DataFrame(deltaT_CNRmaxes, columns=["delta datetime"])
deltaT_CNRmaxes_df["delta datetime"] = deltaT_CNRmaxes_df["delta datetime"].dt.seconds/3600
mean_CNRlag_time = np.nanmean(deltaT_CNRmaxes_df["delta datetime"])
stdev_CNRlag_time = np.std(deltaT_CNRmaxes_df["delta datetime"])
print("Mean lag time oceanside max to CNR U/S max =", int(round(mean_CNRlag_time)), "hours and", int(round(abs(mean_CNRlag_time-round(mean_CNRlag_time))*60)), "minutes.")
# The head in the HR is increasing fastest when the ocean is at its peak, and maxes when ocean hits mean.
print("Standard Deviation of lag time oceanside max to CNR U/S max =", int(round(abs(stdev_CNRlag_time-round(stdev_CNRlag_time))*60)), "minutes.")
plt.figure()
plt.scatter(mdates.date2num(max_dates_ocean), deltaT_CNRmaxes_df["delta datetime"], label="Time from Oceanside Peak to CNR U/S Peak")

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_time_hrs)
plt.legend(loc='lower center', bbox_to_anchor = (0.5,0.55), fontsize=18)

#%% Lag Times, Dog Leg

deltaT_DLmaxes_df = pd.DataFrame(deltaT_DLmaxes, columns=["delta datetime"])
deltaT_DLmaxes_df["delta datetime"] = deltaT_DLmaxes_df["delta datetime"].dt.seconds/3600
mean_DLlag_time = np.nanmean(deltaT_DLmaxes_df["delta datetime"])
stdev_DLlag_time = np.std(deltaT_DLmaxes_df["delta datetime"])
print("Mean lag time oceanside max to Dog Leg max =", int(round(mean_DLlag_time)), "hours and", int(round(abs(mean_DLlag_time-round(mean_DLlag_time))*60)), "minutes.")
# The head in the HR is increasing fastest when the ocean is at its peak, and maxes when ocean hits mean.
print("Standard Deviation of lag time oceanside max to Dog Leg max =", int(round(abs(stdev_DLlag_time-round(stdev_DLlag_time))*60)), "minutes.")
plt.figure()
plt.scatter(mdates.date2num(max_dates_ocean), deltaT_DLmaxes_df["delta datetime"], label="Time from Oceanside Peak to Dog Leg Peak")

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_time_hrs)
plt.legend(loc='lower center', bbox_to_anchor = (0.5,0.55), fontsize=18)

#%% Lag Times, High Toss

deltaT_HTmaxes_df = pd.DataFrame(deltaT_HTmaxes, columns=["delta datetime"])
deltaT_HTmaxes_df["delta datetime"] = deltaT_HTmaxes_df["delta datetime"].dt.seconds/3600
mean_HTlag_time = np.nanmean(deltaT_HTmaxes_df["delta datetime"])
stdev_HTlag_time = np.std(deltaT_HTmaxes_df["delta datetime"])
print("Mean lag time oceanside max to High Toss max =", int(round(mean_HTlag_time)), "hours and", int(round(abs(mean_HTlag_time-round(mean_HTlag_time))*60)), "minutes.")
# The head in the HR is increasing fastest when the ocean is at its peak, and maxes when ocean hits mean.
print("Standard Deviation of lag time oceanside max to High Toss max =", int(round(abs(stdev_HTlag_time-round(stdev_HTlag_time))*60)), "minutes.")
plt.figure()
plt.scatter(mdates.date2num(max_dates_ocean), deltaT_HTmaxes_df["delta datetime"], label="Time from Oceanside Peak to High Toss Peak")

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_time_hrs)
plt.legend(loc='lower center', bbox_to_anchor = (0.5,0.55), fontsize=18)

# Mean wave travel time from CNR U/S to High Toss is 20 minutes.

#%% Time Series Plot of Max Oceanside, Matching HR Levels, and Resultant HR Maxes

ax = oceanside_max_HR_match_res_df.plot.scatter(x="datetime", y="Max Gage height, m, Ocean side", color='LightBlue', label = 'Max Gage height, m , Ocean side')
oceanside_max_HR_match_res_df.plot.scatter(x="datetime", y="Gage height, m, HR side", color='LightGreen', label = 'Gage height, m , HR side', ax=ax)
oceanside_max_HR_match_res_df.plot.scatter(x="datetime", y="Resulting Max Gage height, m, HR side", color='Orange', label = 'Resulting Max Gage height, m , HR side', ax=ax)

HRlagrise_datenum = mdates.date2num(oceanside_max_HR_match_res_df["datetime"])
idx_oceanside_dates = np.isfinite(HRlagrise_datenum) & np.isfinite(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"])
idx_HRmatch_dates = np.isfinite(HRlagrise_datenum) & np.isfinite(oceanside_max_HR_match_res_df["Gage height, m, HR side"])
idx_HRmax_dates = np.isfinite(HRlagrise_datenum) & np.isfinite(oceanside_max_HR_match_res_df["Resulting Max Gage height, m, HR side"])
z_oceanside_dates = np.polyfit(HRlagrise_datenum[idx_HRmatch_dates], oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"][idx_HRmatch_dates], 1)
z_HRmatch_dates = np.polyfit(HRlagrise_datenum[idx_HRmatch_dates], oceanside_max_HR_match_res_df["Gage height, m, HR side"][idx_HRmatch_dates], 1)
z_HRmax_dates = np.polyfit(HRlagrise_datenum[idx_HRmax_dates], oceanside_max_HR_match_res_df["Resulting Max Gage height, m, HR side"][idx_HRmax_dates], 1)
p_oceanside_dates = np.poly1d(z_oceanside_dates)
p_HRmatch_dates = np.poly1d(z_HRmatch_dates)
p_HRmax_dates = np.poly1d(z_HRmax_dates)
polyX_HRfit_dates = np.linspace(HRlagrise_datenum[idx_HRmatch_dates].min(), HRlagrise_datenum[idx_HRmatch_dates].max(), 100)

pylab.plot(polyX_HRfit_dates,p_oceanside_dates(polyX_HRfit_dates),"c")
pylab.plot(polyX_HRfit_dates,p_HRmatch_dates(polyX_HRfit_dates),"Green")
pylab.plot(polyX_HRfit_dates,p_HRmax_dates(polyX_HRfit_dates),"Yellow")

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_elev)
plt.legend(loc='lower right',bbox_to_anchor=(0.62,0),fontsize=18)

#%% Time Series Plot of Max Oceanside,and Resultant CTD Maxes

ax = oceanside_max_CNR_match_res_df.plot.scatter(x="datetime", y="Resulting Max Gage height, m, CNR U/S", color='LightBlue', label = 'Resulting Max Gage height, CNR U/S')
oceanside_max_DL_match_res_df.plot.scatter(x="datetime", y="Resulting Max Gage height, m, Dog Leg", color='LightGreen', label = 'Resulting Max Gage height, Dog Leg', ax=ax)
oceanside_max_HT_match_res_df.plot.scatter(x="datetime", y="Resulting Max Gage height, m, High Toss", color='Orange', label = 'Resulting Max Gage height, High Toss', ax=ax)

HRlagrise_datenum = mdates.date2num(oceanside_max_HR_match_res_df["datetime"]) # Same date range for everything
idx_CNR_dates = np.isfinite(HRlagrise_datenum) & np.isfinite(oceanside_max_CNR_match_res_df["Resulting Max Gage height, m, CNR U/S"])
idx_DL_dates = np.isfinite(HRlagrise_datenum) & np.isfinite(oceanside_max_DL_match_res_df["Resulting Max Gage height, m, Dog Leg"])
idx_HT_dates = np.isfinite(HRlagrise_datenum) & np.isfinite(oceanside_max_HT_match_res_df["Resulting Max Gage height, m, High Toss"])
z_CNR_dates = np.polyfit(HRlagrise_datenum[idx_CNR_dates], oceanside_max_CNR_match_res_df["Resulting Max Gage height, m, CNR U/S"][idx_CNR_dates], 1)
z_DL_dates = np.polyfit(HRlagrise_datenum[idx_DL_dates], oceanside_max_DL_match_res_df["Resulting Max Gage height, m, Dog Leg"][idx_DL_dates], 1)
z_HT_dates = np.polyfit(HRlagrise_datenum[idx_HT_dates], oceanside_max_HT_match_res_df["Resulting Max Gage height, m, High Toss"][idx_HT_dates], 1)
p_CNR_dates = np.poly1d(z_CNR_dates)
p_DL_dates = np.poly1d(z_DL_dates)
p_HT_dates = np.poly1d(z_HT_dates)
polyX_HRfit_dates = np.linspace(HRlagrise_datenum[idx_HRmatch_dates].min(), HRlagrise_datenum[idx_HRmatch_dates].max(), 100)

pylab.plot(polyX_HRfit_dates,p_CNR_dates(polyX_HRfit_dates),"c")
pylab.plot(polyX_HRfit_dates,p_DL_dates(polyX_HRfit_dates),"Yellow")
pylab.plot(polyX_HRfit_dates,p_HT_dates(polyX_HRfit_dates),"Green")

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_elev)
plt.legend(loc='lower right',bbox_to_anchor=(0.62,0),fontsize=18)

#%% Max Oceanside vs. Matching HRside Levels & vs. Resultant HRside Maxes

ax = oceanside_max_HR_match_res_df.plot.scatter(x="Max Gage height, m, Ocean side", y="Gage height, m, HR side", marker='.', label='Herring River Levels, Same Time')
oceanside_max_HR_match_res_df.plot.scatter(x="Max Gage height, m, Ocean side", y="Resulting Max Gage height, m, HR side", marker='+', color='Green', label='Maximum Herring River Levels, Tidally Lagged', ax=ax)
plt.xlabel('Maximum Gage height, Ocean side of dike [m NAVD88]', fontsize=22)
plt.ylabel('Gage height, Herring River side of dike [m NAVD88]', fontsize=22)
plt.legend(loc='lower right', bbox_to_anchor=(1,0.2), fontsize=22)

# Linear fits to half hour time series levels    
idx_HRmatch = np.isfinite(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"]) & np.isfinite(oceanside_max_HR_match_res_df["Gage height, m, HR side"])
idx_HRmax = np.isfinite(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"]) & np.isfinite(oceanside_max_HR_match_res_df["Resulting Max Gage height, m, HR side"])
z_HRmatch = np.polyfit(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].loc[idx_HRmatch], oceanside_max_HR_match_res_df["Gage height, m, HR side"].loc[idx_HRmatch], 1)
z_HRmax = np.polyfit(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].loc[idx_HRmax], oceanside_max_HR_match_res_df["Resulting Max Gage height, m, HR side"].loc[idx_HRmax], 1)
p_HRmatch = np.poly1d(z_HRmatch)
p_HRmax = np.poly1d(z_HRmax)
polyX_HRfit = np.linspace(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].loc[idx_HRmatch].min(), oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].loc[idx_HRmatch].max(), 100)

pylab.plot(polyX_HRfit,p_HRmatch(polyX_HRfit),"c")
pylab.plot(polyX_HRfit,p_HRmax(polyX_HRfit),"LightGreen")
# the line equations:
print("y=%.6fx+(%.6f)"%(z_HRmax[0],z_HRmax[1]))

oceanside_max_HR_match_res_nona_df = oceanside_max_HR_match_res_df.dropna()
oceanside_max_HR_match_res_nona_df = oceanside_max_HR_match_res_df.drop(columns=["Gage height, m, HR side"])
oceanside_max_HR_match_res_nona_df.dropna(inplace=True)

# Using "Uncertainties" Module, Linear Regression

x = oceanside_max_HR_match_res_nona_df['Max Gage height, m, Ocean side'].values
y = oceanside_max_HR_match_res_nona_df['Resulting Max Gage height, m, HR side'].values
n = len(y)

popt, pcov = curve_fit(f_line, x, y)

# retrieve parameter values
a = popt[0]
b = popt[1]
print('Optimal Values')
print('a: ' + str(a))
print('b: ' + str(b))

# compute r^2
r2 = 1.0-(sum((y-f_line(x,a,b))**2)/((n-1.0)*np.var(y,ddof=1)))
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

lpb, upb = predband(px, x, y, popt, f_line, conf=0.95)

# plot the regression
plt.plot(px, nom, c='black', label='Regressed Fit (Linear)')

# uncertainty lines (95% confidence)
plt.plot(px, nom - 1.96 * std, c='orange', label='95% Confidence Region')
plt.plot(px, nom + 1.96 * std, c='orange')
# prediction band (95% confidence)
plt.plot(px, lpb, 'k--',label='95% Prediction Band')
plt.plot(px, upb, 'k--')
plt.xlabel('WF Harbor Near-Dike Maximum Levels [m NAVD88]', fontsize=22)
plt.ylabel('Herring River Near-Dike \n Proceeding Maximum Levels [m NAVD88]', fontsize=22)
plt.legend(loc='lower right', bbox_to_anchor=(0.95,0.05), fontsize=22)

# Using "Uncertainties" Module, Nonlinear Regression
def f_exp(x, a, b, c):
    return a * np.exp(b*x) + c

popt, pcov = curve_fit(f_exp, x, y)

# retrieve parameter values
a = popt[0]
b = popt[1]
c = popt[2]
print('Optimal Values')
print('a: ' + str(a))
print('b: ' + str(b))
print('c: ' + str(c))

# compute r^2
r2 = 1.0-(sum((y-f_exp(x,a,b,c))**2)/((n-1.0)*np.var(y,ddof=1)))
print('R^2: ' + str(r2))

# calculate parameter confidence interval
a,b,c = unc.correlated_values(popt, pcov)
print('Uncertainty')
print('a: ' + str(a))
print('b: ' + str(b))
print('c: ' + str(c))

# plot data
plt.scatter(x, y, s=3, label='Data')

# calculate regression confidence interval
px = np.linspace(np.nanmin(x), np.nanmax(x), 100)
py = a*unp.exp(b*px)+c
nom = unp.nominal_values(py)
std = unp.std_devs(py)

lpb, upb = predband(px, x, y, popt, f_exp, conf=0.95)

# plot the regression
plt.plot(px, nom, c='gray', label='y=a exp(b x) + c')

# uncertainty lines (95% confidence)
plt.plot(px, nom - 1.96 * std, c='orange',\
         label='95% Confidence Region')
plt.plot(px, nom + 1.96 * std, c='orange')
# prediction band (95% confidence)
plt.plot(px, lpb, 'k--',label='95% Prediction Band')
plt.plot(px, upb, 'k--')
plt.xlabel('WF Harbor Near-Dike Maximum Levels [m NAVD88]', fontsize=22)
plt.ylabel('Herring River Near-Dike \n Proceeding Maximum Levels [m NAVD88]', fontsize=22)
plt.legend(loc='lower right', bbox_to_anchor=(0.95,0.05), fontsize=22)

# Using PyCSE Regress



#%% Color different years
index_halfway = int(len(oceanside_max_HR_match_res_df)/2)

# 2018
ax = oceanside_max_HR_match_res_df.iloc[0:index_halfway].plot.scatter(x="Max Gage height, m, Ocean side", y="Gage height, m, HR side", marker='.', label='Concurrent Herring River Levels, 2018')
oceanside_max_HR_match_res_df.iloc[0:index_halfway].plot.scatter(x="Max Gage height, m, Ocean side", y="Resulting Max Gage height, m, HR side", marker='+', color='Green', label='Subsequent Maximum Herring River Levels, 2018', ax=ax)

# Linear fits to half hour time series levels    
idx_HRmatch = np.isfinite(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].iloc[0:index_halfway]) & np.isfinite(oceanside_max_HR_match_res_df["Gage height, m, HR side"].iloc[0:index_halfway])
idx_HRmax = np.isfinite(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].iloc[0:index_halfway]) & np.isfinite(oceanside_max_HR_match_res_df["Resulting Max Gage height, m, HR side"].iloc[0:index_halfway])
z_HRmatch = np.polyfit(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].iloc[0:index_halfway].loc[idx_HRmatch], oceanside_max_HR_match_res_df["Gage height, m, HR side"].iloc[0:index_halfway].loc[idx_HRmatch], 1)
z_HRmax = np.polyfit(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].iloc[0:index_halfway].loc[idx_HRmax], oceanside_max_HR_match_res_df["Resulting Max Gage height, m, HR side"].iloc[0:index_halfway].loc[idx_HRmax], 1)
p_HRmatch = np.poly1d(z_HRmatch)
p_HRmax = np.poly1d(z_HRmax)
polyX_HRfit = np.linspace(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].iloc[0:index_halfway].loc[idx_HRmatch].min(), oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].iloc[0:index_halfway].loc[idx_HRmatch].max(), 100)

pylab.plot(polyX_HRfit,p_HRmatch(polyX_HRfit),"c")
pylab.plot(polyX_HRfit,p_HRmax(polyX_HRfit),"LightGreen")

# 2019
oceanside_max_HR_match_res_df.iloc[index_halfway:-1].plot.scatter(x="Max Gage height, m, Ocean side", y="Gage height, m, HR side", marker=1, color='Orange', label='Concurrent Herring River Levels, 2019', ax=ax)
oceanside_max_HR_match_res_df.iloc[index_halfway:-1].plot.scatter(x="Max Gage height, m, Ocean side", y="Resulting Max Gage height, m, HR side", marker=2, color='Purple', label='Subsequent Maximum Herring River Levels, 2019', ax=ax)
plt.xlabel('Maximum Gage height, Ocean side of dike [m NAVD88]', fontsize=22)
plt.ylabel('Gage height, Herring River side of dike [m NAVD88]', fontsize=22)
plt.legend(loc='lower right', bbox_to_anchor=(1,0.05), fontsize=22)

# Linear fits to half hour time series levels    
idx_HRmatch = np.isfinite(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].iloc[index_halfway:-1]) & np.isfinite(oceanside_max_HR_match_res_df["Gage height, m, HR side"].iloc[index_halfway:-1])
idx_HRmax = np.isfinite(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].iloc[index_halfway:-1]) & np.isfinite(oceanside_max_HR_match_res_df["Resulting Max Gage height, m, HR side"].iloc[index_halfway:-1])
z_HRmatch = np.polyfit(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].iloc[index_halfway:-1].loc[idx_HRmatch], oceanside_max_HR_match_res_df["Gage height, m, HR side"].iloc[index_halfway:-1].loc[idx_HRmatch], 1)
z_HRmax = np.polyfit(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].iloc[index_halfway:-1].loc[idx_HRmax], oceanside_max_HR_match_res_df["Resulting Max Gage height, m, HR side"].iloc[index_halfway:-1].loc[idx_HRmax], 1)
p_HRmatch = np.poly1d(z_HRmatch)
p_HRmax = np.poly1d(z_HRmax)
polyX_HRfit = np.linspace(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].iloc[index_halfway:-1].loc[idx_HRmatch].min(), oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].iloc[index_halfway:-1].loc[idx_HRmatch].max(), 100)

pylab.plot(polyX_HRfit,p_HRmatch(polyX_HRfit),"Yellow")
pylab.plot(polyX_HRfit,p_HRmax(polyX_HRfit),"Violet")

#%% Difference in maximum HR level and level at oceanside max

HRside_diff_maxmatch = oceanside_max_HR_match_res_df["Resulting Max Gage height, m, HR side"]-oceanside_max_HR_match_res_df["Gage height, m, HR side"]

# Height gained in HR after ocean side reaches maxima
plt.figure()
plt.scatter(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"],HRside_diff_maxmatch,label="Dike-Delayed \n Peak Rise, \n Herring River (HR) \n [$\Delta$m NAVD88]")
plt.xlabel('Maximum Gage height, Ocean side of dike [m NAVD88]', fontsize=22)
plt.ylabel('Magnitude of Delayed Herring River rise [m]', fontsize=22)

idx_HRlagrise = np.isfinite(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"]) & np.isfinite(HRside_diff_maxmatch)
z_HRlagrise = np.polyfit(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].loc[idx_HRlagrise], HRside_diff_maxmatch.loc[idx_HRlagrise], 1)
p_HRlagrise = np.poly1d(z_HRlagrise)
polyX_HRlagfit = np.linspace(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].loc[idx_HRlagrise].min(), oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].loc[idx_HRlagrise].max(), 100)

pylab.plot(polyX_HRlagfit,p_HRlagrise(polyX_HRlagfit),"c")
plt.legend(bbox_to_anchor=(0.9,0.9),fontsize=18)

#%% Difference in maximum HR level and level at oceanside max as function of time

plt.figure()
plt.scatter(oceanside_max_HR_match_res_df["datetime"],HRside_diff_maxmatch,label="Dike-Delayed \n Peak Rise, \n Herring River (HR) \n [$\Delta$m NAVD88]")
plt.xlabel('Date', fontsize=22)
plt.ylabel('Magnitude of Delayed Herring River rise [m]', fontsize=22)

idx_HRlagrise_date = np.isfinite(HRlagrise_datenum) & np.isfinite(HRside_diff_maxmatch)
z_HRlagrise_date = np.polyfit(HRlagrise_datenum[idx_HRlagrise_date], HRside_diff_maxmatch[idx_HRlagrise_date], 1)
p_HRlagrise_date = np.poly1d(z_HRlagrise_date)
polyX_HRlagfit_date = np.linspace(HRlagrise_datenum[idx_HRlagrise_date].min(), HRlagrise_datenum[idx_HRlagrise_date].max(), 100)

pylab.plot(polyX_HRlagfit_date,p_HRlagrise_date(polyX_HRlagfit_date),"c")

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_elev)
plt.legend(bbox_to_anchor=(0.9,0.9),fontsize=18)

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


#%% Tidal Ranges at each sensor

# Combine max and min level arrays, keeping and ordering dates.

# Harbor Side of Dike
maxminlevels_ocean_ordered_df = pd.merge_ordered(oceanside_maxlevels_df, oceanside_minlevels_df, on="datetime")
oceanside_maxes = maxminlevels_ocean_ordered_df["Max Gage height, m, Ocean side"]
oceanside_mins = maxminlevels_ocean_ordered_df["Min Gage height, m, Ocean side"]
datetime_ocean_maxes = maxminlevels_ocean_ordered_df["datetime"]

# HR Side of Dike
maxminlevels_HR_ordered_df = pd.merge_ordered(HRside_maxlevels_df, HRside_minlevels_df, on="datetime")
HRside_maxes = maxminlevels_HR_ordered_df["Max Gage height, m, HR side"]
HRside_mins = maxminlevels_HR_ordered_df["Min Gage height, m, HR side"]
datetime_HR_maxes = maxminlevels_HR_ordered_df["datetime"]

# CNR U/S
maxminlevels_CNR_ordered_df = pd.merge_ordered(CNR_maxlevels_df, CNR_minlevels_df, on="datetime")
CNRUS_maxes = maxminlevels_CNR_ordered_df["Max Gage height, m, CNR U/S"]
CNRUS_mins = maxminlevels_CNR_ordered_df["Min Gage height, m, CNR U/S"]
datetime_CNR_maxes = maxminlevels_CNR_ordered_df["datetime"]

# Dog Leg
maxminlevels_DL_ordered_df = pd.merge_ordered(DL_maxlevels_df, DL_minlevels_df, on="datetime")
DogLeg_maxes = maxminlevels_DL_ordered_df["Max Gage height, m, Dog Leg"]
DogLeg_mins = maxminlevels_DL_ordered_df["Min Gage height, m, Dog Leg"]
datetime_DL_maxes = maxminlevels_DL_ordered_df["datetime"]

# High Toss
maxminlevels_HT_ordered_df = pd.merge_ordered(HT_maxlevels_df, HT_minlevels_df, on="datetime")
HighToss_maxes = maxminlevels_HT_ordered_df["Max Gage height, m, High Toss"]
HighToss_mins = maxminlevels_HT_ordered_df["Min Gage height, m, High Toss"]
datetime_HT_maxes = maxminlevels_HT_ordered_df["datetime"]

# Get the mins resulting (res) from the maxes at a gage
deltaT_oceanranges, y_oceanside_res_mins = MaxToMin(oceanside_mins, oceanside_maxes, datetime_ocean_maxes)
deltaT_HRranges, y_HRside_res_mins = MaxToMin(HRside_mins, HRside_maxes, datetime_HR_maxes)
deltaT_CNRranges, y_CNR_res_mins = MaxToMin(CNRUS_mins, CNRUS_maxes, datetime_CNR_maxes)
deltaT_DLranges, y_DL_res_mins = MaxToMin(DogLeg_mins, DogLeg_maxes, datetime_DL_maxes)
deltaT_HTranges, y_HT_res_mins = MaxToMin(HighToss_mins, HighToss_maxes, datetime_HT_maxes)

# Combine datetimes of maxima with proceeding minima, merge with max levels
# Ranges here are associated with the datetime of the max levels

# Harbor Side of Dike
oceanside_min_arr = np.vstack([max_dates_ocean, y_oceanside_res_mins]).T
oceanside_min_df = pd.DataFrame(oceanside_min_arr, columns=["datetime","Proceeding Min Gage height, m, Ocean side"])
oceanside_min_df["datetime"] = pd.to_datetime(oceanside_min_df["datetime"])
oceanside_min_df["Proceeding Min Gage height, m, Ocean side"] = pd.to_numeric(oceanside_min_df["Proceeding Min Gage height, m, Ocean side"])
max_ocean_min_ocean_df = pd.merge(oceanside_maxlevels_df, oceanside_min_df)
ranges_ocean = (max_ocean_min_ocean_df["Max Gage height, m, Ocean side"]-max_ocean_min_ocean_df["Proceeding Min Gage height, m, Ocean side"])
ranges_ocean_arr = np.vstack([max_dates_ocean, ranges_ocean]).T
ranges_ocean_df = pd.DataFrame(ranges_ocean_arr, columns=["datetime","Range, m, Ocean side"])
ranges_ocean_df["datetime"] = pd.to_datetime(ranges_ocean_df["datetime"])
ranges_ocean_df["Range, m, Ocean side"] = pd.to_numeric(ranges_ocean_df["Range, m, Ocean side"])

# HR Side of Dike
HRside_min_arr = np.vstack([max_dates_HR, y_HRside_res_mins]).T
HRside_min_df = pd.DataFrame(HRside_min_arr, columns=["datetime","Proceeding Min Gage height, m, HR side"])
HRside_min_df["datetime"] = pd.to_datetime(HRside_min_df["datetime"])
HRside_min_df["Proceeding Min Gage height, m, HR side"] = pd.to_numeric(HRside_min_df["Proceeding Min Gage height, m, HR side"])
max_HR_min_HR_df = pd.merge(HRside_maxlevels_df, HRside_min_df)
ranges_HR = (max_HR_min_HR_df["Max Gage height, m, HR side"]-max_HR_min_HR_df["Proceeding Min Gage height, m, HR side"])
ranges_HR_arr = np.vstack([max_dates_HR, ranges_HR]).T
ranges_HR_df = pd.DataFrame(ranges_HR_arr, columns=["datetime","Range, m, HR side"])
ranges_HR_df["datetime"] = pd.to_datetime(ranges_HR_df["datetime"])
ranges_HR_df["Range, m, HR side"] = pd.to_numeric(ranges_HR_df["Range, m, HR side"])

# CNR U/S
CNRUS_min_arr = np.vstack([max_dates_CNR, y_CNR_res_mins]).T
CNRUS_min_df = pd.DataFrame(CNRUS_min_arr, columns=["datetime","Proceeding Min Gage height, m, CNR U/S"])
CNRUS_min_df["datetime"] = pd.to_datetime(CNRUS_min_df["datetime"])
CNRUS_min_df["Proceeding Min Gage height, m, CNR U/S"] = pd.to_numeric(CNRUS_min_df["Proceeding Min Gage height, m, CNR U/S"])
max_CNR_min_CNR_df = pd.merge(CNR_maxlevels_df, CNRUS_min_df)
ranges_CNR = (max_CNR_min_CNR_df["Max Gage height, m, CNR U/S"]-max_CNR_min_CNR_df["Proceeding Min Gage height, m, CNR U/S"])
ranges_CNR_arr = np.vstack([max_dates_CNR, ranges_CNR]).T
ranges_CNR_df = pd.DataFrame(ranges_CNR_arr, columns=["datetime","Range, m, CNR U/S"])
ranges_CNR_df["datetime"] = pd.to_datetime(ranges_CNR_df["datetime"])
ranges_CNR_df["Range, m, CNR U/S"] = pd.to_numeric(ranges_CNR_df["Range, m, CNR U/S"])

# Dog Leg
DogLeg_min_arr = np.vstack([max_dates_DL, y_DL_res_mins]).T
DogLeg_min_df = pd.DataFrame(DogLeg_min_arr, columns=["datetime","Proceeding Min Gage height, m, Dog Leg"])
DogLeg_min_df["datetime"] = pd.to_datetime(DogLeg_min_df["datetime"])
DogLeg_min_df["Proceeding Min Gage height, m, Dog Leg"] = pd.to_numeric(DogLeg_min_df["Proceeding Min Gage height, m, Dog Leg"])
max_DL_min_DL_df = pd.merge(DL_maxlevels_df, DogLeg_min_df)
ranges_DL = (max_DL_min_DL_df["Max Gage height, m, Dog Leg"]-max_DL_min_DL_df["Proceeding Min Gage height, m, Dog Leg"])
ranges_DL_arr = np.vstack([max_dates_DL, ranges_DL]).T
ranges_DL_df = pd.DataFrame(ranges_DL_arr, columns=["datetime","Range, m, Dog Leg"])
ranges_DL_df["datetime"] = pd.to_datetime(ranges_DL_df["datetime"])
ranges_DL_df["Range, m, Dog Leg"] = pd.to_numeric(ranges_DL_df["Range, m, Dog Leg"])

# High Toss
HighToss_min_arr = np.vstack([max_dates_HT, y_HT_res_mins]).T
HighToss_min_df = pd.DataFrame(HighToss_min_arr, columns=["datetime","Proceeding Min Gage height, m, High Toss"])
HighToss_min_df["datetime"] = pd.to_datetime(HighToss_min_df["datetime"])
HighToss_min_df["Proceeding Min Gage height, m, High Toss"] = pd.to_numeric(HighToss_min_df["Proceeding Min Gage height, m, High Toss"])
max_HT_min_HT_df = pd.merge(HT_maxlevels_df, HighToss_min_df)
ranges_HT = (max_HT_min_HT_df["Max Gage height, m, High Toss"]-max_HT_min_HT_df["Proceeding Min Gage height, m, High Toss"])
ranges_HT_arr = np.vstack([max_dates_HT, ranges_HT]).T
ranges_HT_df = pd.DataFrame(ranges_HT_arr, columns=["datetime","Range, m, High Toss"])
ranges_HT_df["datetime"] = pd.to_datetime(ranges_HT_df["datetime"])
ranges_HT_df["Range, m, High Toss"] = pd.to_numeric(ranges_HT_df["Range, m, High Toss"])

#%% Range Times, HR Side

# Time Series
deltaT_HRranges_df = pd.DataFrame(deltaT_HRranges, columns=["delta datetime"])
deltaT_HRranges_df["delta datetime"] = deltaT_HRranges_df["delta datetime"].dt.seconds/3600
mean_HRrange_time = np.nanmean(deltaT_HRranges_df["delta datetime"])
stdev_HRrange_time = np.std(deltaT_HRranges_df["delta datetime"])
print("Mean range time HRside =", int(round(mean_HRrange_time)), "hours and", int(round(abs(mean_HRrange_time-round(mean_HRrange_time))*60)), "minutes.")
# This makes sense, as the time from the peak to the mean level outside is 1/4 tidal cycle, or 3 hrs 13 minutes.
# The head in the HR is increasing fastest when the ocean is at its peak, and maxes when ocean hits mean.
print("Standard Deviation of range time HRside =", int(round(abs(stdev_HRrange_time-round(stdev_HRrange_time))*60)), "minutes.")
plt.figure()
plt.scatter(mdates.date2num(max_dates_HR), deltaT_HRranges_df["delta datetime"], label="Time of HRside Range (max to min)")

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_time_hrs)
plt.legend(loc='lower center', bbox_to_anchor = (0.5,0.45), fontsize=18)

# Vs. Harbor Maxes
HRside_deltaT_arr = np.vstack([max_dates_HR, deltaT_HRranges]).T
HRside_deltaT_df = pd.DataFrame(HRside_deltaT_arr, columns=["datetime","HR Drain Time"])
HRside_deltaT_df["datetime"] = pd.to_datetime(HRside_deltaT_df["datetime"])
HRside_deltaT_df["HR Drain Time"] = HRside_deltaT_df["HR Drain Time"].dt.seconds/3600

# Separate based on seasons, 0 = Oct-Apr (most rech), 1 = May-Sept (least rech)
HRside_deltaT_df['Season, Oct-Apr=0, May-Sept=1'] = [0 if x.month in [10,11,12,1,2,3,4] else 1 for x in HRside_deltaT_df["datetime"]]

maxocean_HRdT_ordered_df = pd.merge_ordered(oceanside_maxlevels_df, HRside_deltaT_df, on="datetime")
HRside_drains = maxocean_HRdT_ordered_df["HR Drain Time"]
HRside_seasons = maxocean_HRdT_ordered_df['Season, Oct-Apr=0, May-Sept=1']

deltaT_max_to_drain, y_HRside_res_drains = ResMax(HRside_drains, oceanHR_maxes, datetime_oceanHR_maxes)
deltaT_max_to_drain2, y_HRside_res_seasons = ResMax(HRside_seasons, oceanHR_maxes, datetime_oceanHR_maxes)

drains_seasons_HR_arr = np.vstack([y_oceanside_maxes, y_HRside_res_drains, y_HRside_res_seasons]).T
drains_seasons_HR_df = pd.DataFrame(drains_seasons_HR_arr, columns=["Max Gage height, m, Ocean side","HR Drain Time",'Season, Oct-Apr=0, May-Sept=1'])
drains_seasons_HR_df["Max Gage height, m, Ocean side"] = pd.to_numeric(drains_seasons_HR_df["Max Gage height, m, Ocean side"])
drains_seasons_HR_df["HR Drain Time"] = pd.to_numeric(drains_seasons_HR_df["HR Drain Time"])
drains_seasons_HR_df['Season, Oct-Apr=0, May-Sept=1'] = pd.to_numeric(drains_seasons_HR_df['Season, Oct-Apr=0, May-Sept=1'])

fg = sns.FacetGrid(data=drains_seasons_HR_df, hue='Season, Oct-Apr=0, May-Sept=1')
fg.map(plt.scatter, "Max Gage height, m, Ocean side", "HR Drain Time").add_legend()

plt.figure()
plt.scatter(y_oceanside_maxes, y_HRside_res_drains)
plt.xlabel('Maximum Gage height, Ocean side of dike [m NAVD88]', fontsize=22)
plt.ylabel('Time to Drain River at HR sensor [hours]', fontsize=22)

idx_HRdrain = np.isfinite(y_oceanside_maxes) & np.isfinite(y_HRside_res_drains)
z_HRdrain = np.polyfit(y_oceanside_maxes[idx_HRdrain], y_HRside_res_drains[idx_HRdrain], 1)
p_HRdrain = np.poly1d(z_HRdrain)
polyX_HR_drainfit = np.linspace(y_oceanside_maxes[idx_HRdrain].min(), y_oceanside_maxes[idx_HRdrain].max(), 100)

# pylab.plot(polyX_HR_drainfit,p_HRdrain(polyX_HR_drainfit),"c")

#%% Range Times, CNR U/S

deltaT_CNRranges_df = pd.DataFrame(deltaT_CNRranges, columns=["delta datetime"])
deltaT_CNRranges_df["delta datetime"] = deltaT_CNRranges_df["delta datetime"].dt.seconds/3600
mean_CNRrange_time = np.nanmean(deltaT_CNRranges_df["delta datetime"])
stdev_CNRrange_time = np.std(deltaT_CNRranges_df["delta datetime"])
print("Mean range time CNR U/S =", int(round(mean_CNRrange_time)), "hours and", int(round(abs(mean_CNRrange_time-round(mean_CNRrange_time))*60)), "minutes.")
# This makes sense, as the time from the peak to the mean level outside is 1/4 tidal cycle, or 3 hrs 13 minutes.
# The head in the CNR is increasing fastest when the ocean is at its peak, and maxes when ocean hits mean.
print("Standard Deviation of range time CNR U/S =", int(round(abs(stdev_CNRrange_time-round(stdev_CNRrange_time))*60)), "minutes.")
plt.figure()
plt.scatter(mdates.date2num(max_dates_CNR), deltaT_CNRranges_df["delta datetime"], label="Time of CNR U/S Range (max to min)")

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_time_hrs)
plt.legend(loc='lower center', bbox_to_anchor = (0.5,0.45), fontsize=18)

#%% Range Times, Dog Leg

deltaT_DLranges_df = pd.DataFrame(deltaT_DLranges, columns=["delta datetime"])
deltaT_DLranges_df["delta datetime"] = deltaT_DLranges_df["delta datetime"].dt.seconds/3600
mean_HRrange_time = np.nanmean(deltaT_DLranges_df["delta datetime"])
stdev_HRrange_time = np.std(deltaT_DLranges_df["delta datetime"])
print("Mean range time Dog Leg =", int(round(mean_HRrange_time)), "hours and", int(round(abs(mean_HRrange_time-round(mean_HRrange_time))*60)), "minutes.")
# This makes sense, as the time from the peak to the mean level outside is 1/4 tidal cycle, or 3 hrs 13 minutes.
# The head in the HR is increasing fastest when the ocean is at its peak, and maxes when ocean hits mean.
print("Standard Deviation of range time Dog Leg =", int(round(abs(stdev_HRrange_time-round(stdev_HRrange_time))*60)), "minutes.")
plt.figure()
plt.scatter(mdates.date2num(max_dates_DL), deltaT_DLranges_df["delta datetime"], label="Time of Dog Leg Range (max to min)")

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_time_hrs)
plt.legend(loc='lower center', bbox_to_anchor = (0.5,0.45), fontsize=18)

#%% Range Times, High Toss

deltaT_HTranges_df = pd.DataFrame(deltaT_HTranges, columns=["delta datetime"])
deltaT_HTranges_df["delta datetime"] = deltaT_HTranges_df["delta datetime"].dt.seconds/3600
mean_HRrange_time = np.nanmean(deltaT_HTranges_df["delta datetime"])
stdev_HRrange_time = np.std(deltaT_HTranges_df["delta datetime"])
print("Mean range time High Toss =", int(round(mean_HRrange_time)), "hours and", int(round(abs(mean_HRrange_time-round(mean_HRrange_time))*60)), "minutes.")
# This makes sense, as the time from the peak to the mean level outside is 1/4 tidal cycle, or 3 hrs 13 minutes.
# The head in the HR is increasing fastest when the ocean is at its peak, and maxes when ocean hits mean.
print("Standard Deviation of range time High Toss =", int(round(abs(stdev_HRrange_time-round(stdev_HRrange_time))*60)), "minutes.")
plt.figure()
plt.scatter(mdates.date2num(max_dates_HT), deltaT_HTranges_df["delta datetime"], label="Time of High Toss Range (max to min)")

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_time_hrs)
plt.legend(loc='lower center', bbox_to_anchor = (0.5,0.45), fontsize=18)

# Range seems to not be very meaningful as the time from peak to trough is shorter than the time
# from trough to peak. This time lengthens (gets closer to the actual tidal range time of 6.2 hours)
# as we get farther from the dike.

#%% Compare Tidal ranges in HR at max and min oceanside levels
"""
Mean High and Low Tides at Dike (to determine changes in amplitude and amplitude decay from ocean to river)
"""
# Dataframe of Oceanside maxes and HR levels at same time.
# Herring River Side of Dike
maxocean_rangeHR_ordered_df = pd.merge_ordered(oceanside_maxlevels_df, ranges_HR_df, on="datetime")
oceanHR_max_range = maxocean_rangeHR_ordered_df["Max Gage height, m, Ocean side"]
HRside_max_range = maxocean_rangeHR_ordered_df["Range, m, HR side"]
datetime_oceanHR_max_range = maxocean_rangeHR_ordered_df["datetime"]

# Need to plot HR separate from CTD sensors (magnitudes are slightly off) - might be the bottleneck effect though

# CNR U/S
maxocean_rangeCNR_ordered_df = pd.merge_ordered(oceanside_maxlevels_df, ranges_CNR_df, on="datetime")
oceanCNR_max_range = maxocean_rangeCNR_ordered_df["Max Gage height, m, Ocean side"]
CNRUS_max_range = maxocean_rangeCNR_ordered_df["Range, m, CNR U/S"]
datetime_oceanCNR_max_range = maxocean_rangeCNR_ordered_df["datetime"]

# Dog Leg
maxocean_rangeDL_ordered_df = pd.merge_ordered(oceanside_maxlevels_df, ranges_DL_df, on="datetime")
oceanDL_max_range = maxocean_rangeDL_ordered_df["Max Gage height, m, Ocean side"]
DogLeg_max_range = maxocean_rangeDL_ordered_df["Range, m, Dog Leg"]
datetime_oceanDL_max_range = maxocean_rangeDL_ordered_df["datetime"]

# High Toss
maxocean_rangeHT_ordered_df = pd.merge_ordered(oceanside_maxlevels_df, ranges_HT_df, on="datetime")
oceanHT_max_range = maxocean_rangeHT_ordered_df["Max Gage height, m, Ocean side"]
HighToss_max_range = maxocean_rangeHT_ordered_df["Range, m, High Toss"]
datetime_oceanHT_max_range = maxocean_rangeHT_ordered_df["datetime"]

# Get the ranges reached in the HR resulting (res) from the maxes in the harbor
# Using ResMax function will work
# deltaT outputs should be the same as those for the max comparison
deltaT_HR_max_range, y_HRside_res_range = ResMax(HRside_max_range, oceanHR_max_range, datetime_oceanHR_max_range)
deltaT_CNR_max_range, y_CNR_res_range = ResMax(CNRUS_max_range, oceanCNR_max_range, datetime_oceanCNR_max_range)
deltaT_DL_max_range, y_DL_res_range = ResMax(DogLeg_max_range, oceanDL_max_range, datetime_oceanDL_max_range)
deltaT_HT_max_range, y_HT_res_range = ResMax(HighToss_max_range, oceanHT_max_range, datetime_oceanHT_max_range)

# Combine datetimes of maxima in harbor with resulting maxima in river, merge with levels at the same time

# HR Side of Dike
HRside_resrange_arr = np.vstack([max_dates_ocean, y_HRside_res_range]).T
HRside_resrange_df = pd.DataFrame(HRside_resrange_arr, columns=["datetime","Resulting Range, m, HR side"])
HRside_resrange_df["datetime"] = pd.to_datetime(HRside_resrange_df["datetime"])
HRside_resrange_df["Resulting Range, m, HR side"] = pd.to_numeric(HRside_resrange_df["Resulting Range, m, HR side"])

# CNR U/S of Dike
CNRUS_resrange_arr = np.vstack([max_dates_ocean, y_CNR_res_range]).T
CNRUS_resrange_df = pd.DataFrame(CNRUS_resrange_arr, columns=["datetime","Resulting Range, m, CNR U/S"])
CNRUS_resrange_df["datetime"] = pd.to_datetime(CNRUS_resrange_df["datetime"])
CNRUS_resrange_df["Resulting Range, m, CNR U/S"] = pd.to_numeric(CNRUS_resrange_df["Resulting Range, m, CNR U/S"])

# Dog Leg of Dike
DogLeg_resrange_arr = np.vstack([max_dates_ocean, y_DL_res_range]).T
DogLeg_resrange_df = pd.DataFrame(DogLeg_resrange_arr, columns=["datetime","Resulting Range, m, Dog Leg"])
DogLeg_resrange_df["datetime"] = pd.to_datetime(DogLeg_resrange_df["datetime"])
DogLeg_resrange_df["Resulting Range, m, Dog Leg"] = pd.to_numeric(DogLeg_resrange_df["Resulting Range, m, Dog Leg"])

# High Toss of Dike
HighToss_resrange_arr = np.vstack([max_dates_ocean, y_HT_res_range]).T
HighToss_resrange_df = pd.DataFrame(HighToss_resrange_arr, columns=["datetime","Resulting Range, m, High Toss"])
HighToss_resrange_df["datetime"] = pd.to_datetime(HighToss_resrange_df["datetime"])
HighToss_resrange_df["Resulting Range, m, High Toss"] = pd.to_numeric(HighToss_resrange_df["Resulting Range, m, High Toss"])

# Dataframe of Oceanside maxes and resulting HR ranges at same time.
oceanside_max_HRrange_df = pd.merge(oceanside_maxlevels_df, HRside_resrange_df)
oceanside_max_CNRrange_df = pd.merge(oceanside_maxlevels_df, CNRUS_resrange_df)
oceanside_max_DLrange_df = pd.merge(oceanside_maxlevels_df, DogLeg_resrange_df)
oceanside_max_HTrange_df = pd.merge(oceanside_maxlevels_df, HighToss_resrange_df)

#%% Time Series Plot of Max Oceanside, and Resultant HR Ranges

ax = oceanside_max_HRrange_df.plot.scatter(x="datetime", y="Resulting Range, m, HR side", color='LightGreen', label = "Resulting Range, m, HR side")

HRrange_datenum = mdates.date2num(oceanside_max_HRrange_df["datetime"])
idx_HRrange_dates = np.isfinite(HRrange_datenum) & np.isfinite(oceanside_max_HRrange_df["Resulting Range, m, HR side"])
z_HRrange_dates = np.polyfit(HRrange_datenum[idx_HRrange_dates], oceanside_max_HRrange_df["Resulting Range, m, HR side"][idx_HRrange_dates], 1)
p_HRrange_dates = np.poly1d(z_HRrange_dates)
polyX_HRrange_dates = np.linspace(HRrange_datenum[idx_HRrange_dates].min(), HRrange_datenum[idx_HRrange_dates].max(), 100)

pylab.plot(polyX_HRrange_dates,p_HRrange_dates(polyX_HRrange_dates),"Green",label='Regressed Herring River Ranges')

# Show X-axis major tick marks as dates
DateAxisFmt(ylabel_elev)
plt.legend(loc='center', bbox_to_anchor=(0.41,0.85), fontsize=22)

#%% Max Oceanside vs. Resultant Ranges

ax = oceanside_max_HRrange_df.plot.scatter(x="Max Gage height, m, Ocean side", y="Resulting Range, m, HR side", marker='.', label="Resulting Range, m, HR side")
plt.xlabel('Maximum Gage height, Ocean side of dike [m NAVD88]', fontsize=22)
plt.ylabel('Tidal Range, Herring River side of dike [m]', fontsize=22)
plt.legend(loc='lower right', bbox_to_anchor=(0.95,0.05), fontsize=22)

# Linear fits to half hour time series levels    
idx_HRrange = np.isfinite(oceanside_max_HRrange_df["Max Gage height, m, Ocean side"]) & np.isfinite(oceanside_max_HRrange_df["Resulting Range, m, HR side"])
z_HRrange = np.polyfit(oceanside_max_HRrange_df["Max Gage height, m, Ocean side"].loc[idx_HRrange], oceanside_max_HRrange_df["Resulting Range, m, HR side"].loc[idx_HRrange], 1)
p_HRrange = np.poly1d(z_HRrange)
polyX_HR_rangefit = np.linspace(oceanside_max_HRrange_df["Max Gage height, m, Ocean side"].loc[idx_HRrange].min(), oceanside_max_HRrange_df["Max Gage height, m, Ocean side"].loc[idx_HRrange].max(), 100)

pylab.plot(polyX_HR_rangefit,p_HRrange(polyX_HR_rangefit),"c")
# # the line equations:
# print("y=%.6fx+(%.6f)"%(z_oceanside_hourly[0],z_oceanside_hourly[1]))

ax = oceanside_max_CNRrange_df.plot.scatter(x="Max Gage height, m, Ocean side", y="Resulting Range, m, CNR U/S", color='Red', marker='.', label="CNR U/S")
oceanside_max_DLrange_df.plot.scatter(x="Max Gage height, m, Ocean side", y="Resulting Range, m, Dog Leg", color='Blue', marker='.', label="Dog Leg", ax=ax)
oceanside_max_HTrange_df.plot.scatter(x="Max Gage height, m, Ocean side", y="Resulting Range, m, High Toss", color='Green', marker='.', label="High Toss", ax=ax)
plt.xlabel('Maximum Gage height, Ocean side of dike [m NAVD88]', fontsize=22)
plt.ylabel('Tidal Range, Herring River side of dike [m]', fontsize=22)
plt.legend(loc='lower right', bbox_to_anchor=(0.95,0.05), fontsize=22)

# Linear fits to half hour time series levels    
idx_CNRrange = np.isfinite(oceanside_max_CNRrange_df["Max Gage height, m, Ocean side"]) & np.isfinite(oceanside_max_CNRrange_df["Resulting Range, m, CNR U/S"])
z_CNRrange = np.polyfit(oceanside_max_CNRrange_df["Max Gage height, m, Ocean side"].loc[idx_CNRrange], oceanside_max_CNRrange_df["Resulting Range, m, CNR U/S"].loc[idx_CNRrange], 1)
p_CNRrange = np.poly1d(z_CNRrange)
idx_DLrange = np.isfinite(oceanside_max_DLrange_df["Max Gage height, m, Ocean side"]) & np.isfinite(oceanside_max_DLrange_df["Resulting Range, m, Dog Leg"])
z_DLrange = np.polyfit(oceanside_max_DLrange_df["Max Gage height, m, Ocean side"].loc[idx_DLrange], oceanside_max_DLrange_df["Resulting Range, m, Dog Leg"].loc[idx_DLrange], 1)
p_DLrange = np.poly1d(z_DLrange)
idx_HTrange = np.isfinite(oceanside_max_HTrange_df["Max Gage height, m, Ocean side"]) & np.isfinite(oceanside_max_HTrange_df["Resulting Range, m, High Toss"])
z_HTrange = np.polyfit(oceanside_max_HTrange_df["Max Gage height, m, Ocean side"].loc[idx_HTrange], oceanside_max_HTrange_df["Resulting Range, m, High Toss"].loc[idx_HTrange], 1)
p_HTrange = np.poly1d(z_HTrange)

pylab.plot(polyX_HR_rangefit,p_CNRrange(polyX_HR_rangefit),"Salmon")
pylab.plot(polyX_HR_rangefit,p_DLrange(polyX_HR_rangefit),"c")
pylab.plot(polyX_HR_rangefit,p_HTrange(polyX_HR_rangefit),"lime")
# # the line equations:
# print("y=%.6fx+(%.6f)"%(z_oceanside_hourly[0],z_oceanside_hourly[1]))
# print("y=%.6fx+(%.6f)"%(z_HRside_hourly[0],z_HRside_hourly[1]))
# print("y=%.6fx+(%.6f)"%(z_HRside_hourly[0],z_HRside_hourly[1]))

#%% Summary Statistics

max_frames = [oceanside_maxlevels_df, HRside_maxlevels_df, CNR_maxlevels_df, DL_maxlevels_df, HT_maxlevels_df]
min_frames = [oceanside_minlevels_df, HRside_minlevels_df, CNR_minlevels_df, DL_minlevels_df, HT_minlevels_df]
range_frames = [ranges_ocean_df, ranges_HR_df, ranges_CNR_df, ranges_DL_df, ranges_HT_df]
HR_max_resam30min_df = reduce(lambda left, right: pd.merge_ordered(left, right, on=["datetime"]), max_frames)
HR_min_resam30min_df = reduce(lambda left, right: pd.merge_ordered(left, right, on=["datetime"]), min_frames)  
HR_range_resam30min_df = reduce(lambda left, right: pd.merge_ordered(left, right, on=["datetime"]), range_frames)

HR_all_5m_stats = HR_all_resam_df.describe()
HR_slice_30m_stats = HR_all_resam30min_df_slice.describe()
HR_max_30m_stats = HR_max_resam30min_df.describe()
HR_min_30m_stats = HR_min_resam30min_df.describe()
HR_range_30m_stats = HR_range_resam30min_df.describe()

#%% Fourier Transform of Tidal Data

# Interpolate between nan values to keep the same time delta
oceanside_levels_30m_nonans = np.array(HR_all_resam30min_df_slice["Gage height, m, Ocean side"])
nans, x = nan_helper(oceanside_levels_30m_nonans)
oceanside_levels_30m_nonans[nans]= np.interp(x(nans), x(~nans), oceanside_levels_30m_nonans[~nans])

sp = np.fft.fft(oceanside_levels_30m_nonans)
freq = np.fft.fftfreq(oceanside_levels_30m_nonans.shape[-1])
plt.plot(freq, sp.real, freq, sp.imag) # pointless?

#%% Using PyTides to get tidal constituents
# Downloaded version to work with Python 3.7 and above, from github user yudevan
# https://github.com/yudevan/pytides/commit/507f2bc5d19fa5e427045cc2bf9ed724daf67f0c

oceanside_levels_30m_nonans_df = pd.DataFrame(oceanside_levels_30m_nonans, index=HR_all_resam30min_df_slice['datetime'])

demeaned_oceanside_levels_30m = oceanside_levels_30m_nonans_df.values - HR_slice_30m_stats['Gage height, m, Ocean side']['mean']
# dates_oceanside_30m = np.array(HR_all_resam30min_df_slice['datetime']).to_datetime()

##Prepare a list of datetimes, each 30 minutes apart, for a week.
# prediction_t0 = HR_all_resam30min_df_slice['datetime'].iloc[-1]
prediction_t0 = datetime(1945,5,8) # Start of Boston Verified Data
hours = 0.5*np.arange(56486 * 24 * 2) # 56486 days from May 8, 1945 to Jan 1, 2100 (replace with "7" to represent a week)
times = Tide._times(prediction_t0, hours)

##Fit the tidal data to the harmonic model using Pytides
my_tide = Tide.decompose(oceanside_levels_30m_nonans.T, HR_all_resam30min_df_slice['datetime'])
##Predict the tides using the Pytides model.
my_prediction = my_tide.at(times)

my_prediction_df = pd.DataFrame(my_prediction, index=times)

##Plot the results
plt.plot(HR_all_resam30min_df_slice['datetime'], oceanside_levels_30m_nonans)
plt.plot(times, my_prediction, label="Pytides")
# plt.plot(hours, noaa_predicted, label="NOAA Prediction")
# plt.plot(hours, noaa_verified, label="NOAA Verified")
DateAxisFmt(ylabel_elev)
plt.legend()
# plt.title('Comparison of Pytides and NOAA predictions for Station: ' + str(station_id))
# plt.xlabel('Hours since ' + str(prediction_t0) + '(GMT)')

constituent = [c.name for c in my_tide.model['constituent']]
constituent_df = pd.DataFrame(my_tide.model, index=constituent).drop('constituent', axis=1)
constituent_df.sort_values('amplitude', ascending=False).head(30) # Sorts by constituent with largest amplitude

print('Form number %s, the tide is %s.' %(my_tide.form_number()[0], my_tide.classify()))

#%% API for Boston Data

#Long Beach Airport station
station_id = '8443970'

moda_strt = '0101'
moda_end = '1231'

dates_levs = []
dates_Boston = []
elevs_Boston = []

for year in range(1946, 2020): # one year back from end year
    year = str(year)
    # nextyear = str(int(year)+1)
    print('working on year '+year)
    
    #make the api call
    # r = requests.get('https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&datatypeid=TAVG&limit=1000&stationid=GHCND:USW00023129&startdate='+year+'-01-01&enddate='+year+'-12-31', headers={'token':Token})
    r = requests.get('https://tidesandcurrents.noaa.gov/api/datagetter?product=hourly_height&application&application=NOS.COOPS.TAC.WL&begin_date='+year+moda_strt+'&end_date='+year+moda_end+'&datum=NAVD&station=8443970&time_zone=GMT&units=metric&format=json')   #load the api response as a json
    d = json.loads(r.text)
    #get all items in the response
    dates_levs = [item for item in d['data']]
    #get the date field from all readings
    dates_Boston += [item['t'] for item in dates_levs]
    #get the elev field from all readings
    elevs_Boston += [item['v'] for item in dates_levs]
    
#fill unverified levels with nans
elevs_Boston = [np.nan if x == '' else x for x in elevs_Boston]

#initialize dataframe
df_elevs_Boston = pd.DataFrame()

#populate date and average temperature fields (cast string date to datetime and convert temperature from tenths of Celsius to Fahrenheit)
df_elevs_Boston['datetime'] = [datetime.strptime(d, "%Y-%m-%d %H:%M") for d in dates_Boston]
df_elevs_Boston['Verified NAVD Water Levels'] = [float(v) for v in elevs_Boston]
plt.scatter(df_elevs_Boston['datetime'],df_elevs_Boston['Verified NAVD Water Levels'])

dates_Boston_hourly = (mdates.date2num(df_elevs_Boston['datetime'])-mdates.date2num(df_elevs_Boston['datetime'].iloc[0]))*24

plt.figure()
plt.scatter(dates_Boston_hourly,df_elevs_Boston['Verified NAVD Water Levels'])
idx_Boston = np.isfinite(dates_Boston_hourly) & np.isfinite(df_elevs_Boston['Verified NAVD Water Levels'])
z_Boston = np.polyfit(dates_Boston_hourly[idx_Boston], df_elevs_Boston['Verified NAVD Water Levels'].loc[idx_Boston], 2)
p_Boston = np.poly1d(z_Boston)
polyX_Boston = np.linspace(dates_Boston_hourly.min(), dates_Boston_hourly.max(), 100)

pylab.plot(polyX_Boston,p_Boston(polyX_Boston),"green", label='Boston Trend')

#%% Save
df_elevs_Boston.to_csv(os.path.join(data_dir, 'Water Level Data', 'NOAA 8443970 Boston MA', 'CO-OPS_Boston_8443970_wl_hourly_111946_12312019.csv'), index = False)

#%% Linear regression of Difference in maximum HR level and level at oceanside max vs. oceanside max

# sns.distplot(HRside_diff_maxmatch)

# X = oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"].values.reshape(-1,1)
# y = HRside_diff_maxmatch.values.reshape(-1,1)
# X_train, X_test, y_train, y_test = train_test_split(X[idx_HRlagrise], y[idx_HRlagrise], test_size=0.2, random_state=0)

# regressor = LinearRegression()  
# regressor.fit(X_train, y_train) #training the algorithm

# #To retrieve the intercept:
# print(regressor.intercept_)
# #For retrieving the slope:
# print(regressor.coef_)

# y_pred = regressor.predict(X_test)

# df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
# df

# # df1 = df.head(25)
# # df1.plot(kind='bar',figsize=(16,10))
# # plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# # plt.show()

# plt.scatter(X_test, y_test,  color='gray')
# plt.plot(X_test, y_pred, color='red', linewidth=2)
# plt.show()

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#%% Thesis Plots

fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
(ax1, ax2), (ax3, ax4) = axs
# fig.suptitle('Sharing X_test per column, y_test per row')
oceanside_max_HR_match_res_df.plot.scatter(x="Max Gage height, m, Ocean side", y="Gage height, m, HR side", marker="1", ax=ax1)
oceanside_max_HR_match_res_df.plot.scatter(x="Max Gage height, m, Ocean side", y="Resulting Max Gage height, m, HR side", marker="2", color='Green', ax=ax1)
ax1.plot(polyX_HRfit,p_HRmatch(polyX_HRfit),"c")
ax1.plot(polyX_HRfit,p_HRmax(polyX_HRfit),"LightGreen")
oceanside_max_HR_match_res_df.plot.scatter(x="datetime", y="Gage height, m, HR side", marker="1", label='HR Levels at WH Peaks', ax=ax2)
oceanside_max_HR_match_res_df.plot.scatter(x="datetime", y="Resulting Max Gage height, m, HR side", marker="2", color='Green', label='Resultant HR Peak Levels', ax=ax2)
ax2.plot(polyX_HRfit_dates,p_HRmatch_dates(polyX_HRfit_dates),"c")
ax2.plot(polyX_HRfit_dates,p_HRmax_dates(polyX_HRfit_dates),"LightGreen")
ax2.legend(loc='lower right', fontsize=22)
ax3.scatter(oceanside_max_HR_match_res_df["Max Gage height, m, Ocean side"], HRside_diff_maxmatch,  marker="|", color='gray')
ax3.plot(polyX_HRlagfit,p_HRlagrise(polyX_HRlagfit), color='red', linewidth=2)
ax4.scatter(mdates.date2num(max_dates_ocean), HRside_diff_maxmatch, marker="|", color='purple')
ax4.plot(polyX_HRlagfit_date,p_HRlagrise_date(polyX_HRlagfit_date), 'violet')
    
ax1.set_ylabel('Gage Height, \n Herring River (HR) \n [m NAVD88]', fontsize=22)
ax3.set_ylabel('Dike-Delayed \n Peak Rise, \n Herring River (HR) \n [$\Delta$m NAVD88]', fontsize=22)
ax3.set_xlabel('Maximum Gage Height, \n Wellfleet Harbor (WH) \n [m NAVD88]', fontsize=22)
ax4.set_xlabel('Date [Year-Month]', fontsize=22, labelpad=15)

loc = mdates.AutoDateLocator()
ax4.xaxis.set_major_locator(loc)
ax4.xaxis.set_minor_locator(loc)
ax4.xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
ax4.tick_params(labelrotation=45)

plt.gcf().subplots_adjust(bottom=0.25)

# Put picture of dike before plots.

# A longer time series may show a more representative trend in the resulting increase in levels in the 
# HR based on increasing maximum sea level. The best way to characterize how far the resulting increase
# deviates from a linear trend is to model the system as effectively as possible, varying levels outside the 
# dike 

# nomenclature: outside the dike means the wellfleet harbor, inside means the HR.


