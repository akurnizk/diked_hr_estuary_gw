# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:59:14 2019

@author: akurnizk
"""


# import utm
import csv
import math
# import flopy
import sys,os
import calendar
import dateutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns; sns.set()
mpl.rc('xtick', labelsize=22)     
mpl.rc('ytick', labelsize=22)
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import flopy.utils.binaryfile as bf

cgw_code_dir = 'E:\Python KMB - CGW' # Location of BitBucket folder containing cgw folder
sys.path.insert(0,cgw_code_dir)

from matplotlib import pylab
from scipy.io import loadmat
# from shapely.geometry import Point
from datetime import datetime, time, timedelta
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

# Assign name and create modflow model object

work_dir = os.path.join('E:\Herring Models\Seasonal')
data_dir = os.path.join('E:\Data')

mean_sea_level = 0.843 # Datum in meters at closest NOAA station (8447435), Chatham, Lydia Cove MA
# https://tidesandcurrents.noaa.gov/datums.html?units=1&epoch=0&id=8447435&name=Chatham%2C+Lydia+Cove&state=MA
# use this value for the right boundary
# use data from sonic recorder on ocean side of dike for left boundary

#%% Loading Information from HR Dike Sensors

with open(os.path.join(data_dir,"Surface Water Level Data","USGS 011058798 Herring R at Chequessett Neck Rd","Gage height_ft.txt")) as f:
    reader = csv.reader(f, delimiter="\t")
    HR_dike_all_levels = list(reader)

HR_dike_oceanside_levels = []
HR_dike_HRside_levels = []
for line in range(len(HR_dike_all_levels)-31):
    HR_dike_oceanside_levels.append([HR_dike_all_levels[line+31][2],HR_dike_all_levels[line+31][4]])
    HR_dike_HRside_levels.append([HR_dike_all_levels[line+31][2],HR_dike_all_levels[line+31][6]])

HR_dike_oceanside_levels = np.array(HR_dike_oceanside_levels)
HR_dike_HRside_levels = np.array(HR_dike_HRside_levels)

#%% Dike Levels
"""
Ocean side of dike
"""
# date2num returns Number of days (fraction part represents hours, minutes, seconds, ms) since 0001-01-01 00:00:00 UTC, plus one.
x_oceanside, y_oceanside = HR_dike_oceanside_levels.T

dates_oceanside = [dateutil.parser.parse(x) for x in x_oceanside]
x_oceanside_datenum = mdates.date2num(dates_oceanside)
y_oceanside[np.where(y_oceanside == '')] = np.nan
y_oceanside = y_oceanside.astype(np.float)*0.3048 # feet to meters

pylab.plot(x_oceanside_datenum, y_oceanside, 'o', markersize=1)
idx_oceanside = np.isfinite(x_oceanside_datenum) & np.isfinite(y_oceanside)
z_oceanside = np.polyfit(x_oceanside_datenum[idx_oceanside], y_oceanside[idx_oceanside], 1)
p_oceanside = np.poly1d(z_oceanside)

polyX_oceanside = np.linspace(x_oceanside_datenum.min(), x_oceanside_datenum.max(), 100)
pylab.plot(polyX_oceanside,p_oceanside(polyX_oceanside),"c", label='Mean Sea Level')
# the line equation:
print("y=%.6fx+(%.6f)"%(z_oceanside[0],z_oceanside[1]))

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=18)
plt.ylabel('Elevation (m)', fontsize=16)
plt.legend()

pylab.show()

for i in range(len(y_oceanside)):
    if ~np.isnan(y_oceanside[i]):
        print("Index value for starting date of oceanside data = ",i)
        date_oceanside_start = x_oceanside[i] 
        break
    
for i in reversed(range(len(y_oceanside))):
    if ~np.isnan(y_oceanside[i]):
        print("Index value for ending date of oceanside data = ",i)
        date_oceanside_end = x_oceanside[i] 
        break

sealev_june2015 = z_oceanside[0]*x_oceanside_datenum.min()+z_oceanside[1]
sealev_today = z_oceanside[0]*x_oceanside_datenum.max()+z_oceanside[1]
slr_june2015tojuly2019 = sealev_today-sealev_june2015
slr_oneyear = slr_june2015tojuly2019/1484*365

"""
Herring River side of dike
"""
# date2num returns Number of days (fraction part represents hours, minutes, seconds, ms) since 0001-01-01 00:00:00 UTC, plus one.
x_HRside, y_HRside = HR_dike_HRside_levels.T

dates_HRside = [dateutil.parser.parse(x) for x in x_HRside]
x_HRside_datenum = mdates.date2num(dates_HRside)
y_HRside[np.where(y_HRside == '')] = np.nan
y_HRside[np.where(y_HRside == 'Eqp')] = np.nan # remove equipment failures
y_HRside = y_HRside.astype(np.float)*0.3048 # feet to meters

pylab.plot(x_HRside_datenum, y_HRside, '+', markersize=1)
idx_HRside = np.isfinite(x_HRside_datenum) & np.isfinite(y_HRside)
z_HRside = np.polyfit(x_HRside_datenum[idx_HRside], y_HRside[idx_HRside], 1)
p_HRside = np.poly1d(z_HRside)

polyX_HRside = np.linspace(x_HRside_datenum.min(), x_HRside_datenum.max(), 100)
pylab.plot(polyX_HRside,p_HRside(polyX_HRside),"r", label='Mean River Level')
# the line equation:
print("y=%.6fx+(%.6f)"%(z_HRside[0],z_HRside[1]))

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.legend()

pylab.show()

for i in range(len(y_HRside)):
    if ~np.isnan(y_HRside[i]):
        print(i)
        date_HRside_start = x_HRside[i] 
        break
    
for i in reversed(range(len(y_HRside))):
    if ~np.isnan(y_HRside[i]):
        print(i)
        date_HRside_end = x_HRside[i] 
        break

rivlev_june2015 = z_HRside[0]*x_HRside_datenum.min()+z_HRside[1]
rivlev_today = z_HRside[0]*x_HRside_datenum.max()+z_HRside[1]
rlr_june2015tojuly2019 = rivlev_today-rivlev_june2015
rlr_oneyear = rlr_june2015tojuly2019/1484*365

difinmeans_june2015 = sealev_june2015 - rivlev_june2015
difinmeans_july2019 = sealev_today - rivlev_today

#%% Dike Discharges
"""
Discharge through dike
Measurements are taken every 5 minutes
Filtering takes a 1 hour average
"""
with open(os.path.join(data_dir,"Surface Water Level Data","USGS 011058798 Herring R at Chequessett Neck Rd","Discharge_cfs_Discharge.txt")) as f:
    reader = csv.reader(f, delimiter="\t")
    HR_dike_all_discharge = list(reader)

HR_dike_discharge = []
HR_dike_discharge_filtered = []
for line in range(len(HR_dike_all_discharge)-30):
    HR_dike_discharge.append([HR_dike_all_discharge[line+30][2],HR_dike_all_discharge[line+30][4]])
    HR_dike_discharge_filtered.append([HR_dike_all_discharge[line+30][2],HR_dike_all_discharge[line+30][6]])

HR_dike_discharge = np.array(HR_dike_discharge)
HR_dike_discharge_filtered = np.array(HR_dike_discharge_filtered)

x_discharge, y_discharge = HR_dike_discharge.T
x_discharge_filtered, y_discharge_filtered = HR_dike_discharge_filtered.T

dates_discharge = [dateutil.parser.parse(x) for x in x_discharge]
dates_discharge_filtered = [dateutil.parser.parse(x) for x in x_discharge_filtered]
x_discharge_datenum = mdates.date2num(dates_discharge)
x_discharge_filtered_datenum = mdates.date2num(dates_discharge_filtered)

y_discharge[np.where(y_discharge == '')] = np.nan
y_discharge_filtered[np.where(y_discharge_filtered == '')] = np.nan

y_discharge = y_discharge.astype(np.float)*0.028316847 # cfs to cms
y_discharge_filtered = y_discharge_filtered.astype(np.float)*0.028316847 # cfs to cms

# Plotting
plt.figure()
pylab.plot(x_discharge_datenum, y_discharge, '+', markersize=1)
pylab.plot(x_discharge_filtered_datenum, y_discharge_filtered, 'o', markersize=1)

# Trendline (eliminates no-data points from consideration)
idx_discharge = np.isfinite(x_discharge_datenum) & np.isfinite(y_discharge)
idx_discharge_filtered = np.isfinite(x_discharge_filtered_datenum) & np.isfinite(y_discharge_filtered)
z_discharge = np.polyfit(x_discharge_datenum[idx_discharge], y_discharge[idx_discharge], 1)
z_discharge_filtered = np.polyfit(x_discharge_filtered_datenum[idx_discharge_filtered], y_discharge_filtered[idx_discharge_filtered], 1)
p_discharge = np.poly1d(z_discharge)
p_discharge_filtered = np.poly1d(z_discharge_filtered)

polyX_discharge = np.linspace(x_discharge_datenum.min(), x_discharge_datenum.max(), 100)
polyX_discharge_filtered = np.linspace(x_discharge_filtered_datenum.min(), x_discharge_filtered_datenum.max(), 100)
pylab.plot(polyX_discharge, p_discharge(polyX_discharge),"g", label='Mean Unfiltered Discharge (cms)')
pylab.plot(polyX_discharge_filtered, p_discharge_filtered(polyX_discharge_filtered), "m", label='Mean Filtered Discharge (cms)')
# the line equations:
print("Unfiltered, "+("y=%.6fx+(%.6f)"%(z_discharge[0],z_discharge[1])))
print("Filtered, "+("y=%.6fx+(%.6f)"%(z_discharge_filtered[0],z_discharge_filtered[1])))

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.gca().set_ylim([-0.4,0.2]) # set the y-axis limits
plt.xlabel('Date', fontsize=18)
plt.ylabel('Discharge (cms), Elevation (m)', fontsize=16)

# set x bounds to discharge times for sea and river level polyfits
polyX_HRside = np.linspace(x_discharge_datenum.min(), x_discharge_datenum.max(), 100)
polyX_oceanside = np.linspace(x_discharge_datenum.min(), x_discharge_datenum.max(), 100)

pylab.plot(polyX_discharge,p_HRside(polyX_discharge),"r", label='Mean River Level (m)')
pylab.plot(polyX_discharge,p_oceanside(polyX_discharge),"c", label='Mean Sea Level (m)')

plt.legend()

pylab.show()

for i in range(len(y_discharge)):
    if ~np.isnan(y_discharge[i]):
        date_discharge_start = x_discharge[i]
        print(i, date_discharge_start)
        break
    
for i in reversed(range(len(y_discharge))):
    if ~np.isnan(y_discharge[i]):       
        date_discharge_end = x_discharge[i]
        print(i, date_discharge_end)
        break

# Max and min of trendlines
discharge_june2015 = z_discharge[0]*x_discharge_datenum.min()+z_discharge[1]
discharge_oct2017 = z_discharge[0]*x_discharge_datenum.max()+z_discharge[1]
change_in_discharge_june2015tooct2017 = discharge_june2015-discharge_oct2017
change_in_discharge_oneyear = change_in_discharge_june2015tooct2017/(x_discharge_datenum[-1]-x_discharge_datenum[0])*365
print("Unfiltered discharge is decreasing by ", ("%.3f"%(change_in_discharge_oneyear)), "cms per year.")
print("Unfiltered discharge goes from an average of ", ("%.3f"%(discharge_june2015)), "cms in June 2015 (into ocean)")
print("to ", ("%.3f"%(discharge_oct2017)), "cms in October 2017 (into river).")

discharge_filtered_june2015 = z_discharge_filtered[0]*x_discharge_filtered_datenum.min()+z_discharge_filtered[1]
discharge_filtered_oct2017 = z_discharge_filtered[0]*x_discharge_filtered_datenum.max()+z_discharge_filtered[1]
change_in_filt_discharge_june2015tooct2017 = discharge_filtered_june2015-discharge_filtered_oct2017
change_in_filt_discharge_oneyear = change_in_filt_discharge_june2015tooct2017/(x_discharge_filtered_datenum[-1]-x_discharge_filtered_datenum[0])*365
print("Filtered discharge is decreasing by ", ("%.3f"%(change_in_filt_discharge_oneyear)), "cms per year.")
print("Filtered discharge goes from an average of ", ("%.3f"%(discharge_filtered_june2015)), "cms in June 2015 (into ocean)")
print("to ", ("%.3f"%(discharge_filtered_oct2017)), "cms in October 2017 (into river).")

#%% Mean High and Low Dike Levels, Hourly
"""
Mean High and Low Tides at Dike (to determine changes in amplitude and amplitude decay from ocean to river)
"""
# Need to remove nan vals and reduce measurement frequency
nan_indices_oceanside = []
for i in range(len(y_oceanside)):
    if np.isnan(y_oceanside[i]):
        nan_indices_oceanside.append(i)

nan_indices_HRside = []
for i in range(len(y_HRside)):
    if np.isnan(y_HRside[i]):
        nan_indices_HRside.append(i)

y_oceanside_nonans = y_oceanside.tolist()
x_oceanside_datenum_nonans = x_oceanside_datenum.tolist()
for index in sorted(nan_indices_oceanside, reverse=True):
    del y_oceanside_nonans[index]
    del x_oceanside_datenum_nonans[index]
y_oceanside_nonans = np.array(y_oceanside_nonans)
x_oceanside_datenum_nonans = np.array(x_oceanside_datenum_nonans) 

y_HRside_nonans = y_HRside.tolist()
x_HRside_datenum_nonans = x_HRside_datenum.tolist()
for index in sorted(nan_indices_HRside, reverse=True):
    del y_HRside_nonans[index]
    del x_HRside_datenum_nonans[index]
y_HRside_nonans = np.array(y_HRside_nonans)
x_HRside_datenum_nonans = np.array(x_HRside_datenum_nonans)

# convert numbered datetime back to standard (allows determination of minutes)
dates_oceanside_nonans = mdates.num2date(x_oceanside_datenum_nonans)
dates_HRside_nonans = mdates.num2date(x_HRside_datenum_nonans)

hourly_indices_oceanside = []
for i in range(len(dates_oceanside_nonans)):
    if dates_oceanside_nonans[i].minute == 0: # minute is only zero on the hour
        hourly_indices_oceanside.append(i)

hourly_indices_HRside = []
for i in range(len(dates_HRside_nonans)):
    if dates_HRside_nonans[i].minute == 0: # minute is only zero on the hour
        hourly_indices_HRside.append(i)
        
y_oceanside_hourly = []
x_oceanside_datenum_hourly = []
for index in sorted(hourly_indices_oceanside):
    y_oceanside_hourly.append(y_oceanside_nonans[index])
    x_oceanside_datenum_hourly.append(x_oceanside_datenum_nonans[index])
y_oceanside_hourly = np.array(y_oceanside_hourly)
x_oceanside_datenum_hourly = np.array(x_oceanside_datenum_hourly)
    
y_HRside_hourly = []
x_HRside_datenum_hourly = []
for index in sorted(hourly_indices_HRside):
    y_HRside_hourly.append(y_HRside_nonans[index])
    x_HRside_datenum_hourly.append(x_HRside_datenum_nonans[index])
y_HRside_hourly = np.array(y_HRside_hourly)
x_HRside_datenum_hourly = np.array(x_HRside_datenum_hourly)

# plot hourly levels    
pylab.plot(x_oceanside_datenum_hourly, y_oceanside_hourly, 'o', markersize=1)
pylab.plot(x_HRside_datenum_hourly, y_HRside_hourly, 'o', markersize=1)
idx_oceanside_hourly = np.isfinite(x_oceanside_datenum_hourly) & np.isfinite(y_oceanside_hourly)
idx_HRside_hourly = np.isfinite(x_HRside_datenum_hourly) & np.isfinite(y_HRside_hourly)
z_oceanside_hourly = np.polyfit(x_oceanside_datenum_hourly[idx_oceanside_hourly], y_oceanside_hourly[idx_oceanside_hourly], 1)
z_HRside_hourly = np.polyfit(x_HRside_datenum_hourly[idx_HRside_hourly], y_HRside_hourly[idx_HRside_hourly], 1)
p_oceanside_hourly = np.poly1d(z_oceanside_hourly)
p_HRside_hourly = np.poly1d(z_HRside_hourly)

polyX_oceanside_hourly = np.linspace(x_oceanside_datenum_hourly.min(), x_oceanside_datenum_hourly.max(), 100)
polyX_HRside_hourly = np.linspace(x_HRside_datenum_hourly.min(), x_HRside_datenum_hourly.max(), 100)
pylab.plot(polyX_oceanside_hourly,p_oceanside_hourly(polyX_oceanside_hourly),"c", label='Mean Sea Level')
pylab.plot(polyX_HRside_hourly,p_HRside_hourly(polyX_HRside_hourly),"r", label='Mean River Level')
# the line equation:
print("y=%.6fx+(%.6f)"%(z_oceanside_hourly[0],z_oceanside_hourly[1]))
print("y=%.6fx+(%.6f)"%(z_HRside_hourly[0],z_HRside_hourly[1]))

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=18)
plt.ylabel('Elevation (m)', fontsize=16)
plt.legend()

pylab.show()

# Concatenate dates and levels
HR_dike_oceanside_levels_hourly = np.vstack((x_oceanside_datenum_hourly, y_oceanside_hourly)).T
HR_dike_HRside_levels_hourly = np.vstack((x_HRside_datenum_hourly, y_HRside_hourly)).T
    
# max and min vals for tides
HR_dike_oceanside_maxlevels = []
HR_dike_oceanside_minlevels = []
for i in range(len(HR_dike_oceanside_levels_hourly)-2): # length of oceanside and HRside level arrays are the same
    if (HR_dike_oceanside_levels_hourly[i+1][1] > HR_dike_oceanside_levels_hourly[i][1]) & (HR_dike_oceanside_levels_hourly[i+2][1] < HR_dike_oceanside_levels_hourly[i+1][1]) & (HR_dike_oceanside_levels_hourly[i+1][1] > p_oceanside(polyX_oceanside).mean()):
        HR_dike_oceanside_maxlevels.append([HR_dike_oceanside_levels_hourly[i+1][0], HR_dike_oceanside_levels_hourly[i+1][1]]) # high tides    
    if (HR_dike_oceanside_levels_hourly[i+1][1] < HR_dike_oceanside_levels_hourly[i][1]) & (HR_dike_oceanside_levels_hourly[i+2][1] > HR_dike_oceanside_levels_hourly[i+1][1]) & (HR_dike_oceanside_levels_hourly[i+1][1] < p_oceanside(polyX_oceanside).mean()):
        HR_dike_oceanside_minlevels.append([HR_dike_oceanside_levels_hourly[i+1][0], HR_dike_oceanside_levels_hourly[i+1][1]])
HR_dike_oceanside_maxlevels = np.array(HR_dike_oceanside_maxlevels)
HR_dike_oceanside_minlevels = np.array(HR_dike_oceanside_minlevels) 
    
HR_dike_HRside_maxlevels = []
HR_dike_HRside_minlevels = []    
for i in range(len(HR_dike_HRside_levels_hourly)-2): # length of oceanside and HRside level arrays are the same
    if (HR_dike_HRside_levels_hourly[i+1][1] > HR_dike_HRside_levels_hourly[i][1]) & (HR_dike_HRside_levels_hourly[i+2][1] < HR_dike_HRside_levels_hourly[i+1][1]) & (HR_dike_HRside_levels_hourly[i+1][1] > p_HRside(polyX_HRside).mean()):
        HR_dike_HRside_maxlevels.append([HR_dike_HRside_levels_hourly[i+1][0], HR_dike_HRside_levels_hourly[i+1][1]]) # high tides    
    if (HR_dike_HRside_levels_hourly[i+1][1] < HR_dike_HRside_levels_hourly[i][1]) & (HR_dike_HRside_levels_hourly[i+2][1] > HR_dike_HRside_levels_hourly[i+1][1]) & (HR_dike_HRside_levels_hourly[i+1][1] < p_HRside(polyX_HRside).mean()):
        HR_dike_HRside_minlevels.append([HR_dike_HRside_levels_hourly[i+1][0], HR_dike_HRside_levels_hourly[i+1][1]])    
HR_dike_HRside_maxlevels = np.array(HR_dike_HRside_maxlevels)
HR_dike_HRside_minlevels = np.array(HR_dike_HRside_minlevels) 

#%% Mean High and Low Dike Levels, Oceanside and HRside
"""
Ocean side of dike mins and maxes (hourly time steps)
"""
x_oceanside_datenum_maxlevels, y_oceanside_maxlevels = HR_dike_oceanside_maxlevels.T
x_oceanside_datenum_minlevels, y_oceanside_minlevels = HR_dike_oceanside_minlevels.T

plt.figure()
pylab.plot(x_oceanside_datenum_maxlevels, y_oceanside_maxlevels, 'o', markersize=1)
pylab.plot(x_oceanside_datenum_minlevels, y_oceanside_minlevels, 'o', markersize=1)
idx_oceanside_max = np.isfinite(x_oceanside_datenum_maxlevels) & np.isfinite(y_oceanside_maxlevels)
idx_oceanside_min = np.isfinite(x_oceanside_datenum_minlevels) & np.isfinite(y_oceanside_minlevels)
z_oceanside_max = np.polyfit(x_oceanside_datenum_maxlevels[idx_oceanside_max], y_oceanside_maxlevels[idx_oceanside_max], 1)
z_oceanside_min = np.polyfit(x_oceanside_datenum_minlevels[idx_oceanside_min], y_oceanside_minlevels[idx_oceanside_min], 1)
p_oceanside_max = np.poly1d(z_oceanside_max)
p_oceanside_min = np.poly1d(z_oceanside_min)

polyX_oceanside_max = np.linspace(x_oceanside_datenum_maxlevels.min(), x_oceanside_datenum_maxlevels.max(), 100)
polyX_oceanside_min = np.linspace(x_oceanside_datenum_minlevels.min(), x_oceanside_datenum_minlevels.max(), 100)
pylab.plot(polyX_oceanside_max,p_oceanside_max(polyX_oceanside_max),"c", label='Mean High Sea Level')
pylab.plot(polyX_oceanside_min,p_oceanside_min(polyX_oceanside_min),"m", label='Mean Low Sea Level')
# the line equation:
print("y=%.6fx+(%.6f)"%(z_oceanside_max[0],z_oceanside_max[1]))
print("y=%.6fx+(%.6f)"%(z_oceanside_min[0],z_oceanside_min[1]))

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=18)
plt.ylabel('Elevation (m)', fontsize=16)
plt.legend()

pylab.show()

# Max and min of trendlines
sealev_oct2017_max = z_oceanside_max[0]*x_oceanside_datenum_maxlevels.min()+z_oceanside_max[1]
sealev_oct2017_min = z_oceanside_min[0]*x_oceanside_datenum_minlevels.min()+z_oceanside_min[1]
sealev_june2019_max = z_oceanside_max[0]*x_oceanside_datenum_maxlevels.max()+z_oceanside_max[1]
sealev_june2019_min = z_oceanside_min[0]*x_oceanside_datenum_minlevels.max()+z_oceanside_min[1]
slrhigh_oct2017tojune2019 = sealev_june2019_max-sealev_oct2017_max
slrlow_oct2017tojune2019 = sealev_june2019_min-sealev_oct2017_min
slrhigh_oneyear = slrhigh_oct2017tojune2019/(x_oceanside_datenum_maxlevels[-1]-x_oceanside_datenum_maxlevels[0])*365
slrlow_oneyear = slrlow_oct2017tojune2019/(x_oceanside_datenum_minlevels[-1]-x_oceanside_datenum_minlevels[0])*365
    
print("Maximum mean sea level on the ocean side of the dike is increasing by ", ("%.3f"%(slrhigh_oneyear)), "m per year.")
print("Maximum mean sea level at that location goes from an average of ", ("%.3f"%(sealev_oct2017_max)), "m in October 2017")
print("to ", ("%.3f"%(sealev_june2019_max)), "m in June 2019.")

print("Minimum mean sea level on the ocean side of the dike is increasing by ", ("%.3f"%(slrlow_oneyear)), "m per year.")
print("Minimum mean sea level at that location goes from an average of ", ("%.3f"%(sealev_oct2017_min)), "m in October 2017")
print("to ", ("%.3f"%(sealev_june2019_min)), "m in June 2019. This mean is primarily influenced by the Herring River.")

# Amplitudes and ranges
amp_oceanside = p_oceanside_max(polyX_oceanside_max) - p_oceanside(polyX_oceanside)
amp_oceanside_avg = amp_oceanside.mean()
range_oceanside = p_oceanside_max(polyX_oceanside_max) - p_oceanside_min(polyX_oceanside_min)
range_oceanside_avg = range_oceanside.mean()

print("Average tidal range between October 2017 and June 2019 on the ocean side of the dike is ", ("%.3f"%(range_oceanside_avg)), " m.")

"""
HR side of dike mins and maxes (hourly time steps)
"""
x_HRside_datenum_maxlevels, y_HRside_maxlevels = HR_dike_HRside_maxlevels.T
x_HRside_datenum_minlevels, y_HRside_minlevels = HR_dike_HRside_minlevels.T

pylab.plot(x_HRside_datenum_maxlevels, y_HRside_maxlevels, 'o', markersize=1)
pylab.plot(x_HRside_datenum_minlevels, y_HRside_minlevels, 'o', markersize=1)
idx_HRside_max = np.isfinite(x_HRside_datenum_maxlevels) & np.isfinite(y_HRside_maxlevels)
idx_HRside_min = np.isfinite(x_HRside_datenum_minlevels) & np.isfinite(y_HRside_minlevels)
z_HRside_max = np.polyfit(x_HRside_datenum_maxlevels[idx_HRside_max], y_HRside_maxlevels[idx_HRside_max], 1)
z_HRside_min = np.polyfit(x_HRside_datenum_minlevels[idx_HRside_min], y_HRside_minlevels[idx_HRside_min], 1)
p_HRside_max = np.poly1d(z_HRside_max)
p_HRside_min = np.poly1d(z_HRside_min)

polyX_HRside_max = np.linspace(x_HRside_datenum_maxlevels.min(), x_HRside_datenum_maxlevels.max(), 100)
polyX_HRside_min = np.linspace(x_HRside_datenum_minlevels.min(), x_HRside_datenum_minlevels.max(), 100)
pylab.plot(polyX_HRside_max,p_HRside_max(polyX_HRside_max),"g", label='Mean High River Level')
pylab.plot(polyX_HRside_min,p_HRside_min(polyX_HRside_min),"r", label='Mean Low River Level')
# the line equation:
print("y=%.6fx+(%.6f)"%(z_HRside_max[0],z_HRside_max[1]))
print("y=%.6fx+(%.6f)"%(z_HRside_min[0],z_HRside_min[1]))

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=18)
plt.ylabel('Elevation (m)', fontsize=16)
plt.legend()

pylab.show()

print("There are ", y_oceanside_maxlevels.size, "data points for maximum tidal stage on the ocean side of the dike.")
print("There are ", y_oceanside_minlevels.size, "data points for minimum tidal stage on the ocean side of the dike.")
print("There are ", y_HRside_maxlevels.size, "data points for maximum stage on the river side of the dike.")
print("There are ", y_HRside_minlevels.size, "data points for minimum stage on the river side of the dike.")

# Max and min of trendlines
rivlev_june2015_max = z_HRside_max[0]*x_HRside_datenum_maxlevels.min()+z_HRside_max[1]
rivlev_june2015_min = z_HRside_min[0]*x_HRside_datenum_minlevels.min()+z_HRside_min[1]
rivlev_july2019_max = z_HRside_max[0]*x_HRside_datenum_maxlevels.max()+z_HRside_max[1]
rivlev_july2019_min = z_HRside_min[0]*x_HRside_datenum_minlevels.max()+z_HRside_min[1]
rlrhigh_june2015tojuly2019 = rivlev_july2019_max-rivlev_june2015_max
rlrlow_june2015tojuly2019 = rivlev_july2019_min-rivlev_june2015_min
rlrhigh_oneyear = rlrhigh_june2015tojuly2019/(x_HRside_datenum_maxlevels[-1]-x_HRside_datenum_maxlevels[0])*365
rlrlow_oneyear = rlrlow_june2015tojuly2019/(x_HRside_datenum_minlevels[-1]-x_HRside_datenum_minlevels[0])*365
    
print("Maximum mean river level on the river side of the dike is increasing by ", ("%.3f"%(rlrhigh_oneyear)), "m per year.")
print("Maximum mean river level at that location goes from an average of ", ("%.3f"%(rivlev_june2015_max)), "m in June 2015")
print("to ", ("%.3f"%(rivlev_july2019_max)), "m in July 2019.")

print("Minimum mean river level on the river side of the dike is increasing by ", ("%.3f"%(rlrlow_oneyear)), "m per year.")
print("Minimum mean river level at that location goes from an average of ", ("%.3f"%(rivlev_june2015_min)), "m in June 2015")
print("to ", ("%.3f"%(rivlev_july2019_min)), "m in July 2019.")

# Amplitudes and ranges
amp_HRside = p_HRside_max(polyX_HRside_max) - p_HRside(polyX_HRside)
amp_HRside_avg = amp_HRside.mean()
range_HRside = p_HRside_max(polyX_HRside_max) - p_HRside_min(polyX_HRside_min)
range_HRside_avg = range_HRside.mean()

print("Average tidal range between June 2015 and July 2019 on the river side of the dike is ", ("%.3f"%(range_HRside_avg)), " m.")

range_HRside_june2015 = p_HRside_max(polyX_HRside_max).min() - p_HRside_min(polyX_HRside_min).min()
range_HRside_july2019 = p_HRside_max(polyX_HRside_max).max() - p_HRside_min(polyX_HRside_min).max()

range_oceanside_june2015 = p_oceanside_max(polyX_HRside_max).min() - p_oceanside_min(polyX_HRside_min).min()
range_oceanside_july2019 = p_oceanside_max(polyX_HRside_max).max() - p_oceanside_min(polyX_HRside_min).max()

#%% NPS CTD Sensor Data

with open(os.path.join(data_dir,"Surface Water Level Data","NPS CTD Sensors","Water_Elevation,_NAVD88-File_Import-07-17-2019_17-29.csv")) as f:
    reader = csv.reader(f, delimiter=",")
    NPS_CTD_all_levels = list(reader)

HighToss_levels = []
MillCreek_levels = []
CNRUS_levels = []
DogLeg_levels = []
OldSaw_levels = []
for line in range(len(NPS_CTD_all_levels)-3):
    HighToss_levels.append([NPS_CTD_all_levels[line+3][0],NPS_CTD_all_levels[line+3][2]])
    MillCreek_levels.append([NPS_CTD_all_levels[line+3][0],NPS_CTD_all_levels[line+3][3]])
    CNRUS_levels.append([NPS_CTD_all_levels[line+3][0],NPS_CTD_all_levels[line+3][4]])
    DogLeg_levels.append([NPS_CTD_all_levels[line+3][0],NPS_CTD_all_levels[line+3][5]])
    OldSaw_levels.append([NPS_CTD_all_levels[line+3][0],NPS_CTD_all_levels[line+3][6]])

HighToss_levels = np.array(HighToss_levels)
MillCreek_levels = np.array(MillCreek_levels)
CNRUS_levels = np.array(CNRUS_levels)
DogLeg_levels = np.array(DogLeg_levels)
OldSaw_levels = np.array(OldSaw_levels)

#%% NPS CTD CNR U/S Salinity
with open(os.path.join(data_dir,"Salinity and Conductance Data","CNRUS_ctd_salinity.csv")) as f:
    reader = csv.reader(f, delimiter=",")
    NPS_CTD_all_salinity = list(reader)

CNRUS_salinity = []
for line in range(len(NPS_CTD_all_salinity)-1):
    CNRUS_salinity.append([NPS_CTD_all_salinity[line+1][0],NPS_CTD_all_salinity[line+1][1]])

CNRUS_salinity = np.array(CNRUS_salinity)

x_CNRUS_sal, y_CNRUS_sal = CNRUS_salinity.T

dates_CNRUS_sal = [dateutil.parser.parse(x) for x in x_CNRUS_sal]
x_CNRUS_sal_datenum = mdates.date2num(dates_CNRUS_sal)
y_CNRUS_sal[np.where(y_CNRUS_sal == '')] = np.nan
y_CNRUS_sal = y_CNRUS_sal.astype(np.float)
    
idx_CNRUS_sal = np.isfinite(x_CNRUS_sal_datenum) & np.isfinite(y_CNRUS_sal)
z_CNRUS_sal = np.polyfit(x_CNRUS_sal_datenum[idx_CNRUS_sal], y_CNRUS_sal[idx_CNRUS_sal], 1)
p_CNRUS_sal = np.poly1d(z_CNRUS_sal)
polyX_CNRUS_sal = np.linspace(x_CNRUS_sal_datenum.min(), x_CNRUS_sal_datenum.max(), 100)

nan_indices_CNRUS_sal = []
for i in range(len(y_CNRUS_sal)):
    if np.isnan(y_CNRUS_sal[i]):
        nan_indices_CNRUS_sal.append(i)

y_CNRUS_sal_nonans = y_CNRUS_sal.tolist()
x_CNRUS_sal_datenum_nonans = x_CNRUS_sal_datenum.tolist()
for index in sorted(nan_indices_CNRUS_sal, reverse=True):
    del y_CNRUS_sal_nonans[index]
    del x_CNRUS_sal_datenum_nonans[index]
y_CNRUS_sal_nonans = np.array(y_CNRUS_sal_nonans)
x_CNRUS_sal_datenum_nonans = np.array(x_CNRUS_sal_datenum_nonans) 

dates_CNRUS_sal_nonans = mdates.num2date(x_CNRUS_sal_datenum_nonans)

hourly_indices_CNRUS_sal = []
for i in range(len(dates_CNRUS_sal_nonans)):
    if dates_CNRUS_sal_nonans[i].minute == 0:
        hourly_indices_CNRUS_sal.append(i)

y_CNRUS_sal_hourly = []
x_CNRUS_sal_datenum_hourly = []
for index in sorted(hourly_indices_CNRUS_sal):
    y_CNRUS_sal_hourly.append(y_CNRUS_sal_nonans[index])
    x_CNRUS_sal_datenum_hourly.append(x_CNRUS_sal_datenum_nonans[index])
y_CNRUS_sal_hourly = np.array(y_CNRUS_sal_hourly)
x_CNRUS_sal_datenum_hourly = np.array(x_CNRUS_sal_datenum_hourly)

HR_dike_CNRUS_sal_levels_hourly = np.vstack((x_CNRUS_sal_datenum_hourly, y_CNRUS_sal_hourly)).T

# max and min vals for tides
HR_dike_CNRUS_sal_maxlevels = []
HR_dike_CNRUS_sal_minlevels = []
for i in range(len(HR_dike_CNRUS_sal_levels_hourly)-2): 
    if (HR_dike_CNRUS_sal_levels_hourly[i+1][1] > HR_dike_CNRUS_sal_levels_hourly[i][1]) & (HR_dike_CNRUS_sal_levels_hourly[i+2][1] < HR_dike_CNRUS_sal_levels_hourly[i+1][1]) & (HR_dike_CNRUS_sal_levels_hourly[i+1][1] > p_CNRUS_sal(polyX_CNRUS_sal).mean()):
        HR_dike_CNRUS_sal_maxlevels.append([HR_dike_CNRUS_sal_levels_hourly[i+1][0], HR_dike_CNRUS_sal_levels_hourly[i+1][1]]) # high tides    
    if (HR_dike_CNRUS_sal_levels_hourly[i+1][1] < HR_dike_CNRUS_sal_levels_hourly[i][1]) & (HR_dike_CNRUS_sal_levels_hourly[i+2][1] > HR_dike_CNRUS_sal_levels_hourly[i+1][1]) & (HR_dike_CNRUS_sal_levels_hourly[i+1][1] < p_CNRUS_sal(polyX_CNRUS_sal).mean()):
        HR_dike_CNRUS_sal_minlevels.append([HR_dike_CNRUS_sal_levels_hourly[i+1][0], HR_dike_CNRUS_sal_levels_hourly[i+1][1]])
HR_dike_CNRUS_sal_maxlevels = np.array(HR_dike_CNRUS_sal_maxlevels)
HR_dike_CNRUS_sal_minlevels = np.array(HR_dike_CNRUS_sal_minlevels) 

x_CNRUS_sal_datenum_maxlevels, y_CNRUS_sal_maxlevels = HR_dike_CNRUS_sal_maxlevels.T
x_CNRUS_sal_datenum_minlevels, y_CNRUS_sal_minlevels = HR_dike_CNRUS_sal_minlevels.T

# plots
plt.figure()
pylab.plot(x_CNRUS_sal_datenum_maxlevels, y_CNRUS_sal_maxlevels, 'go', markersize=1)
pylab.plot(x_CNRUS_sal_datenum_minlevels, y_CNRUS_sal_minlevels, 'mo', markersize=1)

# trendlines
idx_CNRUS_sal_max = np.isfinite(x_CNRUS_sal_datenum_maxlevels) & np.isfinite(y_CNRUS_sal_maxlevels)
idx_CNRUS_sal_min = np.isfinite(x_CNRUS_sal_datenum_minlevels) & np.isfinite(y_CNRUS_sal_minlevels)
z_CNRUS_sal_max = np.polyfit(x_CNRUS_sal_datenum_maxlevels[idx_CNRUS_sal_max], y_CNRUS_sal_maxlevels[idx_CNRUS_sal_max], 1)
z_CNRUS_sal_min = np.polyfit(x_CNRUS_sal_datenum_minlevels[idx_CNRUS_sal_min], y_CNRUS_sal_minlevels[idx_CNRUS_sal_min], 1)
p_CNRUS_sal_max = np.poly1d(z_CNRUS_sal_max)
p_CNRUS_sal_min = np.poly1d(z_CNRUS_sal_min)

# plotted trendlines
polyX_CNRUS_sal_max = np.linspace(x_CNRUS_sal_datenum_maxlevels.min(), x_CNRUS_sal_datenum_maxlevels.max(), 100)
polyX_CNRUS_sal_min = np.linspace(x_CNRUS_sal_datenum_minlevels.min(), x_CNRUS_sal_datenum_minlevels.max(), 100)
pylab.plot(polyX_CNRUS_sal_max,p_CNRUS_sal_max(polyX_CNRUS_sal_max),"lightgreen", label='High Tide')
pylab.plot(polyX_CNRUS_sal_min,p_CNRUS_sal_min(polyX_CNRUS_sal_min),"mediumpurple", label='Low Tide')
# the line equation:
print("y=%.6fx+(%.6f)"%(z_CNRUS_sal_max[0],z_CNRUS_sal_max[1]))
print("y=%.6fx+(%.6f)"%(z_CNRUS_sal_min[0],z_CNRUS_sal_min[1]))

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel(r'Date $\left[YYYY-MM\right]$', fontsize=26)
plt.ylabel('Salinity at CNR U/S (ppt)', fontsize=26)
plt.legend(loc='best', fontsize=22)

pylab.show()

#%% Plots of NPS CTD Sensor Data
"""
NPS CTD Data, Plotted: Old Saw -> CNR U/S -> Mill Creek -> Dog Leg -> High Toss
"""
x_HighToss, y_HighToss = HighToss_levels.T
x_MillCreek, y_MillCreek = MillCreek_levels.T
x_CNRUS, y_CNRUS = CNRUS_levels.T
x_DogLeg, y_DogLeg = DogLeg_levels.T
x_OldSaw, y_OldSaw = OldSaw_levels.T

# parse dates and convert to number format, replace blanks with nans
dates_HighToss = [dateutil.parser.parse(x) for x in x_HighToss]
x_HighToss_datenum = mdates.date2num(dates_HighToss)
y_HighToss[np.where(y_HighToss == '')] = np.nan
y_HighToss = y_HighToss.astype(np.float)

dates_MillCreek = [dateutil.parser.parse(x) for x in x_MillCreek]
x_MillCreek_datenum = mdates.date2num(dates_MillCreek)
y_MillCreek[np.where(y_MillCreek == '')] = np.nan
y_MillCreek = y_MillCreek.astype(np.float)

dates_CNRUS = [dateutil.parser.parse(x) for x in x_CNRUS]
x_CNRUS_datenum = mdates.date2num(dates_CNRUS)
y_CNRUS[np.where(y_CNRUS == '')] = np.nan
y_CNRUS = y_CNRUS.astype(np.float)

dates_DogLeg = [dateutil.parser.parse(x) for x in x_DogLeg]
x_DogLeg_datenum = mdates.date2num(dates_DogLeg)
y_DogLeg[np.where(y_DogLeg == '')] = np.nan
y_DogLeg = y_DogLeg.astype(np.float)

dates_OldSaw = [dateutil.parser.parse(x) for x in x_OldSaw]
x_OldSaw_datenum = mdates.date2num(dates_OldSaw)
y_OldSaw[np.where(y_OldSaw == '')] = np.nan
y_OldSaw = y_OldSaw.astype(np.float)

# plot all (default order = blue, orange, green, red, purple)
pylab.plot(x_HighToss_datenum, y_HighToss, 'o', markersize=1)
pylab.plot(x_MillCreek_datenum, y_MillCreek, 'o', markersize=1)
pylab.plot(x_CNRUS_datenum, y_CNRUS, 'o', markersize=1)
pylab.plot(x_DogLeg_datenum, y_DogLeg, 'o', markersize=1)
pylab.plot(x_OldSaw_datenum, y_OldSaw, 'o', markersize=1)

idx_HighToss = np.isfinite(x_HighToss_datenum) & np.isfinite(y_HighToss)
z_HighToss = np.polyfit(x_HighToss_datenum[idx_HighToss], y_HighToss[idx_HighToss], 1)
p_HighToss = np.poly1d(z_HighToss)

idx_MillCreek = np.isfinite(x_MillCreek_datenum) & np.isfinite(y_MillCreek)
z_MillCreek = np.polyfit(x_MillCreek_datenum[idx_MillCreek], y_MillCreek[idx_MillCreek], 1)
p_MillCreek = np.poly1d(z_MillCreek)

idx_CNRUS = np.isfinite(x_CNRUS_datenum) & np.isfinite(y_CNRUS)
z_CNRUS = np.polyfit(x_CNRUS_datenum[idx_CNRUS], y_CNRUS[idx_CNRUS], 1)
p_CNRUS = np.poly1d(z_CNRUS)

idx_DogLeg = np.isfinite(x_DogLeg_datenum) & np.isfinite(y_DogLeg)
z_DogLeg = np.polyfit(x_DogLeg_datenum[idx_DogLeg], y_DogLeg[idx_DogLeg], 1)
p_DogLeg = np.poly1d(z_DogLeg)

idx_OldSaw = np.isfinite(x_OldSaw_datenum) & np.isfinite(y_OldSaw)
z_OldSaw = np.polyfit(x_OldSaw_datenum[idx_OldSaw], y_OldSaw[idx_OldSaw], 1)
p_OldSaw = np.poly1d(z_OldSaw)

polyX_HighToss = np.linspace(x_HighToss_datenum.min(), x_HighToss_datenum.max(), 100)
pylab.plot(polyX_HighToss,p_HighToss(polyX_HighToss),"c", label='Mean High Toss Level')
polyX_MillCreek = np.linspace(x_MillCreek_datenum.min(), x_MillCreek_datenum.max(), 100)
pylab.plot(polyX_MillCreek,p_MillCreek(polyX_MillCreek),"y", label='Mean Mill Creek Level')
polyX_CNRUS = np.linspace(x_CNRUS_datenum.min(), x_CNRUS_datenum.max(), 100)
pylab.plot(polyX_CNRUS,p_CNRUS(polyX_CNRUS),"lime", label='Mean HR near-dike Level')
polyX_DogLeg = np.linspace(x_DogLeg_datenum.min(), x_DogLeg_datenum.max(), 100)
pylab.plot(polyX_DogLeg,p_DogLeg(polyX_DogLeg),"salmon", label='Mean Dog Leg Level')
polyX_OldSaw = np.linspace(x_OldSaw_datenum.min(), x_OldSaw_datenum.max(), 100)
pylab.plot(polyX_OldSaw,p_OldSaw(polyX_OldSaw),"m", label='Mean Old Saw (Wellfleet) Sea Level')
# the line equation:
print("y=%.6fx+(%.6f)"%(z_HighToss[0],z_HighToss[1]))
print("y=%.6fx+(%.6f)"%(z_MillCreek[0],z_MillCreek[1]))
print("y=%.6fx+(%.6f)"%(z_CNRUS[0],z_CNRUS[1]))
print("y=%.6fx+(%.6f)"%(z_DogLeg[0],z_DogLeg[1]))
print("y=%.6fx+(%.6f)"%(z_OldSaw[0],z_OldSaw[1]))

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=18)
plt.ylabel('Elevation (m)', fontsize=16)
plt.legend()

pylab.show()

#%% CTD Data Means

# Need to remove nan vals and reduce measurement frequency for oceanside data
nan_indices_HighToss = []
nan_indices_MillCreek = []
nan_indices_CNRUS = []
nan_indices_DogLeg = []
nan_indices_OldSaw = []
for i in range(len(y_HighToss)): # HighToss, MillCreek, CNRUS, DogLeg, and OldSaw all the same length
    if np.isnan(y_HighToss[i]):
        nan_indices_HighToss.append(i)
    if np.isnan(y_MillCreek[i]):
        nan_indices_MillCreek.append(i)
    if np.isnan(y_CNRUS[i]):
        nan_indices_CNRUS.append(i)
    if np.isnan(y_DogLeg[i]):
        nan_indices_DogLeg.append(i)
    if np.isnan(y_OldSaw[i]):
        nan_indices_OldSaw.append(i)

y_HighToss_nonans = y_HighToss.tolist()
x_HighToss_datenum_nonans = x_HighToss_datenum.tolist()
for index in sorted(nan_indices_HighToss, reverse=True):
    del y_HighToss_nonans[index]
    del x_HighToss_datenum_nonans[index]
y_HighToss_nonans = np.array(y_HighToss_nonans)
x_HighToss_datenum_nonans = np.array(x_HighToss_datenum_nonans) 

y_MillCreek_nonans = y_MillCreek.tolist()
x_MillCreek_datenum_nonans = x_MillCreek_datenum.tolist()
for index in sorted(nan_indices_MillCreek, reverse=True):
    del y_MillCreek_nonans[index]
    del x_MillCreek_datenum_nonans[index]
y_MillCreek_nonans = np.array(y_MillCreek_nonans)
x_MillCreek_datenum_nonans = np.array(x_MillCreek_datenum_nonans) 

y_CNRUS_nonans = y_CNRUS.tolist()
x_CNRUS_datenum_nonans = x_CNRUS_datenum.tolist()
for index in sorted(nan_indices_CNRUS, reverse=True):
    del y_CNRUS_nonans[index]
    del x_CNRUS_datenum_nonans[index]
y_CNRUS_nonans = np.array(y_CNRUS_nonans)
x_CNRUS_datenum_nonans = np.array(x_CNRUS_datenum_nonans) 

y_DogLeg_nonans = y_DogLeg.tolist()
x_DogLeg_datenum_nonans = x_DogLeg_datenum.tolist()
for index in sorted(nan_indices_DogLeg, reverse=True):
    del y_DogLeg_nonans[index]
    del x_DogLeg_datenum_nonans[index]
y_DogLeg_nonans = np.array(y_DogLeg_nonans)
x_DogLeg_datenum_nonans = np.array(x_DogLeg_datenum_nonans) 

y_OldSaw_nonans = y_OldSaw.tolist()
x_OldSaw_datenum_nonans = x_OldSaw_datenum.tolist()
for index in sorted(nan_indices_OldSaw, reverse=True):
    del y_OldSaw_nonans[index]
    del x_OldSaw_datenum_nonans[index]
y_OldSaw_nonans = np.array(y_OldSaw_nonans)
x_OldSaw_datenum_nonans = np.array(x_OldSaw_datenum_nonans) 

# convert numbered datetime back to standard
dates_HighToss_nonans = mdates.num2date(x_HighToss_datenum_nonans)
dates_MillCreek_nonans = mdates.num2date(x_MillCreek_datenum_nonans)
dates_CNRUS_nonans = mdates.num2date(x_CNRUS_datenum_nonans)
dates_DogLeg_nonans = mdates.num2date(x_DogLeg_datenum_nonans)
dates_OldSaw_nonans = mdates.num2date(x_OldSaw_datenum_nonans)

# convert to hourly time intervals
# High Toss
hourly_indices_HighToss = []
for i in range(len(dates_HighToss_nonans)):
    if dates_HighToss_nonans[i].minute == 0:
        hourly_indices_HighToss.append(i)

y_HighToss_hourly = []
x_HighToss_datenum_hourly = []
for index in sorted(hourly_indices_HighToss):
    y_HighToss_hourly.append(y_HighToss_nonans[index])
    x_HighToss_datenum_hourly.append(x_HighToss_datenum_nonans[index])
y_HighToss_hourly = np.array(y_HighToss_hourly)
x_HighToss_datenum_hourly = np.array(x_HighToss_datenum_hourly)

# Mill Creek
hourly_indices_MillCreek = []
for i in range(len(dates_MillCreek_nonans)):
    if dates_MillCreek_nonans[i].minute == 0:
        hourly_indices_MillCreek.append(i)

y_MillCreek_hourly = []
x_MillCreek_datenum_hourly = []
for index in sorted(hourly_indices_MillCreek):
    y_MillCreek_hourly.append(y_MillCreek_nonans[index])
    x_MillCreek_datenum_hourly.append(x_MillCreek_datenum_nonans[index])
y_MillCreek_hourly = np.array(y_MillCreek_hourly)
x_MillCreek_datenum_hourly = np.array(x_MillCreek_datenum_hourly)

# CNRUS
hourly_indices_CNRUS = []
for i in range(len(dates_CNRUS_nonans)):
    if dates_CNRUS_nonans[i].minute == 0:
        hourly_indices_CNRUS.append(i)

y_CNRUS_hourly = []
x_CNRUS_datenum_hourly = []
for index in sorted(hourly_indices_CNRUS):
    y_CNRUS_hourly.append(y_CNRUS_nonans[index])
    x_CNRUS_datenum_hourly.append(x_CNRUS_datenum_nonans[index])
y_CNRUS_hourly = np.array(y_CNRUS_hourly)
x_CNRUS_datenum_hourly = np.array(x_CNRUS_datenum_hourly)

# Dog Leg
hourly_indices_DogLeg = []
for i in range(len(dates_DogLeg_nonans)):
    if dates_DogLeg_nonans[i].minute == 0:
        hourly_indices_DogLeg.append(i)

y_DogLeg_hourly = []
x_DogLeg_datenum_hourly = []
for index in sorted(hourly_indices_DogLeg):
    y_DogLeg_hourly.append(y_DogLeg_nonans[index])
    x_DogLeg_datenum_hourly.append(x_DogLeg_datenum_nonans[index])
y_DogLeg_hourly = np.array(y_DogLeg_hourly)
x_DogLeg_datenum_hourly = np.array(x_DogLeg_datenum_hourly)

# Old Saw
hourly_indices_OldSaw = []
for i in range(len(dates_OldSaw_nonans)):
    if dates_OldSaw_nonans[i].minute == 0:
        hourly_indices_OldSaw.append(i)

y_OldSaw_hourly = []
x_OldSaw_datenum_hourly = []
for index in sorted(hourly_indices_OldSaw):
    y_OldSaw_hourly.append(y_OldSaw_nonans[index])
    x_OldSaw_datenum_hourly.append(x_OldSaw_datenum_nonans[index])
y_OldSaw_hourly = np.array(y_OldSaw_hourly)
x_OldSaw_datenum_hourly = np.array(x_OldSaw_datenum_hourly)

# plot hourly    
plt.figure()
pylab.plot(x_HighToss_datenum_hourly, y_HighToss_hourly, 'o', markersize=1)
pylab.plot(x_MillCreek_datenum_hourly, y_MillCreek_hourly, 'o', markersize=1)
pylab.plot(x_CNRUS_datenum_hourly, y_CNRUS_hourly, 'o', markersize=1)
pylab.plot(x_DogLeg_datenum_hourly, y_DogLeg_hourly, 'o', markersize=1)
pylab.plot(x_OldSaw_datenum_hourly, y_OldSaw_hourly, 'o', markersize=1)

# High Toss Trendline
idx_HighToss_hourly = np.isfinite(x_HighToss_datenum_hourly) & np.isfinite(y_HighToss_hourly)
z_HighToss_hourly = np.polyfit(x_HighToss_datenum_hourly[idx_HighToss_hourly], y_HighToss_hourly[idx_HighToss_hourly], 1)
p_HighToss_hourly = np.poly1d(z_HighToss_hourly)

# Mill Creek Trendline
idx_MillCreek_hourly = np.isfinite(x_MillCreek_datenum_hourly) & np.isfinite(y_MillCreek_hourly)
z_MillCreek_hourly = np.polyfit(x_MillCreek_datenum_hourly[idx_MillCreek_hourly], y_MillCreek_hourly[idx_MillCreek_hourly], 1)
p_MillCreek_hourly = np.poly1d(z_MillCreek_hourly)

# CNRUS Trendline
idx_CNRUS_hourly = np.isfinite(x_CNRUS_datenum_hourly) & np.isfinite(y_CNRUS_hourly)
z_CNRUS_hourly = np.polyfit(x_CNRUS_datenum_hourly[idx_CNRUS_hourly], y_CNRUS_hourly[idx_CNRUS_hourly], 1)
p_CNRUS_hourly = np.poly1d(z_CNRUS_hourly)

# Dog Leg Trendline
idx_DogLeg_hourly = np.isfinite(x_DogLeg_datenum_hourly) & np.isfinite(y_DogLeg_hourly)
z_DogLeg_hourly = np.polyfit(x_DogLeg_datenum_hourly[idx_DogLeg_hourly], y_DogLeg_hourly[idx_DogLeg_hourly], 1)
p_DogLeg_hourly = np.poly1d(z_DogLeg_hourly)

# Old Saw Trendline
idx_OldSaw_hourly = np.isfinite(x_OldSaw_datenum_hourly) & np.isfinite(y_OldSaw_hourly)
z_OldSaw_hourly = np.polyfit(x_OldSaw_datenum_hourly[idx_OldSaw_hourly], y_OldSaw_hourly[idx_OldSaw_hourly], 1)
p_OldSaw_hourly = np.poly1d(z_OldSaw_hourly)

# Trendlines plotted
polyX_HighToss_hourly = np.linspace(x_HighToss_datenum_hourly.min(), x_HighToss_datenum_hourly.max(), 100)
polyX_MillCreek_hourly = np.linspace(x_MillCreek_datenum_hourly.min(), x_MillCreek_datenum_hourly.max(), 100)
polyX_CNRUS_hourly = np.linspace(x_CNRUS_datenum_hourly.min(), x_CNRUS_datenum_hourly.max(), 100)
polyX_DogLeg_hourly = np.linspace(x_DogLeg_datenum_hourly.min(), x_DogLeg_datenum_hourly.max(), 100)
polyX_OldSaw_hourly = np.linspace(x_OldSaw_datenum_hourly.min(), x_OldSaw_datenum_hourly.max(), 100)
pylab.plot(polyX_HighToss_hourly,p_HighToss_hourly(polyX_HighToss_hourly),"c", label='Mean Hourly High Toss Level')
pylab.plot(polyX_MillCreek_hourly,p_MillCreek_hourly(polyX_MillCreek_hourly),"y", label='Mean Hourly Mill Creek Level')
pylab.plot(polyX_CNRUS_hourly,p_CNRUS_hourly(polyX_CNRUS_hourly),"lime", label='Mean Hourly CNRUS Level')
pylab.plot(polyX_DogLeg_hourly,p_DogLeg_hourly(polyX_DogLeg_hourly),"salmon", label='Mean Hourly Dog Leg Level')
pylab.plot(polyX_OldSaw_hourly,p_OldSaw_hourly(polyX_OldSaw_hourly),"m", label='Mean Hourly Old Saw Level')
# the line equation:
print("y=%.6fx+(%.6f)"%(z_HighToss_hourly[0],z_HighToss_hourly[1]))
print("y=%.6fx+(%.6f)"%(z_MillCreek_hourly[0],z_MillCreek_hourly[1]))
print("y=%.6fx+(%.6f)"%(z_CNRUS_hourly[0],z_CNRUS_hourly[1]))
print("y=%.6fx+(%.6f)"%(z_DogLeg_hourly[0],z_DogLeg_hourly[1]))
print("y=%.6fx+(%.6f)"%(z_OldSaw_hourly[0],z_OldSaw_hourly[1]))

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=18)
plt.ylabel('Elevation (m)', fontsize=16)
plt.legend()

pylab.show()

#%% CTD Data Highs and Lows

"""
Mean Highs and Lows (to determine changes in amplitude and amplitude decay from ocean to river)
"""
# Concatenate dates and levels
HR_dike_HighToss_levels_hourly = np.vstack((x_HighToss_datenum_hourly, y_HighToss_hourly)).T
HR_dike_MillCreek_levels_hourly = np.vstack((x_MillCreek_datenum_hourly, y_MillCreek_hourly)).T
HR_dike_CNRUS_levels_hourly = np.vstack((x_CNRUS_datenum_hourly, y_CNRUS_hourly)).T
HR_dike_DogLeg_levels_hourly = np.vstack((x_DogLeg_datenum_hourly, y_DogLeg_hourly)).T
HR_dike_OldSaw_levels_hourly = np.vstack((x_OldSaw_datenum_hourly, y_OldSaw_hourly)).T
    
# max and min vals for tides
HR_dike_HighToss_maxlevels = []
HR_dike_HighToss_minlevels = []
for i in range(len(HR_dike_HighToss_levels_hourly)-2): 
    if (HR_dike_HighToss_levels_hourly[i+1][1] > HR_dike_HighToss_levels_hourly[i][1]) & (HR_dike_HighToss_levels_hourly[i+2][1] < HR_dike_HighToss_levels_hourly[i+1][1]) & (HR_dike_HighToss_levels_hourly[i+1][1] > p_HighToss(polyX_HighToss).mean()):
        HR_dike_HighToss_maxlevels.append([HR_dike_HighToss_levels_hourly[i+1][0], HR_dike_HighToss_levels_hourly[i+1][1]]) # high tides    
    if (HR_dike_HighToss_levels_hourly[i+1][1] < HR_dike_HighToss_levels_hourly[i][1]) & (HR_dike_HighToss_levels_hourly[i+2][1] > HR_dike_HighToss_levels_hourly[i+1][1]) & (HR_dike_HighToss_levels_hourly[i+1][1] < p_HighToss(polyX_HighToss).mean()):
        HR_dike_HighToss_minlevels.append([HR_dike_HighToss_levels_hourly[i+1][0], HR_dike_HighToss_levels_hourly[i+1][1]])
HR_dike_HighToss_maxlevels = np.array(HR_dike_HighToss_maxlevels)
HR_dike_HighToss_minlevels = np.array(HR_dike_HighToss_minlevels) 
    
HR_dike_MillCreek_maxlevels = [] # these are seasonal, not tidal
HR_dike_MillCreek_minlevels = [] # these are seasonal, not tidal
for i in range(len(HR_dike_MillCreek_levels_hourly)-2): 
    if (HR_dike_MillCreek_levels_hourly[i+1][1] > HR_dike_MillCreek_levels_hourly[i][1]) & (HR_dike_MillCreek_levels_hourly[i+2][1] < HR_dike_MillCreek_levels_hourly[i+1][1]) & (HR_dike_MillCreek_levels_hourly[i+1][1] > p_MillCreek(polyX_MillCreek).mean()):
        HR_dike_MillCreek_maxlevels.append([HR_dike_MillCreek_levels_hourly[i+1][0], HR_dike_MillCreek_levels_hourly[i+1][1]]) # high tides    
    if (HR_dike_MillCreek_levels_hourly[i+1][1] < HR_dike_MillCreek_levels_hourly[i][1]) & (HR_dike_MillCreek_levels_hourly[i+2][1] > HR_dike_MillCreek_levels_hourly[i+1][1]) & (HR_dike_MillCreek_levels_hourly[i+1][1] < p_MillCreek(polyX_MillCreek).mean()):
        HR_dike_MillCreek_minlevels.append([HR_dike_MillCreek_levels_hourly[i+1][0], HR_dike_MillCreek_levels_hourly[i+1][1]])
HR_dike_MillCreek_maxlevels = np.array(HR_dike_MillCreek_maxlevels) # these are seasonal, not tidal
HR_dike_MillCreek_minlevels = np.array(HR_dike_MillCreek_minlevels) # these are seasonal, not tidal

HR_dike_CNRUS_maxlevels = []
HR_dike_CNRUS_minlevels = []
for i in range(len(HR_dike_CNRUS_levels_hourly)-2): 
    if (HR_dike_CNRUS_levels_hourly[i+1][1] > HR_dike_CNRUS_levels_hourly[i][1]) & (HR_dike_CNRUS_levels_hourly[i+2][1] < HR_dike_CNRUS_levels_hourly[i+1][1]) & (HR_dike_CNRUS_levels_hourly[i+1][1] > p_CNRUS(polyX_CNRUS).mean()):
        HR_dike_CNRUS_maxlevels.append([HR_dike_CNRUS_levels_hourly[i+1][0], HR_dike_CNRUS_levels_hourly[i+1][1]]) # high tides    
    if (HR_dike_CNRUS_levels_hourly[i+1][1] < HR_dike_CNRUS_levels_hourly[i][1]) & (HR_dike_CNRUS_levels_hourly[i+2][1] > HR_dike_CNRUS_levels_hourly[i+1][1]) & (HR_dike_CNRUS_levels_hourly[i+1][1] < p_CNRUS(polyX_CNRUS).mean()):
        HR_dike_CNRUS_minlevels.append([HR_dike_CNRUS_levels_hourly[i+1][0], HR_dike_CNRUS_levels_hourly[i+1][1]])
HR_dike_CNRUS_maxlevels = np.array(HR_dike_CNRUS_maxlevels)
HR_dike_CNRUS_minlevels = np.array(HR_dike_CNRUS_minlevels) 

HR_dike_DogLeg_maxlevels = []
HR_dike_DogLeg_minlevels = []
for i in range(len(HR_dike_DogLeg_levels_hourly)-2): 
    if (HR_dike_DogLeg_levels_hourly[i+1][1] > HR_dike_DogLeg_levels_hourly[i][1]) & (HR_dike_DogLeg_levels_hourly[i+2][1] < HR_dike_DogLeg_levels_hourly[i+1][1]) & (HR_dike_DogLeg_levels_hourly[i+1][1] > p_DogLeg(polyX_DogLeg).mean()):
        HR_dike_DogLeg_maxlevels.append([HR_dike_DogLeg_levels_hourly[i+1][0], HR_dike_DogLeg_levels_hourly[i+1][1]]) # high tides    
    if (HR_dike_DogLeg_levels_hourly[i+1][1] < HR_dike_DogLeg_levels_hourly[i][1]) & (HR_dike_DogLeg_levels_hourly[i+2][1] > HR_dike_DogLeg_levels_hourly[i+1][1]) & (HR_dike_DogLeg_levels_hourly[i+1][1] < p_DogLeg(polyX_DogLeg).mean()):
        HR_dike_DogLeg_minlevels.append([HR_dike_DogLeg_levels_hourly[i+1][0], HR_dike_DogLeg_levels_hourly[i+1][1]])
HR_dike_DogLeg_maxlevels = np.array(HR_dike_DogLeg_maxlevels)
HR_dike_DogLeg_minlevels = np.array(HR_dike_DogLeg_minlevels) 

HR_dike_OldSaw_maxlevels = []
HR_dike_OldSaw_minlevels = []
for i in range(len(HR_dike_OldSaw_levels_hourly)-2): 
    if (HR_dike_OldSaw_levels_hourly[i+1][1] > HR_dike_OldSaw_levels_hourly[i][1]) & (HR_dike_OldSaw_levels_hourly[i+2][1] < HR_dike_OldSaw_levels_hourly[i+1][1]) & (HR_dike_OldSaw_levels_hourly[i+1][1] > p_OldSaw(polyX_OldSaw).mean()):
        HR_dike_OldSaw_maxlevels.append([HR_dike_OldSaw_levels_hourly[i+1][0], HR_dike_OldSaw_levels_hourly[i+1][1]]) # high tides    
    if (HR_dike_OldSaw_levels_hourly[i+1][1] < HR_dike_OldSaw_levels_hourly[i][1]) & (HR_dike_OldSaw_levels_hourly[i+2][1] > HR_dike_OldSaw_levels_hourly[i+1][1]) & (HR_dike_OldSaw_levels_hourly[i+1][1] < p_OldSaw(polyX_OldSaw).mean()):
        HR_dike_OldSaw_minlevels.append([HR_dike_OldSaw_levels_hourly[i+1][0], HR_dike_OldSaw_levels_hourly[i+1][1]])
HR_dike_OldSaw_maxlevels = np.array(HR_dike_OldSaw_maxlevels)
HR_dike_OldSaw_minlevels = np.array(HR_dike_OldSaw_minlevels) 

"""
Misc sensor mins and maxes (hourly time steps)
"""
x_HighToss_datenum_maxlevels, y_HighToss_maxlevels = HR_dike_HighToss_maxlevels.T
x_HighToss_datenum_minlevels, y_HighToss_minlevels = HR_dike_HighToss_minlevels.T
x_MillCreek_datenum_maxlevels, y_MillCreek_maxlevels = HR_dike_MillCreek_maxlevels.T # these are seasonal, not tidal
x_MillCreek_datenum_minlevels, y_MillCreek_minlevels = HR_dike_MillCreek_minlevels.T # these are seasonal, not tidal
x_CNRUS_datenum_maxlevels, y_CNRUS_maxlevels = HR_dike_CNRUS_maxlevels.T
x_CNRUS_datenum_minlevels, y_CNRUS_minlevels = HR_dike_CNRUS_minlevels.T
x_DogLeg_datenum_maxlevels, y_DogLeg_maxlevels = HR_dike_DogLeg_maxlevels.T
x_DogLeg_datenum_minlevels, y_DogLeg_minlevels = HR_dike_DogLeg_minlevels.T
x_OldSaw_datenum_maxlevels, y_OldSaw_maxlevels = HR_dike_OldSaw_maxlevels.T
x_OldSaw_datenum_minlevels, y_OldSaw_minlevels = HR_dike_OldSaw_minlevels.T

# plots
plt.figure()
pylab.plot(x_HighToss_datenum_maxlevels, y_HighToss_maxlevels, 'o', markersize=1)
pylab.plot(x_HighToss_datenum_minlevels, y_HighToss_minlevels, 'o', markersize=1)
pylab.plot(x_MillCreek_datenum_maxlevels, y_MillCreek_maxlevels, 'o', markersize=1) # might give seasonal range?
pylab.plot(x_MillCreek_datenum_minlevels, y_MillCreek_minlevels, 'o', markersize=1) # might give seasonal range?
pylab.plot(x_CNRUS_datenum_maxlevels, y_CNRUS_maxlevels, 'o', markersize=1)
pylab.plot(x_CNRUS_datenum_minlevels, y_CNRUS_minlevels, 'o', markersize=1)
pylab.plot(x_DogLeg_datenum_maxlevels, y_DogLeg_maxlevels, 'o', markersize=1)
pylab.plot(x_DogLeg_datenum_minlevels, y_DogLeg_minlevels, 'o', markersize=1)
pylab.plot(x_OldSaw_datenum_maxlevels, y_OldSaw_maxlevels, 'o', markersize=1)
pylab.plot(x_OldSaw_datenum_minlevels, y_OldSaw_minlevels, 'o', markersize=1)

# trendlines
idx_HighToss_max = np.isfinite(x_HighToss_datenum_maxlevels) & np.isfinite(y_HighToss_maxlevels)
idx_HighToss_min = np.isfinite(x_HighToss_datenum_minlevels) & np.isfinite(y_HighToss_minlevels)
z_HighToss_max = np.polyfit(x_HighToss_datenum_maxlevels[idx_HighToss_max], y_HighToss_maxlevels[idx_HighToss_max], 1)
z_HighToss_min = np.polyfit(x_HighToss_datenum_minlevels[idx_HighToss_min], y_HighToss_minlevels[idx_HighToss_min], 1)
p_HighToss_max = np.poly1d(z_HighToss_max)
p_HighToss_min = np.poly1d(z_HighToss_min)
idx_MillCreek_max = np.isfinite(x_MillCreek_datenum_maxlevels) & np.isfinite(y_MillCreek_maxlevels) # seasonal?
idx_MillCreek_min = np.isfinite(x_MillCreek_datenum_minlevels) & np.isfinite(y_MillCreek_minlevels) # seasonal?
z_MillCreek_max = np.polyfit(x_MillCreek_datenum_maxlevels[idx_MillCreek_max], y_MillCreek_maxlevels[idx_MillCreek_max], 1) # seasonal?
z_MillCreek_min = np.polyfit(x_MillCreek_datenum_minlevels[idx_MillCreek_min], y_MillCreek_minlevels[idx_MillCreek_min], 1) # seasonal?
p_MillCreek_max = np.poly1d(z_MillCreek_max) # seasonal?
p_MillCreek_min = np.poly1d(z_MillCreek_min) # seasonal?
idx_CNRUS_max = np.isfinite(x_CNRUS_datenum_maxlevels) & np.isfinite(y_CNRUS_maxlevels)
idx_CNRUS_min = np.isfinite(x_CNRUS_datenum_minlevels) & np.isfinite(y_CNRUS_minlevels)
z_CNRUS_max = np.polyfit(x_CNRUS_datenum_maxlevels[idx_CNRUS_max], y_CNRUS_maxlevels[idx_CNRUS_max], 1)
z_CNRUS_min = np.polyfit(x_CNRUS_datenum_minlevels[idx_CNRUS_min], y_CNRUS_minlevels[idx_CNRUS_min], 1)
p_CNRUS_max = np.poly1d(z_CNRUS_max)
p_CNRUS_min = np.poly1d(z_CNRUS_min)
idx_DogLeg_max = np.isfinite(x_DogLeg_datenum_maxlevels) & np.isfinite(y_DogLeg_maxlevels)
idx_DogLeg_min = np.isfinite(x_DogLeg_datenum_minlevels) & np.isfinite(y_DogLeg_minlevels)
z_DogLeg_max = np.polyfit(x_DogLeg_datenum_maxlevels[idx_DogLeg_max], y_DogLeg_maxlevels[idx_DogLeg_max], 1)
z_DogLeg_min = np.polyfit(x_DogLeg_datenum_minlevels[idx_DogLeg_min], y_DogLeg_minlevels[idx_DogLeg_min], 1)
p_DogLeg_max = np.poly1d(z_DogLeg_max)
p_DogLeg_min = np.poly1d(z_DogLeg_min)
idx_OldSaw_max = np.isfinite(x_OldSaw_datenum_maxlevels) & np.isfinite(y_OldSaw_maxlevels)
idx_OldSaw_min = np.isfinite(x_OldSaw_datenum_minlevels) & np.isfinite(y_OldSaw_minlevels)
z_OldSaw_max = np.polyfit(x_OldSaw_datenum_maxlevels[idx_OldSaw_max], y_OldSaw_maxlevels[idx_OldSaw_max], 1)
z_OldSaw_min = np.polyfit(x_OldSaw_datenum_minlevels[idx_OldSaw_min], y_OldSaw_minlevels[idx_OldSaw_min], 1)
p_OldSaw_max = np.poly1d(z_OldSaw_max)
p_OldSaw_min = np.poly1d(z_OldSaw_min)

# plotted trendlines
polyX_HighToss_max = np.linspace(x_HighToss_datenum_maxlevels.min(), x_HighToss_datenum_maxlevels.max(), 100)
polyX_HighToss_min = np.linspace(x_HighToss_datenum_minlevels.min(), x_HighToss_datenum_minlevels.max(), 100)
polyX_MillCreek_max = np.linspace(x_MillCreek_datenum_maxlevels.min(), x_MillCreek_datenum_maxlevels.max(), 100)
polyX_MillCreek_min = np.linspace(x_MillCreek_datenum_minlevels.min(), x_MillCreek_datenum_minlevels.max(), 100)
polyX_CNRUS_max = np.linspace(x_CNRUS_datenum_maxlevels.min(), x_CNRUS_datenum_maxlevels.max(), 100)
polyX_CNRUS_min = np.linspace(x_CNRUS_datenum_minlevels.min(), x_CNRUS_datenum_minlevels.max(), 100)
polyX_DogLeg_max = np.linspace(x_DogLeg_datenum_maxlevels.min(), x_DogLeg_datenum_maxlevels.max(), 100)
polyX_DogLeg_min = np.linspace(x_DogLeg_datenum_minlevels.min(), x_DogLeg_datenum_minlevels.max(), 100)
polyX_OldSaw_max = np.linspace(x_OldSaw_datenum_maxlevels.min(), x_OldSaw_datenum_maxlevels.max(), 100)
polyX_OldSaw_min = np.linspace(x_OldSaw_datenum_minlevels.min(), x_OldSaw_datenum_minlevels.max(), 100)
pylab.plot(polyX_HighToss_max,p_HighToss_max(polyX_HighToss_max),"blue", label='Mean High HT Level')
pylab.plot(polyX_HighToss_min,p_HighToss_min(polyX_HighToss_min),"orange", label='Mean Low HT Level')
pylab.plot(polyX_MillCreek_max,p_MillCreek_max(polyX_MillCreek_max),"green", label='Mean High MC Level') # seasonal?
pylab.plot(polyX_MillCreek_min,p_MillCreek_min(polyX_MillCreek_min),"red", label='Mean Low MC Level') # seasonal?
pylab.plot(polyX_CNRUS_max,p_CNRUS_max(polyX_CNRUS_max),"purple", label='Mean High CNR Level')
pylab.plot(polyX_CNRUS_min,p_CNRUS_min(polyX_CNRUS_min),"brown", label='Mean Low CNR Level')
pylab.plot(polyX_DogLeg_max,p_DogLeg_max(polyX_DogLeg_max),"pink", label='Mean High DL Level')
pylab.plot(polyX_DogLeg_min,p_DogLeg_min(polyX_DogLeg_min),"grey", label='Mean Low DL Level')
pylab.plot(polyX_OldSaw_max,p_OldSaw_max(polyX_OldSaw_max),"yellow", label='Mean High OS Level')
pylab.plot(polyX_OldSaw_min,p_OldSaw_min(polyX_OldSaw_min),"cyan", label='Mean Low OS Level')
# the line equation:
print("y=%.6fx+(%.6f)"%(z_HighToss_max[0],z_HighToss_max[1]))
print("y=%.6fx+(%.6f)"%(z_HighToss_min[0],z_HighToss_min[1]))
print("y=%.6fx+(%.6f)"%(z_MillCreek_max[0],z_MillCreek_max[1]))
print("y=%.6fx+(%.6f)"%(z_MillCreek_min[0],z_MillCreek_min[1]))
print("y=%.6fx+(%.6f)"%(z_CNRUS_max[0],z_CNRUS_max[1]))
print("y=%.6fx+(%.6f)"%(z_CNRUS_min[0],z_CNRUS_min[1]))
print("y=%.6fx+(%.6f)"%(z_DogLeg_max[0],z_DogLeg_max[1]))
print("y=%.6fx+(%.6f)"%(z_DogLeg_min[0],z_DogLeg_min[1]))
print("y=%.6fx+(%.6f)"%(z_OldSaw_max[0],z_OldSaw_max[1]))
print("y=%.6fx+(%.6f)"%(z_OldSaw_min[0],z_OldSaw_min[1]))

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=18)
plt.ylabel('Elevation (m)', fontsize=16)
plt.legend()

pylab.show()

# Max and min of trendlines
# High Toss
HTlev_july2017_max = z_HighToss_max[0]*x_HighToss_datenum_maxlevels.min()+z_HighToss_max[1]
HTlev_july2017_min = z_HighToss_min[0]*x_HighToss_datenum_minlevels.min()+z_HighToss_min[1]
HTlev_july2019_max = z_HighToss_max[0]*x_HighToss_datenum_maxlevels.max()+z_HighToss_max[1]
HTlev_july2019_min = z_HighToss_min[0]*x_HighToss_datenum_minlevels.max()+z_HighToss_min[1]
HTrlrhigh_july2017tojuly2019 = HTlev_july2019_max-HTlev_july2017_max
HTrlrlow_july2017tojuly2019 = HTlev_july2019_min-HTlev_july2017_min
HTrlrhigh_oneyear = HTrlrhigh_july2017tojuly2019/(x_HighToss_datenum_maxlevels[-1]-x_HighToss_datenum_maxlevels[0])*365
HTrlrlow_oneyear = HTrlrlow_july2017tojuly2019/(x_HighToss_datenum_minlevels[-1]-x_HighToss_datenum_minlevels[0])*365
    
print("Maximum mean river level at the High Toss sensor is increasing by ", ("%.3f"%(HTrlrhigh_oneyear)), "m per year.")
print("Maximum mean river level at that location goes from an average of ", ("%.3f"%(HTlev_july2017_max)), "m in July 2017")
print("to ", ("%.3f"%(HTlev_july2019_max)), "m in July 2019.")

print("Minimum mean river level at the High Toss sensor is increasing by ", ("%.3f"%(HTrlrlow_oneyear)), "m per year.")
print("Minimum mean river level at that location goes from an average of ", ("%.3f"%(HTlev_july2017_min)), "m in July 2017")
print("to ", ("%.3f"%(HTlev_july2019_min)), "m in July 2019.")

# Mill Creek Seasonal
MClev_july2017_max = z_MillCreek_max[0]*x_MillCreek_datenum_maxlevels.min()+z_MillCreek_max[1]
MClev_july2017_min = z_MillCreek_min[0]*x_MillCreek_datenum_minlevels.min()+z_MillCreek_min[1]
MClev_july2019_max = z_MillCreek_max[0]*x_MillCreek_datenum_maxlevels.max()+z_MillCreek_max[1]
MClev_july2019_min = z_MillCreek_min[0]*x_MillCreek_datenum_minlevels.max()+z_MillCreek_min[1]
MCrlrhigh_july2017tojuly2019 = MClev_july2019_max-MClev_july2017_max
MCrlrlow_july2017tojuly2019 = MClev_july2019_min-MClev_july2017_min
MCrlrhigh_oneyear = MCrlrhigh_july2017tojuly2019/(x_MillCreek_datenum_maxlevels[-1]-x_MillCreek_datenum_maxlevels[0])*365
MCrlrlow_oneyear = MCrlrlow_july2017tojuly2019/(x_MillCreek_datenum_minlevels[-1]-x_MillCreek_datenum_minlevels[0])*365

print("Maximum mean Mill Creek level is increasing by ", ("%.3f"%(MCrlrhigh_oneyear)), "m per year.")
print("Maximum mean creek level at that location goes from an average of ", ("%.3f"%(MClev_july2017_max)), "m in July 2017")
print("to ", ("%.3f"%(MClev_july2019_max)), "m in July 2019.")

print("Minimum mean Mill Creek level is increasing by ", ("%.3f"%(MCrlrlow_oneyear)), "m per year.")
print("Minimum mean creek level at that location goes from an average of ", ("%.3f"%(MClev_july2017_min)), "m in July 2017")
print("to ", ("%.3f"%(MClev_july2019_min)), "m in July 2019.")

# CNR U/S
CNRlev_july2017_max = z_CNRUS_max[0]*x_CNRUS_datenum_maxlevels.min()+z_CNRUS_max[1]
CNRlev_july2017_min = z_CNRUS_min[0]*x_CNRUS_datenum_minlevels.min()+z_CNRUS_min[1]
CNRlev_july2019_max = z_CNRUS_max[0]*x_CNRUS_datenum_maxlevels.max()+z_CNRUS_max[1]
CNRlev_july2019_min = z_CNRUS_min[0]*x_CNRUS_datenum_minlevels.max()+z_CNRUS_min[1]
CNRrlrhigh_july2017tojuly2019 = CNRlev_july2019_max-CNRlev_july2017_max
CNRrlrlow_july2017tojuly2019 = CNRlev_july2019_min-CNRlev_july2017_min
CNRrlrhigh_oneyear = CNRrlrhigh_july2017tojuly2019/(x_CNRUS_datenum_maxlevels[-1]-x_CNRUS_datenum_maxlevels[0])*365
CNRrlrlow_oneyear = CNRrlrlow_july2017tojuly2019/(x_CNRUS_datenum_minlevels[-1]-x_CNRUS_datenum_minlevels[0])*365
    
print("Maximum mean river level at the CNR U/S sensor is increasing by ", ("%.3f"%(CNRrlrhigh_oneyear)), "m per year.")
print("Maximum mean river level at that location goes from an average of ", ("%.3f"%(CNRlev_july2017_max)), "m in July 2017")
print("to ", ("%.3f"%(CNRlev_july2019_max)), "m in July 2019.")

print("Minimum mean river level at the CNR U/S sensor is increasing by ", ("%.3f"%(CNRrlrlow_oneyear)), "m per year.")
print("Minimum mean river level at that location goes from an average of ", ("%.3f"%(CNRlev_july2017_min)), "m in July 2017")
print("to ", ("%.3f"%(CNRlev_july2019_min)), "m in July 2019.")

# Dog Leg
DLlev_july2017_max = z_DogLeg_max[0]*x_DogLeg_datenum_maxlevels.min()+z_DogLeg_max[1]
DLlev_july2017_min = z_DogLeg_min[0]*x_DogLeg_datenum_minlevels.min()+z_DogLeg_min[1]
DLlev_july2019_max = z_DogLeg_max[0]*x_DogLeg_datenum_maxlevels.max()+z_DogLeg_max[1]
DLlev_july2019_min = z_DogLeg_min[0]*x_DogLeg_datenum_minlevels.max()+z_DogLeg_min[1]
DLrlrhigh_july2017tojuly2019 = DLlev_july2019_max-DLlev_july2017_max
DLrlrlow_july2017tojuly2019 = DLlev_july2019_min-DLlev_july2017_min
DLrlrhigh_oneyear = DLrlrhigh_july2017tojuly2019/(x_DogLeg_datenum_maxlevels[-1]-x_DogLeg_datenum_maxlevels[0])*365
DLrlrlow_oneyear = DLrlrlow_july2017tojuly2019/(x_DogLeg_datenum_minlevels[-1]-x_DogLeg_datenum_minlevels[0])*365
    
print("Maximum mean river level at Dog Leg is increasing by ", ("%.3f"%(DLrlrhigh_oneyear)), "m per year.")
print("Maximum mean river level at that location goes from an average of ", ("%.3f"%(DLlev_july2017_max)), "m in July 2017")
print("to ", ("%.3f"%(DLlev_july2019_max)), "m in July 2019.")

print("Minimum mean river level at Dog Leg is increasing by ", ("%.3f"%(DLrlrlow_oneyear)), "m per year.")
print("Minimum mean river level at that location goes from an average of ", ("%.3f"%(DLlev_july2017_min)), "m in July 2017")
print("to ", ("%.3f"%(DLlev_july2019_min)), "m in July 2019.")

# Old Saw
OSlev_june2018_max = z_OldSaw_max[0]*x_OldSaw_datenum_maxlevels.min()+z_OldSaw_max[1]
OSlev_june2018_min = z_OldSaw_min[0]*x_OldSaw_datenum_minlevels.min()+z_OldSaw_min[1]
OSlev_dec2018_max = z_OldSaw_max[0]*x_OldSaw_datenum_maxlevels.max()+z_OldSaw_max[1]
OSlev_dec2018_min = z_OldSaw_min[0]*x_OldSaw_datenum_minlevels.max()+z_OldSaw_min[1]
OSslrhigh_june2018todec2018 = OSlev_dec2018_max-OSlev_june2018_max
OSslrlow_june2018todec2018 = OSlev_dec2018_min-OSlev_june2018_min
    
print("Maximum mean sea level at Old Saw changed", ("%.3f"%(OSslrhigh_june2018todec2018)), "m between June 2018 and December 2018.")
print("Minimum mean sea level at Old Saw changed", ("%.3f"%(OSslrlow_june2018todec2018)), "m between June 2018 and December 2018.")

# Max and min of trendlines
# High Toss
HTlev_july2017_max = z_HighToss_max[0]*x_HighToss_datenum_maxlevels.min()+z_HighToss_max[1]
HTlev_july2017_min = z_HighToss_min[0]*x_HighToss_datenum_minlevels.min()+z_HighToss_min[1]
HTlev_july2019_max = z_HighToss_max[0]*x_HighToss_datenum_maxlevels.max()+z_HighToss_max[1]
HTlev_july2019_min = z_HighToss_min[0]*x_HighToss_datenum_minlevels.max()+z_HighToss_min[1]
HTrlrhigh_july2017tojuly2019 = HTlev_july2019_max-HTlev_july2017_max
HTrlrlow_july2017tojuly2019 = HTlev_july2019_min-HTlev_july2017_min
HTrlrhigh_oneyear = HTrlrhigh_july2017tojuly2019/(x_HighToss_datenum_maxlevels[-1]-x_HighToss_datenum_maxlevels[0])*365
HTrlrlow_oneyear = HTrlrlow_july2017tojuly2019/(x_HighToss_datenum_minlevels[-1]-x_HighToss_datenum_minlevels[0])*365
    
print("Maximum mean river level at the High Toss sensor is increasing by ", ("%.3f"%(HTrlrhigh_oneyear)), "m per year.")
print("Maximum mean river level at that location goes from an average of ", ("%.3f"%(HTlev_july2017_max)), "m in July 2017")
print("to ", ("%.3f"%(HTlev_july2019_max)), "m in July 2019.")

print("Minimum mean river level at the High Toss sensor is increasing by ", ("%.3f"%(HTrlrlow_oneyear)), "m per year.")
print("Minimum mean river level at that location goes from an average of ", ("%.3f"%(HTlev_july2017_min)), "m in July 2017")
print("to ", ("%.3f"%(HTlev_july2019_min)), "m in July 2019.")

#%% Regression plot between Salinity levels and Water Elevations, CNR U/S CTD (proceeding cells must be run first)

HR_dike_CNRUS_sal_maxlevsdf = pd.DataFrame({'Date':HR_dike_CNRUS_sal_maxlevels[:,0], 'Salinity':HR_dike_CNRUS_sal_maxlevels[:,1]})
HR_dike_CNRUS_sal_minlevsdf = pd.DataFrame({'Date':HR_dike_CNRUS_sal_minlevels[:,0], 'Salinity':HR_dike_CNRUS_sal_minlevels[:,1]})
HR_dike_CNRUS_maxlevsdf = pd.DataFrame({'Date':HR_dike_CNRUS_maxlevels[:,0], 'ElevNAVD88':HR_dike_CNRUS_maxlevels[:,1]})
HR_dike_CNRUS_minlevsdf = pd.DataFrame({'Date':HR_dike_CNRUS_minlevels[:,0], 'ElevNAVD88':HR_dike_CNRUS_minlevels[:,1]})

HR_dike_CNRUS_sal_lev_maxdf = pd.merge(left=HR_dike_CNRUS_maxlevsdf, right=HR_dike_CNRUS_sal_maxlevsdf, left_on='Date', right_on='Date')
HR_dike_CNRUS_sal_lev_mindf = pd.merge(left=HR_dike_CNRUS_minlevsdf, right=HR_dike_CNRUS_sal_minlevsdf, left_on='Date', right_on='Date')

ax = plt.gca()
HR_dike_CNRUS_sal_lev_maxdf.plot(kind='scatter',x='ElevNAVD88',y='Salinity',color='green',ax=ax)
HR_dike_CNRUS_sal_lev_mindf.plot(kind='scatter',x='ElevNAVD88',y='Salinity',color='purple',ax=ax)

plt.xlabel('HR Elevation at CNR U/S, NAVD88 (m)', fontsize=26)
plt.ylabel('Salinity (ppt)',fontsize=26)

HR_dike_CNRUS_sal_max = np.array([HR_dike_CNRUS_sal_lev_maxdf['Salinity']]).T
HR_dike_CNRUS_lev_max = np.array([HR_dike_CNRUS_sal_lev_maxdf['ElevNAVD88']]).T
HR_dike_CNRUS_sal_min = np.array([HR_dike_CNRUS_sal_lev_mindf['Salinity']]).T
HR_dike_CNRUS_lev_min = np.array([HR_dike_CNRUS_sal_lev_mindf['ElevNAVD88']]).T

# trendlines
idx_CNRUS_sal_lev_max = np.isfinite(HR_dike_CNRUS_lev_max) & np.isfinite(HR_dike_CNRUS_sal_max)
idx_CNRUS_sal_lev_min = np.isfinite(HR_dike_CNRUS_lev_min) & np.isfinite(HR_dike_CNRUS_sal_min)
z_CNRUS_sal_lev_max = np.polyfit(HR_dike_CNRUS_lev_max[idx_CNRUS_sal_lev_max], HR_dike_CNRUS_sal_max[idx_CNRUS_sal_lev_max], 1)
z_CNRUS_sal_lev_min = np.polyfit(HR_dike_CNRUS_lev_min[idx_CNRUS_sal_lev_min], HR_dike_CNRUS_sal_min[idx_CNRUS_sal_lev_min], 1)
p_CNRUS_sal_lev_max = np.poly1d(z_CNRUS_sal_lev_max)
p_CNRUS_sal_lev_min = np.poly1d(z_CNRUS_sal_lev_min)

# plotted trendlines
polyX_CNRUS_sal_lev_max = np.linspace(HR_dike_CNRUS_lev_max.min(), HR_dike_CNRUS_lev_max.max(), 100)
polyX_CNRUS_sal_lev_min = np.linspace(HR_dike_CNRUS_lev_min.min(), HR_dike_CNRUS_lev_min.max(), 100)
pylab.plot(polyX_CNRUS_sal_lev_max,p_CNRUS_sal_lev_max(polyX_CNRUS_sal_lev_max),"lightgreen", label='High Tide')
pylab.plot(polyX_CNRUS_sal_lev_min,p_CNRUS_sal_lev_min(polyX_CNRUS_sal_lev_min),"mediumpurple", label='Low Tide')
# the line equation:
print("y=%.6fx+(%.6f)"%(z_CNRUS_sal_lev_max[0],z_CNRUS_sal_lev_max[1]))
print("y=%.6fx+(%.6f)"%(z_CNRUS_sal_lev_min[0],z_CNRUS_sal_lev_min[1]))
plt.legend(loc='best',fontsize=22)

# Performing linear regression to ensure slope and constant match the ones predicted by the trendlines.

HR_dike_CNRUS_sal_max_mean = np.nanmean(HR_dike_CNRUS_sal_max)
HR_dike_CNRUS_lev_max_mean = np.nanmean(HR_dike_CNRUS_lev_max)
HR_dike_CNRUS_sal_min_mean = np.nanmean(HR_dike_CNRUS_sal_min)
HR_dike_CNRUS_lev_min_mean = np.nanmean(HR_dike_CNRUS_lev_min)

HR_dike_CNRUS_sal_max_stdev = np.nanstd(HR_dike_CNRUS_sal_max)
HR_dike_CNRUS_lev_max_stdev = np.nanstd(HR_dike_CNRUS_lev_max)
HR_dike_CNRUS_sal_min_stdev = np.nanstd(HR_dike_CNRUS_sal_min)
HR_dike_CNRUS_lev_min_stdev = np.nanstd(HR_dike_CNRUS_lev_min)

HR_dike_CNRUS_sal_max_count = HR_dike_CNRUS_sal_max.size
HR_dike_CNRUS_lev_max_count = HR_dike_CNRUS_lev_max.size
HR_dike_CNRUS_sal_min_count = HR_dike_CNRUS_sal_min.size
HR_dike_CNRUS_lev_min_count = HR_dike_CNRUS_lev_min.size

HR_dike_CNRUS_sal_lev_max_corrcoeff = 1/HR_dike_CNRUS_sal_max_count*sum((HR_dike_CNRUS_lev_max-HR_dike_CNRUS_lev_max_mean)*(HR_dike_CNRUS_sal_max-HR_dike_CNRUS_sal_max_mean))/HR_dike_CNRUS_lev_max_stdev/HR_dike_CNRUS_sal_max_stdev 
HR_dike_CNRUS_sal_lev_min_corrcoeff = 1/HR_dike_CNRUS_sal_min_count*sum((HR_dike_CNRUS_lev_min-HR_dike_CNRUS_lev_min_mean)*(HR_dike_CNRUS_sal_min-HR_dike_CNRUS_sal_min_mean))/HR_dike_CNRUS_lev_min_stdev/HR_dike_CNRUS_sal_min_stdev

linreg_a_max = HR_dike_CNRUS_sal_lev_max_corrcoeff*HR_dike_CNRUS_sal_max_stdev/HR_dike_CNRUS_lev_max_stdev
linreg_b_max = HR_dike_CNRUS_sal_max_mean-linreg_a_max*HR_dike_CNRUS_lev_max_mean

linreg_a_min = HR_dike_CNRUS_sal_lev_min_corrcoeff*HR_dike_CNRUS_sal_min_stdev/HR_dike_CNRUS_lev_min_stdev
linreg_b_min = HR_dike_CNRUS_sal_min_mean-linreg_a_min*HR_dike_CNRUS_lev_min_mean

#%% Ranges (maxval-minval for each 12hr 25min interval)

tidal_peaktopeak_interval = 12/24 + 25/(60*24) # bin width in days

# Ocean side of dike
bin_start = 0
x_oceanside_rangedates = []    
y_oceanside_mins = []
y_oceanside_maxes = []
for bin_index in range(len(x_oceanside_datenum)):
    datestart = x_oceanside_datenum[bin_start]
    dateend = datestart + (x_oceanside_datenum[bin_index] - x_oceanside_datenum[bin_start])
    date_interval = dateend - datestart
    bin_end = bin_index
    if (date_interval >= tidal_peaktopeak_interval):
            x_oceanside_rangedates.append(x_oceanside_datenum[int((bin_start+bin_end)/2)])
            y_oceanside_mins.append(np.nanmin(y_oceanside[bin_start:bin_end]))
            y_oceanside_maxes.append(np.nanmax(y_oceanside[bin_start:bin_end]))
            bin_start = bin_end
x_oceanside_rangedates = np.array(x_oceanside_rangedates)
y_oceanside_mins = np.array(y_oceanside_mins)
y_oceanside_maxes = np.array(y_oceanside_maxes)
y_oceanside_mins[y_oceanside_mins > np.nanmean(y_oceanside_maxes)] = np.nan
y_oceanside_maxes[y_oceanside_maxes < np.nanmean(y_oceanside_mins)] = np.nan
y_oceanside_ranges = y_oceanside_maxes - y_oceanside_mins

# HR side of dike
bin_start = 0
x_HRside_rangedates = []    
y_HRside_mins = []
y_HRside_maxes = []
for bin_index in range(len(x_HRside_datenum)):
    datestart = x_HRside_datenum[bin_start]
    dateend = datestart + (x_HRside_datenum[bin_index] - x_HRside_datenum[bin_start])
    date_interval = dateend - datestart
    bin_end = bin_index
    if (date_interval >= tidal_peaktopeak_interval):
            x_HRside_rangedates.append(x_HRside_datenum[int((bin_start+bin_end)/2)])
            y_HRside_mins.append(np.nanmin(y_HRside[bin_start:bin_end]))
            y_HRside_maxes.append(np.nanmax(y_HRside[bin_start:bin_end]))
            bin_start = bin_end
x_HRside_rangedates = np.array(x_HRside_rangedates)
y_HRside_mins = np.array(y_HRside_mins)
y_HRside_maxes = np.array(y_HRside_maxes)
y_HRside_mins[y_HRside_mins > np.nanmean(y_HRside_maxes)] = np.nan
y_HRside_maxes[y_HRside_maxes < np.nanmean(y_HRside_mins)] = np.nan
y_HRside_ranges = y_HRside_maxes - y_HRside_mins

# Discharge through dike
bin_start = 0
x_discharge_rangedates = []    
y_discharge_mins = []
y_discharge_maxes = []
for bin_index in range(len(x_discharge_datenum)):
    datestart = x_discharge_datenum[bin_start]
    dateend = datestart + (x_discharge_datenum[bin_index] - x_discharge_datenum[bin_start])
    date_interval = dateend - datestart
    bin_end = bin_index
    if (date_interval >= tidal_peaktopeak_interval):
            x_discharge_rangedates.append(x_discharge_datenum[int((bin_start+bin_end)/2)])
            y_discharge_mins.append(np.nanmin(y_discharge[bin_start:bin_end]))
            y_discharge_maxes.append(np.nanmax(y_discharge[bin_start:bin_end]))
            bin_start = bin_end
x_discharge_rangedates = np.array(x_discharge_rangedates)
y_discharge_mins = np.array(y_discharge_mins)
y_discharge_maxes = np.array(y_discharge_maxes)
y_discharge_mins[y_discharge_mins > np.nanmean(y_discharge_maxes)] = np.nan
y_discharge_maxes[y_discharge_maxes < np.nanmean(y_discharge_mins)] = np.nan
y_discharge_ranges = y_discharge_maxes - y_discharge_mins

# High Toss
bin_start = 0
x_HighToss_rangedates = []    
y_HighToss_mins = []
y_HighToss_maxes = []
for bin_index in range(len(x_HighToss_datenum)):
    datestart = x_HighToss_datenum[bin_start]
    dateend = datestart + (x_HighToss_datenum[bin_index] - x_HighToss_datenum[bin_start])
    date_interval = dateend - datestart
    bin_end = bin_index
    if (date_interval >= tidal_peaktopeak_interval):
            x_HighToss_rangedates.append(x_HighToss_datenum[int((bin_start+bin_end)/2)])
            y_HighToss_mins.append(np.nanmin(y_HighToss[bin_start:bin_end]))
            y_HighToss_maxes.append(np.nanmax(y_HighToss[bin_start:bin_end]))
            bin_start = bin_end
x_HighToss_rangedates = np.array(x_HighToss_rangedates)
y_HighToss_mins = np.array(y_HighToss_mins)
y_HighToss_maxes = np.array(y_HighToss_maxes)
y_HighToss_mins[y_HighToss_mins > np.nanmean(y_HighToss_maxes)] = np.nan
y_HighToss_maxes[y_HighToss_maxes < np.nanmean(y_HighToss_mins)] = np.nan
y_HighToss_ranges = y_HighToss_maxes - y_HighToss_mins

# CNR U/S
bin_start = 0
x_CNRUS_rangedates = []    
y_CNRUS_mins = []
y_CNRUS_maxes = []
for bin_index in range(len(x_CNRUS_datenum)):
    datestart = x_CNRUS_datenum[bin_start]
    dateend = datestart + (x_CNRUS_datenum[bin_index] - x_CNRUS_datenum[bin_start])
    date_interval = dateend - datestart
    bin_end = bin_index
    if (date_interval >= tidal_peaktopeak_interval):
            x_CNRUS_rangedates.append(x_CNRUS_datenum[int((bin_start+bin_end)/2)])
            y_CNRUS_mins.append(np.nanmin(y_CNRUS[bin_start:bin_end]))
            y_CNRUS_maxes.append(np.nanmax(y_CNRUS[bin_start:bin_end]))
            bin_start = bin_end
x_CNRUS_rangedates = np.array(x_CNRUS_rangedates)
y_CNRUS_mins = np.array(y_CNRUS_mins)
y_CNRUS_maxes = np.array(y_CNRUS_maxes)
y_CNRUS_mins[y_CNRUS_mins > np.nanmean(y_CNRUS_maxes)] = np.nan
y_CNRUS_maxes[y_CNRUS_maxes < np.nanmean(y_CNRUS_mins)] = np.nan
y_CNRUS_ranges = y_CNRUS_maxes - y_CNRUS_mins

# Dog Leg
bin_start = 0
x_DogLeg_rangedates = []    
y_DogLeg_mins = []
y_DogLeg_maxes = []
for bin_index in range(len(x_DogLeg_datenum)):
    datestart = x_DogLeg_datenum[bin_start]
    dateend = datestart + (x_DogLeg_datenum[bin_index] - x_DogLeg_datenum[bin_start])
    date_interval = dateend - datestart
    bin_end = bin_index
    if (date_interval >= tidal_peaktopeak_interval):
            x_DogLeg_rangedates.append(x_DogLeg_datenum[int((bin_start+bin_end)/2)])
            y_DogLeg_mins.append(np.nanmin(y_DogLeg[bin_start:bin_end]))
            y_DogLeg_maxes.append(np.nanmax(y_DogLeg[bin_start:bin_end]))
            bin_start = bin_end
x_DogLeg_rangedates = np.array(x_DogLeg_rangedates)
y_DogLeg_mins = np.array(y_DogLeg_mins)
y_DogLeg_maxes = np.array(y_DogLeg_maxes)
y_DogLeg_mins[y_DogLeg_mins > np.nanmean(y_DogLeg_maxes)] = np.nan
y_DogLeg_maxes[y_DogLeg_maxes < np.nanmean(y_DogLeg_mins)] = np.nan
y_DogLeg_ranges = y_DogLeg_maxes - y_DogLeg_mins

# Old Saw
bin_start = 0
x_OldSaw_rangedates = []    
y_OldSaw_mins = []
y_OldSaw_maxes = []
for bin_index in range(len(x_OldSaw_datenum)):
    datestart = x_OldSaw_datenum[bin_start]
    dateend = datestart + (x_OldSaw_datenum[bin_index] - x_OldSaw_datenum[bin_start])
    date_interval = dateend - datestart
    bin_end = bin_index
    if (date_interval >= tidal_peaktopeak_interval):
            x_OldSaw_rangedates.append(x_OldSaw_datenum[int((bin_start+bin_end)/2)])
            y_OldSaw_mins.append(np.nanmin(y_OldSaw[bin_start:bin_end]))
            y_OldSaw_maxes.append(np.nanmax(y_OldSaw[bin_start:bin_end]))
            bin_start = bin_end
x_OldSaw_rangedates = np.array(x_OldSaw_rangedates)
y_OldSaw_mins = np.array(y_OldSaw_mins)
y_OldSaw_maxes = np.array(y_OldSaw_maxes)
y_OldSaw_mins[y_OldSaw_mins > np.nanmean(y_OldSaw_maxes)] = np.nan
y_OldSaw_maxes[y_OldSaw_maxes < np.nanmean(y_OldSaw_mins)] = np.nan
y_OldSaw_ranges = y_OldSaw_maxes - y_OldSaw_mins


# Max and Min Plots
plt.figure()
pylab.plot(x_oceanside_rangedates, y_oceanside_mins, 'o', markersize=1)
pylab.plot(x_oceanside_rangedates, y_oceanside_maxes, 'o', markersize=1)
pylab.plot(x_HRside_rangedates, y_HRside_mins, 'o', markersize=1)
pylab.plot(x_HRside_rangedates, y_HRside_maxes, 'o', markersize=1)
pylab.plot(x_HighToss_rangedates, y_HighToss_mins, 'o', markersize=1)
pylab.plot(x_HighToss_rangedates, y_HighToss_maxes, 'o', markersize=1)
pylab.plot(x_CNRUS_rangedates, y_CNRUS_mins, 'o', markersize=1)
pylab.plot(x_CNRUS_rangedates, y_CNRUS_maxes, 'o', markersize=1)
pylab.plot(x_DogLeg_rangedates, y_DogLeg_mins, 'o', markersize=1)
pylab.plot(x_DogLeg_rangedates, y_DogLeg_maxes, 'o', markersize=1)
pylab.plot(x_OldSaw_rangedates, y_OldSaw_mins, 'o', markersize=1)
pylab.plot(x_OldSaw_rangedates, y_OldSaw_maxes, 'o', markersize=1)

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=18)
plt.ylabel('Elevation (m)', fontsize=16)

pylab.show()

# Range Plots
plt.figure()
pylab.plot(x_OldSaw_rangedates, y_OldSaw_ranges, 'o', markersize=1, label='Old Saw')
pylab.plot(x_oceanside_rangedates, y_oceanside_ranges, 'o', markersize=1, label='Ocean side of dike')
pylab.plot(x_HRside_rangedates, y_HRside_ranges, 'o', markersize=1, label='HR side of dike')
pylab.plot(x_CNRUS_rangedates, y_CNRUS_ranges, 'o', markersize=1, label='CNR U/S')
pylab.plot(x_DogLeg_rangedates, y_DogLeg_ranges, 'o', markersize=1, label='Dog Leg')
pylab.plot(x_HighToss_rangedates, y_HighToss_ranges, 'o', markersize=1, label='High Toss')

loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=18)
plt.ylabel('Tidally Influenced Range (m)', fontsize=16)
plt.legend()

pylab.show()

#%% Mean Ranges and Standard Deviations (extracted from individual tidal cycles)

y_oceanside_meanrange = np.nanmean(y_oceanside_ranges) # meters
y_oceanside_stdrange = np.nanstd(y_oceanside_ranges) # meters
idx_oceanside_ranges = np.isfinite(x_oceanside_rangedates) & np.isfinite(y_oceanside_ranges)
z_oceanside_ranges = np.polyfit(x_oceanside_rangedates[idx_oceanside_ranges], y_oceanside_ranges[idx_oceanside_ranges], 1)
p_oceanside_ranges = np.poly1d(z_oceanside_ranges)
y_HRside_meanrange = np.nanmean(y_HRside_ranges) # meters
y_HRside_stdrange = np.nanstd(y_HRside_ranges) # meters
idx_HRside_ranges = np.isfinite(x_HRside_rangedates) & np.isfinite(y_HRside_ranges)
z_HRside_ranges = np.polyfit(x_HRside_rangedates[idx_HRside_ranges], y_HRside_ranges[idx_HRside_ranges], 1)
p_HRside_ranges = np.poly1d(z_HRside_ranges)
y_HighToss_meanrange = np.nanmean(y_HighToss_ranges) # meters
y_HighToss_stdrange = np.nanstd(y_HighToss_ranges) # meters
idx_HighToss_ranges = np.isfinite(x_HighToss_rangedates) & np.isfinite(y_HighToss_ranges)
z_HighToss_ranges = np.polyfit(x_HighToss_rangedates[idx_HighToss_ranges], y_HighToss_ranges[idx_HighToss_ranges], 1)
p_HighToss_ranges = np.poly1d(z_HighToss_ranges)
y_CNRUS_meanrange = np.nanmean(y_CNRUS_ranges) # meters
y_CNRUS_stdrange = np.nanstd(y_CNRUS_ranges) # meters
idx_CNRUS_ranges = np.isfinite(x_CNRUS_rangedates) & np.isfinite(y_CNRUS_ranges)
z_CNRUS_ranges = np.polyfit(x_CNRUS_rangedates[idx_CNRUS_ranges], y_CNRUS_ranges[idx_CNRUS_ranges], 1)
p_CNRUS_ranges = np.poly1d(z_CNRUS_ranges)
y_DogLeg_meanrange = np.nanmean(y_DogLeg_ranges) # meters
y_DogLeg_stdrange = np.nanstd(y_DogLeg_ranges) # meters
idx_DogLeg_ranges = np.isfinite(x_DogLeg_rangedates) & np.isfinite(y_DogLeg_ranges)
z_DogLeg_ranges = np.polyfit(x_DogLeg_rangedates[idx_DogLeg_ranges], y_DogLeg_ranges[idx_DogLeg_ranges], 1)
p_DogLeg_ranges = np.poly1d(z_DogLeg_ranges)
y_OldSaw_meanrange = np.nanmean(y_OldSaw_ranges) # meters
y_OldSaw_stdrange = np.nanstd(y_OldSaw_ranges) # meters
idx_OldSaw_ranges = np.isfinite(x_OldSaw_rangedates) & np.isfinite(y_OldSaw_ranges)
z_OldSaw_ranges = np.polyfit(x_OldSaw_rangedates[idx_OldSaw_ranges], y_OldSaw_ranges[idx_OldSaw_ranges], 1)
p_OldSaw_ranges = np.poly1d(z_OldSaw_ranges)

# Plots
plt.figure()
polyX_OldSaw_ranges = np.linspace(x_OldSaw_rangedates.min(), x_OldSaw_rangedates.max(), 100)
pylab.plot(polyX_OldSaw_ranges,p_OldSaw_ranges(polyX_OldSaw_ranges),"blue", label='Old Saw')
polyX_oceanside_ranges = np.linspace(x_oceanside_rangedates.min(), x_oceanside_rangedates.max(), 100)
pylab.plot(polyX_oceanside_ranges,p_oceanside_ranges(polyX_oceanside_ranges),"red", label='Ocean side of dike')
polyX_HRside_ranges = np.linspace(x_HRside_rangedates.min(), x_HRside_rangedates.max(), 100)
pylab.plot(polyX_HRside_ranges,p_HRside_ranges(polyX_HRside_ranges),"green", label='HR side of dike')
polyX_CNRUS_ranges = np.linspace(x_CNRUS_rangedates.min(), x_CNRUS_rangedates.max(), 100)
pylab.plot(polyX_CNRUS_ranges,p_CNRUS_ranges(polyX_CNRUS_ranges),"purple", label='CNR U/S')
polyX_DogLeg_ranges = np.linspace(x_DogLeg_rangedates.min(), x_DogLeg_rangedates.max(), 100)
pylab.plot(polyX_DogLeg_ranges,p_DogLeg_ranges(polyX_DogLeg_ranges),"orange", label='Dog Leg')
polyX_HighToss_ranges = np.linspace(x_HighToss_rangedates.min(), x_HighToss_rangedates.max(), 100)
pylab.plot(polyX_HighToss_ranges,p_HighToss_ranges(polyX_HighToss_ranges),"brown", label='High Toss')
# the line equation:
print("y=%.6fx+(%.6f)"%(z_OldSaw_ranges[0],z_OldSaw_ranges[1]))
print("y=%.6fx+(%.6f)"%(z_oceanside_ranges[0],z_oceanside_ranges[1]))
print("y=%.6fx+(%.6f)"%(z_HRside_ranges[0],z_HRside_ranges[1]))
print("y=%.6fx+(%.6f)"%(z_CNRUS_ranges[0],z_CNRUS_ranges[1]))
print("y=%.6fx+(%.6f)"%(z_DogLeg_ranges[0],z_DogLeg_ranges[1]))
print("y=%.6fx+(%.6f)"%(z_HighToss_ranges[0],z_HighToss_ranges[1]))

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mean Tidally Influenced Range (m)', fontsize=16)
plt.legend()

pylab.show()

# G.Earth water-path distances between sensors. 0 km at Old Saw -> oceanside -> HRside -> CNR U/S -> Dog Leg -> High Toss
sensor_displacement_km = np.array([0.00, 1.94, 1.97, 2.08, 3.48, 4.19])

# All CTD sensors use approx. same date range (High Toss is representative)
y_oceanside_range_july2017 = p_oceanside_ranges(polyX_HighToss_ranges).min()
y_oceanside_range_july2019 = p_oceanside_ranges(polyX_HighToss_ranges).max()
y_HRside_range_july2017 = p_HRside_ranges(polyX_HighToss_ranges).min()
y_HRside_range_july2019 = p_HRside_ranges(polyX_HighToss_ranges).max()
y_HighToss_range_july2017 = p_HighToss_ranges(polyX_HighToss_ranges.min())
y_HighToss_range_july2019 = p_HighToss_ranges(polyX_HighToss_ranges.max())
y_CNRUS_range_july2017 = p_CNRUS_ranges(polyX_HighToss_ranges.min())
y_CNRUS_range_july2019 = p_CNRUS_ranges(polyX_HighToss_ranges.max())
y_DogLeg_range_july2017 = p_DogLeg_ranges(polyX_HighToss_ranges.min())
y_DogLeg_range_july2019 = p_DogLeg_ranges(polyX_HighToss_ranges.max())
y_OldSaw_range_july2017 = p_OldSaw_ranges(polyX_HighToss_ranges.min())
y_OldSaw_range_july2019 = p_OldSaw_ranges(polyX_HighToss_ranges.max())

july2017_ranges = np.vstack((sensor_displacement_km, np.array([y_OldSaw_meanrange, y_oceanside_range_july2017, y_HRside_range_july2017, 
                                                     y_CNRUS_range_july2017, y_DogLeg_range_july2017, y_HighToss_range_july2017]))).T
july2019_ranges = np.vstack((sensor_displacement_km, np.array([y_OldSaw_meanrange, y_oceanside_range_july2019, y_HRside_range_july2019, 
                                                     y_CNRUS_range_july2019, y_DogLeg_range_july2019, y_HighToss_range_july2019]))).T

x_july2017_ranges, y_july2017_ranges = july2017_ranges.T
x_july2019_ranges, y_july2019_ranges = july2019_ranges.T

# adding/subtracting half the standard deviation from the ranges (not realistic as the stdev is skewed more towards the maxes)
y_july2017_ranges_plus1std = y_july2017_ranges + 0.5*np.array([y_OldSaw_stdrange, y_oceanside_stdrange, y_HRside_stdrange, 
                                                               y_CNRUS_stdrange, y_DogLeg_stdrange, y_HighToss_stdrange])
y_july2017_ranges_minus1std = y_july2017_ranges - 0.5*np.array([y_OldSaw_stdrange, y_oceanside_stdrange, y_HRside_stdrange, 
                                                               y_CNRUS_stdrange, y_DogLeg_stdrange, y_HighToss_stdrange])
    
y_july2019_ranges_plus1std = y_july2019_ranges + 0.5*np.array([y_OldSaw_stdrange, y_oceanside_stdrange, y_HRside_stdrange, 
                                                               y_CNRUS_stdrange, y_DogLeg_stdrange, y_HighToss_stdrange])
y_july2019_ranges_minus1std = y_july2019_ranges - 0.5*np.array([y_OldSaw_stdrange, y_oceanside_stdrange, y_HRside_stdrange, 
                                                               y_CNRUS_stdrange, y_DogLeg_stdrange, y_HighToss_stdrange])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.margins(0)
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
ax1.grid(True, which='both')
ax1.plot(x_july2017_ranges, y_july2017_ranges, label='July 2017 & 1 sigma stdev')
ax1.fill_between(x_july2017_ranges, y_july2017_ranges_plus1std, y_july2017_ranges_minus1std, facecolor='blue', alpha=0.5)
ax1.plot(x_july2019_ranges, y_july2019_ranges, label='July 2019 & 1 sigma stdev')
ax1.fill_between(x_july2019_ranges, y_july2019_ranges_plus1std, y_july2019_ranges_minus1std, facecolor='orange', alpha=0.5)
ax1.legend(loc='best', bbox_to_anchor=(0.3, 0.1, 0.5, 0.5))
ax1.set_xlabel(r"Water-path Distance from Old Saw (km)")
ax1.set_ylabel(r"Tidally-Influenced Mean Range (m)")
my_xticks = np.array(['Old Saw Mean (6 mo)', 'HR Dike Ocean-side', 'HR Dike HR-side', 'CNR U/S', 'Dog Leg', 'High Toss'])
ax2.set_xticks(x_july2017_ranges)
ax2.set_xticklabels(my_xticks, rotation=55, horizontalalignment='left', fontsize='x-small')
plt.show()
plt.tight_layout()

#%% Standard deviations of maxes/mins/means, amplitudes (max mean to mean) and mean ranges (max mean to min mean)

stdev_oceanside = y_oceanside_nonans.std()
stdev_oceanside_maxlevels = y_oceanside_maxlevels.std()
stdev_oceanside_minlevels = y_oceanside_minlevels.std()
stdev_HRside = y_HRside_nonans.std()
stdev_HRside_maxlevels = y_HRside_maxlevels.std()
stdev_HRside_minlevels = y_HRside_minlevels.std()
stdev_HighToss = y_HighToss_nonans.std()
stdev_HighToss_maxlevels = y_HighToss_maxlevels.std()
stdev_HighToss_minlevels = y_HighToss_minlevels.std()
stdev_CNRUS = y_CNRUS_nonans.std()
stdev_CNRUS_maxlevels = y_CNRUS_maxlevels.std()
stdev_CNRUS_minlevels = y_CNRUS_minlevels.std()
stdev_DogLeg = y_DogLeg_nonans.std()
stdev_DogLeg_maxlevels = y_DogLeg_maxlevels.std()
stdev_DogLeg_minlevels = y_DogLeg_minlevels.std()
stdev_OldSaw = y_OldSaw_nonans.std()
stdev_OldSaw_maxlevels = y_OldSaw_maxlevels.std()
stdev_OldSaw_minlevels  = y_OldSaw_minlevels.std()

amp_HighToss = p_HighToss_max(polyX_HighToss_max) - p_HighToss(polyX_HighToss)
amp_HighToss_avg = amp_HighToss.mean()
range_HighToss = p_HighToss_max(polyX_HighToss_max) - p_HighToss_min(polyX_HighToss_min)
range_HighToss_avg = range_HighToss.mean()
amp_MillCreek = p_MillCreek_max(polyX_MillCreek_max) - p_MillCreek(polyX_MillCreek)
amp_MillCreek_avg = amp_MillCreek.mean()
range_MillCreek = p_MillCreek_max(polyX_MillCreek_max) - p_MillCreek_min(polyX_MillCreek_min)
range_MillCreek_avg = range_MillCreek.mean()
amp_CNRUS = p_CNRUS_max(polyX_CNRUS_max) - p_CNRUS(polyX_CNRUS)
amp_CNRUS_avg = amp_CNRUS.mean()
range_CNRUS = p_CNRUS_max(polyX_CNRUS_max) - p_CNRUS_min(polyX_CNRUS_min)
range_CNRUS_avg = range_CNRUS.mean()
amp_DogLeg = p_DogLeg_max(polyX_DogLeg_max) - p_DogLeg(polyX_DogLeg)
amp_DogLeg_avg = amp_DogLeg.mean()
range_DogLeg = p_DogLeg_max(polyX_DogLeg_max) - p_DogLeg_min(polyX_DogLeg_min)
range_DogLeg_avg = range_DogLeg.mean()
amp_OldSaw = p_OldSaw_max(polyX_OldSaw_max) - p_OldSaw(polyX_OldSaw)
amp_OldSaw_avg = amp_OldSaw.mean()
range_OldSaw = p_OldSaw_max(polyX_OldSaw_max) - p_OldSaw_min(polyX_OldSaw_min)
range_OldSaw_avg = range_OldSaw.mean()

print("Average tidal range between July 2017 and July 2019 at High Toss is ", ("%.3f"%(range_HighToss_avg)), " m.")
print("Average seasonal range between July 2017 and July 2019 at Mill Creek is ", ("%.3f"%(range_MillCreek_avg)), " m.")
print("Average tidal range between July 2017 and July 2019 at CNR U/S is ", ("%.3f"%(range_CNRUS_avg)), " m.")
print("Average tidal range between July 2017 and July 2019 at Dog Leg is ", ("%.3f"%(range_DogLeg_avg)), " m.")
print("Average tidal range between June 2018 and December 2018 at Old Saw is ", ("%.3f"%(range_OldSaw_avg)), " m.")

# G.Earth water-path distances between sensors. 0 km at Old Saw -> oceanside -> HRside -> CNR U/S -> Dog Leg -> High Toss
sensor_displacement_km = np.array([0.00, 1.94, 1.97, 2.08, 3.48, 4.19])

range_HighToss_june2015 = p_HighToss_max(polyX_HRside_max).min() - p_HighToss_min(polyX_HRside_min).min()
range_HighToss_july2019 = p_HighToss_max(polyX_HRside_max).max() - p_HighToss_min(polyX_HRside_min).max()
range_CNRUS_june2015 = p_CNRUS_max(polyX_HRside_max).min() - p_CNRUS_min(polyX_HRside_min).min()
range_CNRUS_july2019 = p_CNRUS_max(polyX_HRside_max).max() - p_CNRUS_min(polyX_HRside_min).max()
range_DogLeg_june2015 = p_DogLeg_max(polyX_HRside_max).min() - p_DogLeg_min(polyX_HRside_min).min()
range_DogLeg_july2019 = p_DogLeg_max(polyX_HRside_max).max() - p_DogLeg_min(polyX_HRside_min).max()
range_OldSaw_june2015 = p_OldSaw_max(polyX_HRside_max).min() - p_OldSaw_min(polyX_HRside_min).min()
range_OldSaw_july2019 = p_OldSaw_max(polyX_HRside_max).max() - p_OldSaw_min(polyX_HRside_min).max()

june2015_ranges = np.vstack((sensor_displacement_km, np.array([range_OldSaw_avg, range_oceanside_june2015, range_HRside_june2015, 
                                                     range_CNRUS_june2015, range_DogLeg_june2015, range_HighToss_june2015]))).T
july2019_ranges = np.vstack((sensor_displacement_km, np.array([range_OldSaw_avg, range_oceanside_july2019, range_HRside_july2019, 
                                                     range_CNRUS_july2019, range_DogLeg_july2019, range_HighToss_july2019]))).T

x_june2015_ranges, y_june2015_ranges = june2015_ranges.T
x_july2019_ranges, y_july2019_ranges = july2019_ranges.T

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.margins(0)
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
ax1.grid(True, which='both')
ax1.plot(x_june2015_ranges, y_june2015_ranges, label='June 2015')
ax1.plot(x_july2019_ranges, y_july2019_ranges, label='July 2019')
ax1.legend(loc='best', bbox_to_anchor=(0.3, 0.1, 0.5, 0.5))
ax1.set_xlabel(r"Water-path Distance from Old Saw (km)")
ax1.set_ylabel(r"Tidally-Influenced Mean Range (m)")
my_xticks = np.array(['Old Saw Mean (6 mo)', 'HR Dike Ocean-side (2 yr)', 'HR Dike HR-side (4 yr)', 'CNR U/S (2 yr)', 'Dog Leg (2 yr)', 'High Toss (2 yr)'])
ax2.set_xticks(x_june2015_ranges)
ax2.set_xticklabels(my_xticks, rotation=55, horizontalalignment='left', fontsize='x-small')
plt.show()
plt.tight_layout()

#%% Monthly Bins for Seasonal Averages

# HR Side of Dike
January_HRside = []
February_HRside = []
March_HRside = []
April_HRside = []
May_HRside = []
June_HRside = []
July_HRside = []
August_HRside = []
September_HRside = []
October_HRside = []
November_HRside = []
December_HRside = []
for i in range(len(dates_HRside_nonans)):
    if dates_HRside_nonans[i].month == 1: 
        January_HRside.append(y_HRside_nonans[i])
    if dates_HRside_nonans[i].month == 2: 
        February_HRside.append(y_HRside_nonans[i])
    if dates_HRside_nonans[i].month == 3: 
        March_HRside.append(y_HRside_nonans[i])
    if dates_HRside_nonans[i].month == 4: 
        April_HRside.append(y_HRside_nonans[i])
    if dates_HRside_nonans[i].month == 5: 
        May_HRside.append(y_HRside_nonans[i])
    if dates_HRside_nonans[i].month == 6:
        June_HRside.append(y_HRside_nonans[i])
    if dates_HRside_nonans[i].month == 7:
        July_HRside.append(y_HRside_nonans[i])
    if dates_HRside_nonans[i].month == 8: 
        August_HRside.append(y_HRside_nonans[i])
    if dates_HRside_nonans[i].month == 9: 
        September_HRside.append(y_HRside_nonans[i])
    if dates_HRside_nonans[i].month == 10: 
        October_HRside.append(y_HRside_nonans[i])
    if dates_HRside_nonans[i].month == 11: 
        November_HRside.append(y_HRside_nonans[i])
    if dates_HRside_nonans[i].month == 12: 
        December_HRside.append(y_HRside_nonans[i])
January_HRside = np.array(January_HRside)
February_HRside = np.array(February_HRside)
March_HRside = np.array(March_HRside)
April_HRside = np.array(April_HRside)
May_HRside = np.array(May_HRside)
June_HRside = np.array(June_HRside)
July_HRside = np.array(July_HRside)
August_HRside = np.array(August_HRside)
September_HRside = np.array(September_HRside)
October_HRside = np.array(October_HRside)
November_HRside = np.array(November_HRside)
December_HRside = np.array(December_HRside)

January_HRside_mean = January_HRside.mean()
February_HRside_mean = February_HRside.mean()
March_HRside_mean = March_HRside.mean()
April_HRside_mean = April_HRside.mean()
May_HRside_mean = May_HRside.mean()
June_HRside_mean = June_HRside.mean()
July_HRside_mean = July_HRside.mean()
August_HRside_mean = August_HRside.mean()
September_HRside_mean = September_HRside.mean()
October_HRside_mean = October_HRside.mean()
November_HRside_mean = November_HRside.mean()
December_HRside_mean = December_HRside.mean()

y_HRside_monthly_means = np.array([January_HRside_mean,February_HRside_mean, March_HRside_mean, April_HRside_mean, May_HRside_mean, 
                          June_HRside_mean, July_HRside_mean, August_HRside_mean, September_HRside_mean, October_HRside_mean, 
                          November_HRside_mean, December_HRside_mean])

# HR Side of Dike, Last 2 years
for i in range(len(dates_HRside_nonans)):
    if (dates_HRside_nonans[i].day == 20) & (dates_HRside_nonans[i].month == 7) & (dates_HRside_nonans[i].year == 2017):
        HRside_2yr_startindex = i
        break
dates_HRside_2yr_nonans = dates_HRside_nonans[HRside_2yr_startindex:]
y_HRside_2yr_nonans = y_HRside_nonans[HRside_2yr_startindex:]

January_HRside_2yr = []
February_HRside_2yr = []
March_HRside_2yr = []
April_HRside_2yr = []
May_HRside_2yr = []
June_HRside_2yr = []
July_HRside_2yr = []
August_HRside_2yr = []
September_HRside_2yr = []
October_HRside_2yr = []
November_HRside_2yr = []
December_HRside_2yr = []
for i in range(len(dates_HRside_2yr_nonans)):
    if dates_HRside_2yr_nonans[i].month == 1: 
        January_HRside_2yr.append(y_HRside_2yr_nonans[i])
    if dates_HRside_2yr_nonans[i].month == 2: 
        February_HRside_2yr.append(y_HRside_2yr_nonans[i])
    if dates_HRside_2yr_nonans[i].month == 3: 
        March_HRside_2yr.append(y_HRside_2yr_nonans[i])
    if dates_HRside_2yr_nonans[i].month == 4: 
        April_HRside_2yr.append(y_HRside_2yr_nonans[i])
    if dates_HRside_2yr_nonans[i].month == 5: 
        May_HRside_2yr.append(y_HRside_2yr_nonans[i])
    if dates_HRside_2yr_nonans[i].month == 6: 
        June_HRside_2yr.append(y_HRside_2yr_nonans[i])
    if dates_HRside_2yr_nonans[i].month == 7: 
        July_HRside_2yr.append(y_HRside_2yr_nonans[i])
    if dates_HRside_2yr_nonans[i].month == 8: 
        August_HRside_2yr.append(y_HRside_2yr_nonans[i])
    if dates_HRside_2yr_nonans[i].month == 9: 
        September_HRside_2yr.append(y_HRside_2yr_nonans[i])
    if dates_HRside_2yr_nonans[i].month == 10: 
        October_HRside_2yr.append(y_HRside_2yr_nonans[i])
    if dates_HRside_2yr_nonans[i].month == 11: 
        November_HRside_2yr.append(y_HRside_2yr_nonans[i])
    if dates_HRside_2yr_nonans[i].month == 12: 
        December_HRside_2yr.append(y_HRside_2yr_nonans[i])
January_HRside_2yr = np.array(January_HRside_2yr)
February_HRside_2yr = np.array(February_HRside_2yr)
March_HRside_2yr = np.array(March_HRside_2yr)
April_HRside_2yr = np.array(April_HRside_2yr)
May_HRside_2yr = np.array(May_HRside_2yr)
June_HRside_2yr = np.array(June_HRside_2yr)
July_HRside_2yr = np.array(July_HRside_2yr)
August_HRside_2yr = np.array(August_HRside_2yr)
September_HRside_2yr = np.array(September_HRside_2yr)
October_HRside_2yr = np.array(October_HRside_2yr)
November_HRside_2yr = np.array(November_HRside_2yr)
December_HRside_2yr = np.array(December_HRside_2yr)

January_HRside_2yr_mean = January_HRside_2yr.mean()
February_HRside_2yr_mean = February_HRside_2yr.mean()
March_HRside_2yr_mean = March_HRside_2yr.mean()
April_HRside_2yr_mean = April_HRside_2yr.mean()
May_HRside_2yr_mean = May_HRside_2yr.mean()
June_HRside_2yr_mean = June_HRside_2yr.mean()
July_HRside_2yr_mean = July_HRside_2yr.mean()
August_HRside_2yr_mean = August_HRside_2yr.mean()
September_HRside_2yr_mean = September_HRside_2yr.mean()
October_HRside_2yr_mean = October_HRside_2yr.mean()
November_HRside_2yr_mean = November_HRside_2yr.mean()
December_HRside_2yr_mean = December_HRside_2yr.mean()

y_HRside_2yr_monthly_means = np.array([January_HRside_2yr_mean,February_HRside_2yr_mean, March_HRside_2yr_mean, April_HRside_2yr_mean, May_HRside_2yr_mean, 
                          June_HRside_2yr_mean, July_HRside_2yr_mean, August_HRside_2yr_mean, September_HRside_2yr_mean, October_HRside_2yr_mean, 
                          November_HRside_2yr_mean, December_HRside_2yr_mean])
    
# CNR U/S
January_CNRUS = []
February_CNRUS = []
March_CNRUS = []
April_CNRUS = []
May_CNRUS = []
June_CNRUS = []
July_CNRUS = []
August_CNRUS = []
September_CNRUS = []
October_CNRUS = []
November_CNRUS = []
December_CNRUS = []
for i in range(len(dates_CNRUS_nonans)):
    if dates_CNRUS_nonans[i].month == 1: 
        January_CNRUS.append(y_CNRUS_nonans[i])
    if dates_CNRUS_nonans[i].month == 2: 
        February_CNRUS.append(y_CNRUS_nonans[i])
    if dates_CNRUS_nonans[i].month == 3: 
        March_CNRUS.append(y_CNRUS_nonans[i])
    if dates_CNRUS_nonans[i].month == 4: 
        April_CNRUS.append(y_CNRUS_nonans[i])
    if dates_CNRUS_nonans[i].month == 5: 
        May_CNRUS.append(y_CNRUS_nonans[i])
    if dates_CNRUS_nonans[i].month == 6: 
        June_CNRUS.append(y_CNRUS_nonans[i])
    if dates_CNRUS_nonans[i].month == 7: 
        July_CNRUS.append(y_CNRUS_nonans[i])
    if dates_CNRUS_nonans[i].month == 8: 
        August_CNRUS.append(y_CNRUS_nonans[i])
    if dates_CNRUS_nonans[i].month == 9: 
        September_CNRUS.append(y_CNRUS_nonans[i])
    if dates_CNRUS_nonans[i].month == 10: 
        October_CNRUS.append(y_CNRUS_nonans[i])
    if dates_CNRUS_nonans[i].month == 11: 
        November_CNRUS.append(y_CNRUS_nonans[i])
    if dates_CNRUS_nonans[i].month == 12: 
        December_CNRUS.append(y_CNRUS_nonans[i])
January_CNRUS = np.array(January_CNRUS)
February_CNRUS = np.array(February_CNRUS)
March_CNRUS = np.array(March_CNRUS)
April_CNRUS = np.array(April_CNRUS)
May_CNRUS = np.array(May_CNRUS)
June_CNRUS = np.array(June_CNRUS)
July_CNRUS = np.array(July_CNRUS)
August_CNRUS = np.array(August_CNRUS)
September_CNRUS = np.array(September_CNRUS)
October_CNRUS = np.array(October_CNRUS)
November_CNRUS = np.array(November_CNRUS)
December_CNRUS = np.array(December_CNRUS)

January_CNRUS_mean = January_CNRUS.mean()
February_CNRUS_mean = February_CNRUS.mean()
March_CNRUS_mean = March_CNRUS.mean()
April_CNRUS_mean = April_CNRUS.mean()
May_CNRUS_mean = May_CNRUS.mean()
June_CNRUS_mean = June_CNRUS.mean()
July_CNRUS_mean = July_CNRUS.mean()
August_CNRUS_mean = August_CNRUS.mean()
September_CNRUS_mean = September_CNRUS.mean()
October_CNRUS_mean = October_CNRUS.mean()
November_CNRUS_mean = November_CNRUS.mean()
December_CNRUS_mean = December_CNRUS.mean()

y_CNRUS_monthly_means = np.array([January_CNRUS_mean,February_CNRUS_mean, March_CNRUS_mean, April_CNRUS_mean, May_CNRUS_mean, 
                          June_CNRUS_mean, July_CNRUS_mean, August_CNRUS_mean, September_CNRUS_mean, October_CNRUS_mean, 
                          November_CNRUS_mean, December_CNRUS_mean])

# Dog Leg
January_DogLeg = []
February_DogLeg = []
March_DogLeg = []
April_DogLeg = []
May_DogLeg = []
June_DogLeg = []
July_DogLeg = []
August_DogLeg = []
September_DogLeg = []
October_DogLeg = []
November_DogLeg = []
December_DogLeg = []
for i in range(len(dates_DogLeg_nonans)):
    if dates_DogLeg_nonans[i].month == 1: 
        January_DogLeg.append(y_DogLeg_nonans[i])
    if dates_DogLeg_nonans[i].month == 2: 
        February_DogLeg.append(y_DogLeg_nonans[i])
    if dates_DogLeg_nonans[i].month == 3: 
        March_DogLeg.append(y_DogLeg_nonans[i])
    if dates_DogLeg_nonans[i].month == 4: 
        April_DogLeg.append(y_DogLeg_nonans[i])
    if dates_DogLeg_nonans[i].month == 5: 
        May_DogLeg.append(y_DogLeg_nonans[i])
    if dates_DogLeg_nonans[i].month == 6: 
        June_DogLeg.append(y_DogLeg_nonans[i])
    if dates_DogLeg_nonans[i].month == 7: 
        July_DogLeg.append(y_DogLeg_nonans[i])
    if dates_DogLeg_nonans[i].month == 8: 
        August_DogLeg.append(y_DogLeg_nonans[i])
    if dates_DogLeg_nonans[i].month == 9: 
        September_DogLeg.append(y_DogLeg_nonans[i])
    if dates_DogLeg_nonans[i].month == 10: 
        October_DogLeg.append(y_DogLeg_nonans[i])
    if dates_DogLeg_nonans[i].month == 11: 
        November_DogLeg.append(y_DogLeg_nonans[i])
    if dates_DogLeg_nonans[i].month == 12: 
        December_DogLeg.append(y_DogLeg_nonans[i])
January_DogLeg = np.array(January_DogLeg)
February_DogLeg = np.array(February_DogLeg)
March_DogLeg = np.array(March_DogLeg)
April_DogLeg = np.array(April_DogLeg)
May_DogLeg = np.array(May_DogLeg)
June_DogLeg = np.array(June_DogLeg)
July_DogLeg = np.array(July_DogLeg)
August_DogLeg = np.array(August_DogLeg)
September_DogLeg = np.array(September_DogLeg)
October_DogLeg = np.array(October_DogLeg)
November_DogLeg = np.array(November_DogLeg)
December_DogLeg = np.array(December_DogLeg)

January_DogLeg_mean = January_DogLeg.mean()
February_DogLeg_mean = February_DogLeg.mean()
March_DogLeg_mean = March_DogLeg.mean()
April_DogLeg_mean = April_DogLeg.mean()
May_DogLeg_mean = May_DogLeg.mean()
June_DogLeg_mean = June_DogLeg.mean()
July_DogLeg_mean = July_DogLeg.mean()
August_DogLeg_mean = August_DogLeg.mean()
September_DogLeg_mean = September_DogLeg.mean()
October_DogLeg_mean = October_DogLeg.mean()
November_DogLeg_mean = November_DogLeg.mean()
December_DogLeg_mean = December_DogLeg.mean()

y_DogLeg_monthly_means = np.array([January_DogLeg_mean,February_DogLeg_mean, March_DogLeg_mean, April_DogLeg_mean, May_DogLeg_mean, 
                          June_DogLeg_mean, July_DogLeg_mean, August_DogLeg_mean, September_DogLeg_mean, October_DogLeg_mean, 
                          November_DogLeg_mean, December_DogLeg_mean])

# High Toss
January_HighToss = []
February_HighToss = []
March_HighToss = []
April_HighToss = []
May_HighToss = []
June_HighToss = []
July_HighToss = []
August_HighToss = []
September_HighToss = []
October_HighToss = []
November_HighToss = []
December_HighToss = []
for i in range(len(dates_HighToss_nonans)):
    if dates_HighToss_nonans[i].month == 1: 
        January_HighToss.append(y_HighToss_nonans[i])
    if dates_HighToss_nonans[i].month == 2: 
        February_HighToss.append(y_HighToss_nonans[i])
    if dates_HighToss_nonans[i].month == 3: 
        March_HighToss.append(y_HighToss_nonans[i])
    if dates_HighToss_nonans[i].month == 4: 
        April_HighToss.append(y_HighToss_nonans[i])
    if dates_HighToss_nonans[i].month == 5: 
        May_HighToss.append(y_HighToss_nonans[i])
    if dates_HighToss_nonans[i].month == 6: 
        June_HighToss.append(y_HighToss_nonans[i])
    if dates_HighToss_nonans[i].month == 7: 
        July_HighToss.append(y_HighToss_nonans[i])
    if dates_HighToss_nonans[i].month == 8: 
        August_HighToss.append(y_HighToss_nonans[i])
    if dates_HighToss_nonans[i].month == 9: 
        September_HighToss.append(y_HighToss_nonans[i])
    if dates_HighToss_nonans[i].month == 10: 
        October_HighToss.append(y_HighToss_nonans[i])
    if dates_HighToss_nonans[i].month == 11: 
        November_HighToss.append(y_HighToss_nonans[i])
    if dates_HighToss_nonans[i].month == 12: 
        December_HighToss.append(y_HighToss_nonans[i])
January_HighToss = np.array(January_HighToss)
February_HighToss = np.array(February_HighToss)
March_HighToss = np.array(March_HighToss)
April_HighToss = np.array(April_HighToss)
May_HighToss = np.array(May_HighToss)
June_HighToss = np.array(June_HighToss)
July_HighToss = np.array(July_HighToss)
August_HighToss = np.array(August_HighToss)
September_HighToss = np.array(September_HighToss)
October_HighToss = np.array(October_HighToss)
November_HighToss = np.array(November_HighToss)
December_HighToss = np.array(December_HighToss)

January_HighToss_mean = January_HighToss.mean()
February_HighToss_mean = February_HighToss.mean()
March_HighToss_mean = March_HighToss.mean()
April_HighToss_mean = April_HighToss.mean()
May_HighToss_mean = May_HighToss.mean()
June_HighToss_mean = June_HighToss.mean()
July_HighToss_mean = July_HighToss.mean()
August_HighToss_mean = August_HighToss.mean()
September_HighToss_mean = September_HighToss.mean()
October_HighToss_mean = October_HighToss.mean()
November_HighToss_mean = November_HighToss.mean()
December_HighToss_mean = December_HighToss.mean()

y_HighToss_monthly_means = np.array([January_HighToss_mean,February_HighToss_mean, March_HighToss_mean, April_HighToss_mean, May_HighToss_mean, 
                          June_HighToss_mean, July_HighToss_mean, August_HighToss_mean, September_HighToss_mean, October_HighToss_mean, 
                          November_HighToss_mean, December_HighToss_mean])
    
months_str = calendar.month_name
x_months = np.array(months_str[1:])
CNRUS_mask = np.isfinite(y_CNRUS_monthly_means)
DogLeg_mask = np.isfinite(y_CNRUS_monthly_means)
HighToss_mask = np.isfinite(y_CNRUS_monthly_means)

# Plots
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid(True, which='both')
# ax1.plot(x_months, y_HRside_monthly_means, label='HR Side of Dike, 2015-2019')
# ax1.plot(x_months, y_HRside_2yr_monthly_means, label='HR Side of Dike, 2017--2019')
ax1.plot(x_months[CNRUS_mask], y_CNRUS_monthly_means[CNRUS_mask], label='CNR U/S, 2017--2019')
ax1.plot(x_months[DogLeg_mask], y_DogLeg_monthly_means[DogLeg_mask], label='Dog Leg, 2017--2019')
ax1.plot(x_months[HighToss_mask], y_HighToss_monthly_means[HighToss_mask], label='High Toss, 2017--2019')
ax1.legend(loc='best', bbox_to_anchor=(0.3, 0.1, 0.1, 0.3))
ax1.set_xlabel(r"Month")
ax1.set_ylabel(r"Mean Water Elevation (m)")
ax1.set_xticklabels(x_months, rotation=55, horizontalalignment='right')
ax1.margins(0)
plt.tight_layout()
plt.show()

riv_sensor_disp_km = np.array([0.00, 1.23, 1.52]) # linear distance from CNR U/S to Dog Leg, and CNR U/S to High Toss
x_new_riv_sensor_disp_km = np.linspace(riv_sensor_disp_km[0], riv_sensor_disp_km[-1], 50) # for polynomial fit

# January CNR U/S to High Toss
jan_sensor_levels = np.array([January_CNRUS_mean, January_DogLeg_mean, January_HighToss_mean])
# no data for February - use mean of Jan and Mar means.
feb_sensor_levels = np.array([(January_CNRUS_mean+March_CNRUS_mean)/2, (January_DogLeg_mean+March_DogLeg_mean)/2, 
                              (January_HighToss_mean+March_HighToss_mean)/2])
mar_sensor_levels = np.array([March_CNRUS_mean, March_DogLeg_mean, March_HighToss_mean])
apr_sensor_levels = np.array([April_CNRUS_mean, April_DogLeg_mean, April_HighToss_mean])
may_sensor_levels = np.array([May_CNRUS_mean, May_DogLeg_mean, May_HighToss_mean])
jun_sensor_levels = np.array([June_CNRUS_mean, June_DogLeg_mean, June_HighToss_mean])
jul_sensor_levels = np.array([July_CNRUS_mean, July_DogLeg_mean, July_HighToss_mean])
aug_sensor_levels = np.array([August_CNRUS_mean, August_DogLeg_mean, August_HighToss_mean])
sep_sensor_levels = np.array([September_CNRUS_mean, September_DogLeg_mean, September_HighToss_mean])
oct_sensor_levels = np.array([October_CNRUS_mean, October_DogLeg_mean, October_HighToss_mean])
nov_sensor_levels = np.array([November_CNRUS_mean, November_DogLeg_mean, November_HighToss_mean])
dec_sensor_levels = np.array([December_CNRUS_mean, December_DogLeg_mean, December_HighToss_mean])

annual_sensor_levels = np.array([np.nanmean(y_CNRUS_monthly_means), np.nanmean(y_DogLeg_monthly_means), np.nanmean(y_HighToss_monthly_means)])

plt.figure()
pylab.plot(riv_sensor_disp_km, jan_sensor_levels, label='Jan')
pylab.plot(riv_sensor_disp_km, feb_sensor_levels, label='Feb')
pylab.plot(riv_sensor_disp_km, mar_sensor_levels, label='Mar')
pylab.plot(riv_sensor_disp_km, apr_sensor_levels, label='Apr')
pylab.plot(riv_sensor_disp_km, may_sensor_levels, label='May')
pylab.plot(riv_sensor_disp_km, jun_sensor_levels, label='Jun')
pylab.plot(riv_sensor_disp_km, jul_sensor_levels, label='July')
pylab.plot(riv_sensor_disp_km, aug_sensor_levels, label='Aug')
pylab.plot(riv_sensor_disp_km, sep_sensor_levels, label='Sep')
pylab.plot(riv_sensor_disp_km, oct_sensor_levels, label='Oct')
pylab.plot(riv_sensor_disp_km, nov_sensor_levels, label='Nov')
pylab.plot(riv_sensor_disp_km, dec_sensor_levels, label='Dec')
pylab.plot(riv_sensor_disp_km, annual_sensor_levels, label='Annual Mean')
plt.legend()

# calculate polynomials
z_jan_sensor_levels = np.polyfit(riv_sensor_disp_km, jan_sensor_levels, 2)
f_jan_sensor_levels = np.poly1d(z_jan_sensor_levels)
y_new_jan_sensor_levels = f_jan_sensor_levels(x_new_riv_sensor_disp_km)
z_feb_sensor_levels = np.polyfit(riv_sensor_disp_km, feb_sensor_levels, 2)
f_feb_sensor_levels = np.poly1d(z_feb_sensor_levels)
y_new_feb_sensor_levels = f_feb_sensor_levels(x_new_riv_sensor_disp_km)
z_mar_sensor_levels = np.polyfit(riv_sensor_disp_km, mar_sensor_levels, 2)
f_mar_sensor_levels = np.poly1d(z_mar_sensor_levels)
y_new_mar_sensor_levels = f_mar_sensor_levels(x_new_riv_sensor_disp_km)
z_apr_sensor_levels = np.polyfit(riv_sensor_disp_km, apr_sensor_levels, 2)
f_apr_sensor_levels = np.poly1d(z_apr_sensor_levels)
y_new_apr_sensor_levels = f_apr_sensor_levels(x_new_riv_sensor_disp_km)
z_may_sensor_levels = np.polyfit(riv_sensor_disp_km, may_sensor_levels, 2)
f_may_sensor_levels = np.poly1d(z_may_sensor_levels)
y_new_may_sensor_levels = f_may_sensor_levels(x_new_riv_sensor_disp_km)
z_jun_sensor_levels = np.polyfit(riv_sensor_disp_km, jun_sensor_levels, 2)
f_jun_sensor_levels = np.poly1d(z_jun_sensor_levels)
y_new_jun_sensor_levels = f_jun_sensor_levels(x_new_riv_sensor_disp_km)
z_jul_sensor_levels = np.polyfit(riv_sensor_disp_km, jul_sensor_levels, 2)
f_jul_sensor_levels = np.poly1d(z_jul_sensor_levels)
y_new_jul_sensor_levels = f_jul_sensor_levels(x_new_riv_sensor_disp_km)
z_aug_sensor_levels = np.polyfit(riv_sensor_disp_km, aug_sensor_levels, 2)
f_aug_sensor_levels = np.poly1d(z_aug_sensor_levels)
y_new_aug_sensor_levels = f_aug_sensor_levels(x_new_riv_sensor_disp_km)
z_sep_sensor_levels = np.polyfit(riv_sensor_disp_km, sep_sensor_levels, 2)
f_sep_sensor_levels = np.poly1d(z_sep_sensor_levels)
y_new_sep_sensor_levels = f_sep_sensor_levels(x_new_riv_sensor_disp_km)
z_oct_sensor_levels = np.polyfit(riv_sensor_disp_km, oct_sensor_levels, 2)
f_oct_sensor_levels = np.poly1d(z_oct_sensor_levels)
y_new_oct_sensor_levels = f_oct_sensor_levels(x_new_riv_sensor_disp_km)
z_nov_sensor_levels = np.polyfit(riv_sensor_disp_km, nov_sensor_levels, 2)
f_nov_sensor_levels = np.poly1d(z_nov_sensor_levels)
y_new_nov_sensor_levels = f_nov_sensor_levels(x_new_riv_sensor_disp_km)
z_dec_sensor_levels = np.polyfit(riv_sensor_disp_km, dec_sensor_levels, 2)
f_dec_sensor_levels = np.poly1d(z_dec_sensor_levels)
y_new_dec_sensor_levels = f_dec_sensor_levels(x_new_riv_sensor_disp_km)
z_annual_sensor_levels = np.polyfit(riv_sensor_disp_km, annual_sensor_levels, 2)
f_annual_sensor_levels = np.poly1d(z_annual_sensor_levels)
y_new_annual_sensor_levels = f_annual_sensor_levels(x_new_riv_sensor_disp_km)

# plot polynomials
plt.figure()
plt.plot(x_new_riv_sensor_disp_km, y_new_jan_sensor_levels, 'x', label='Jan')
plt.plot(x_new_riv_sensor_disp_km, y_new_feb_sensor_levels, 'black', label='Feb')
plt.plot(x_new_riv_sensor_disp_km, y_new_mar_sensor_levels, label='Mar')
plt.plot(x_new_riv_sensor_disp_km, y_new_apr_sensor_levels, label='Apr')
plt.plot(x_new_riv_sensor_disp_km, y_new_may_sensor_levels, label='May')
plt.plot(x_new_riv_sensor_disp_km, y_new_jun_sensor_levels, 'x', label='Jun')
plt.plot(x_new_riv_sensor_disp_km, y_new_jul_sensor_levels, label='Jul')
plt.plot(x_new_riv_sensor_disp_km, y_new_aug_sensor_levels, label='Aug')
plt.plot(x_new_riv_sensor_disp_km, y_new_sep_sensor_levels, label='Sep')
plt.plot(x_new_riv_sensor_disp_km, y_new_oct_sensor_levels, label='Oct')
plt.plot(x_new_riv_sensor_disp_km, y_new_nov_sensor_levels, label='Nov')
plt.plot(x_new_riv_sensor_disp_km, y_new_dec_sensor_levels, label='Dec')
plt.plot(x_new_riv_sensor_disp_km, y_new_annual_sensor_levels, 'o', label='Annual Mean')

# the quadratic equations:
print("y=(%.6f)x^2+(%.6f)x+(%.6f)"%(z_jan_sensor_levels[0],z_jan_sensor_levels[1],z_jan_sensor_levels[2]))
print("y=(%.6f)x^2+(%.6f)x+(%.6f)"%(z_feb_sensor_levels[0],z_feb_sensor_levels[1],z_feb_sensor_levels[2]))
print("y=(%.6f)x^2+(%.6f)x+(%.6f)"%(z_mar_sensor_levels[0],z_mar_sensor_levels[1],z_mar_sensor_levels[2]))
print("y=(%.6f)x^2+(%.6f)x+(%.6f)"%(z_apr_sensor_levels[0],z_apr_sensor_levels[1],z_apr_sensor_levels[2]))
print("y=(%.6f)x^2+(%.6f)x+(%.6f)"%(z_may_sensor_levels[0],z_may_sensor_levels[1],z_may_sensor_levels[2]))
print("y=(%.6f)x^2+(%.6f)x+(%.6f)"%(z_jun_sensor_levels[0],z_jun_sensor_levels[1],z_jun_sensor_levels[2]))
print("y=(%.6f)x^2+(%.6f)x+(%.6f)"%(z_jul_sensor_levels[0],z_jul_sensor_levels[1],z_jul_sensor_levels[2]))
print("y=(%.6f)x^2+(%.6f)x+(%.6f)"%(z_aug_sensor_levels[0],z_aug_sensor_levels[1],z_aug_sensor_levels[2]))
print("y=(%.6f)x^2+(%.6f)x+(%.6f)"%(z_sep_sensor_levels[0],z_sep_sensor_levels[1],z_sep_sensor_levels[2]))
print("y=(%.6f)x^2+(%.6f)x+(%.6f)"%(z_oct_sensor_levels[0],z_oct_sensor_levels[1],z_oct_sensor_levels[2]))
print("y=(%.6f)x^2+(%.6f)x+(%.6f)"%(z_nov_sensor_levels[0],z_nov_sensor_levels[1],z_nov_sensor_levels[2]))
print("y=(%.6f)x^2+(%.6f)x+(%.6f)"%(z_dec_sensor_levels[0],z_dec_sensor_levels[1],z_dec_sensor_levels[2]))

print("y=(%.6f)x^2+(%.6f)x+(%.6f)"%(z_annual_sensor_levels[0],z_annual_sensor_levels[1],z_annual_sensor_levels[2]))

plt.xlim([riv_sensor_disp_km[0], riv_sensor_disp_km[-1]])
plt.xlabel('Distance from CNR U/S to High Toss (km)', fontsize=18)
plt.ylabel('Elevation (m)', fontsize=16)
plt.legend()
plt.show()

# This is from Cheq Model, calculated with river-path distances instead of linear distances.
# displacements from the CNR U/S sensor are in kilometers (polynomials determined from CTD sensor analysis)
# jan_HR_levels=(0.049826)*(disp_CNRUS_sensor**2)+(0.011812)*disp_CNRUS_sensor+(-0.433711)
# feb_HR_levels=(0.036149)*(disp_CNRUS_sensor**2)+(-0.002964)*disp_CNRUS_sensor+(-0.351285)
# mar_HR_levels=(0.022472)*(disp_CNRUS_sensor**2)+(-0.017741)*disp_CNRUS_sensor+(-0.268859)
# apr_HR_levels=(0.038385)*(disp_CNRUS_sensor**2)+(-0.026175)*disp_CNRUS_sensor+(-0.237086)
# may_HR_levels=(-0.024362)*(disp_CNRUS_sensor**2)+(0.070708)*disp_CNRUS_sensor+(-0.237799)
# jun_HR_levels=(-0.006396)*(disp_CNRUS_sensor**2)+(0.042014)*disp_CNRUS_sensor+(-0.224866)
# jul_HR_levels=(0.030679)*(disp_CNRUS_sensor**2)+(-0.024529)*disp_CNRUS_sensor+(-0.276057)
# aug_HR_levels=(0.086084)*(disp_CNRUS_sensor**2)+(-0.136663)*disp_CNRUS_sensor+(-0.292101)
# sep_HR_levels=(0.053613)*(disp_CNRUS_sensor**2)+(-0.063839)*disp_CNRUS_sensor+(-0.285530)
# oct_HR_levels=(0.030415)*(disp_CNRUS_sensor**2)+(0.003245)*disp_CNRUS_sensor+(-0.307802)
# nov_HR_levels=(0.082474)*(disp_CNRUS_sensor**2)+(-0.119651)*disp_CNRUS_sensor+(-0.275408)
# dec_HR_levels=(0.001616)*(disp_CNRUS_sensor**2)+(0.058003)*disp_CNRUS_sensor+(-0.365406)

# idx_oceanside = np.isfinite(x_oceanside_datenum) & np.isfinite(y_oceanside)
# z_oceanside = np.polyfit(x_oceanside_datenum[idx_oceanside], y_oceanside[idx_oceanside], 1)
# p_oceanside = np.poly1d(z_oceanside)

# polyX_oceanside = np.linspace(x_oceanside_datenum.min(), x_oceanside_datenum.max(), 100)
# pylab.plot(polyX_oceanside,p_oceanside(polyX_oceanside),"c", label='Mean Sea Level')
# # the line equation:
# print("y=%.6fx+(%.6f)"%(z_oceanside[0],z_oceanside[1]))

#%% Provincetown Tide Gauge (daily means only)

data_dir = os.path.join('E:\Data')
with open(os.path.join(data_dir,"Water Level Data","USGS 420259070105600 Provincetown Tide Gauge","Gage height_ft (mean).txt")) as f:
    reader = csv.reader(f, delimiter="\t")
    Provincetown_all_levels = list(reader)

Provincetown_daily_levels = []
for line in range(len(Provincetown_all_levels)-30):
    Provincetown_daily_levels.append([Provincetown_all_levels[line+30][2],Provincetown_all_levels[line+30][3]])

Provincetown_daily_levels = np.array(Provincetown_daily_levels)

"""
Provincetown Means
"""
# date2num returns Number of days (fraction part represents hours, minutes, seconds, ms) since 0001-01-01 00:00:00 UTC, plus one.
x_Provincetown, y_Provincetown = Provincetown_daily_levels.T

dates_Provincetown = [dateutil.parser.parse(x) for x in x_Provincetown]
x_Provincetown_datenum = mdates.date2num(dates_Provincetown)
y_Provincetown[np.where(y_Provincetown == '')] = np.nan
y_Provincetown = y_Provincetown.astype(np.float)*0.3048 # feet to meters

pylab.plot(x_Provincetown_datenum, y_Provincetown, 'o', markersize=1, label='Provincetown Daily Means')
idx_Provincetown = np.isfinite(x_Provincetown_datenum) & np.isfinite(y_Provincetown)
z_Provincetown = np.polyfit(x_Provincetown_datenum[idx_Provincetown], y_Provincetown[idx_Provincetown], 1)
p_Provincetown = np.poly1d(z_Provincetown)

polyX_Provincetown = np.linspace(x_Provincetown_datenum.min(), x_Provincetown_datenum.max(), 100)
pylab.plot(polyX_Provincetown,p_Provincetown(polyX_Provincetown),"c", label='Mean Sea Level, Provincetown')
# the line equation:
print("y=%.6fx+(%.6f)"%(z_Provincetown[0],z_Provincetown[1]))

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=18)
plt.ylabel('Elevation (m)', fontsize=16)
plt.legend()

pylab.show()

# Need to remove nan vals and reduce measurement frequency
nan_indices_Provincetown = []
for i in range(len(y_Provincetown)):
    if np.isnan(y_Provincetown[i]):
        nan_indices_Provincetown.append(i)

y_Provincetown_nonans = y_Provincetown.tolist()
x_Provincetown_datenum_nonans = x_Provincetown_datenum.tolist()
for index in sorted(nan_indices_Provincetown, reverse=True):
    del y_Provincetown_nonans[index]
    del x_Provincetown_datenum_nonans[index]
y_Provincetown_nonans = np.array(y_Provincetown_nonans)
x_Provincetown_datenum_nonans = np.array(x_Provincetown_datenum_nonans) 

# convert numbered datetime back to standard (allows determination of minutes)
dates_Provincetown_nonans = mdates.num2date(x_Provincetown_datenum_nonans)

# Provincetown Monthly Bins
January_Provincetown = []
February_Provincetown = []
March_Provincetown = []
April_Provincetown = []
May_Provincetown = []
June_Provincetown = []
July_Provincetown = []
August_Provincetown = []
September_Provincetown = []
October_Provincetown = []
November_Provincetown = []
December_Provincetown = []
for i in range(len(dates_Provincetown_nonans)):
    if dates_Provincetown_nonans[i].month == 1: 
        January_Provincetown.append(y_Provincetown_nonans[i])
    if dates_Provincetown_nonans[i].month == 2: 
        February_Provincetown.append(y_Provincetown_nonans[i])
    if dates_Provincetown_nonans[i].month == 3: 
        March_Provincetown.append(y_Provincetown_nonans[i])
    if dates_Provincetown_nonans[i].month == 4: 
        April_Provincetown.append(y_Provincetown_nonans[i])
    if dates_Provincetown_nonans[i].month == 5: 
        May_Provincetown.append(y_Provincetown_nonans[i])
    if dates_Provincetown_nonans[i].month == 6:
        June_Provincetown.append(y_Provincetown_nonans[i])
    if dates_Provincetown_nonans[i].month == 7:
        July_Provincetown.append(y_Provincetown_nonans[i])
    if dates_Provincetown_nonans[i].month == 8: 
        August_Provincetown.append(y_Provincetown_nonans[i])
    if dates_Provincetown_nonans[i].month == 9: 
        September_Provincetown.append(y_Provincetown_nonans[i])
    if dates_Provincetown_nonans[i].month == 10: 
        October_Provincetown.append(y_Provincetown_nonans[i])
    if dates_Provincetown_nonans[i].month == 11: 
        November_Provincetown.append(y_Provincetown_nonans[i])
    if dates_Provincetown_nonans[i].month == 12: 
        December_Provincetown.append(y_Provincetown_nonans[i])
January_Provincetown = np.array(January_Provincetown)
February_Provincetown = np.array(February_Provincetown)
March_Provincetown = np.array(March_Provincetown)
April_Provincetown = np.array(April_Provincetown)
May_Provincetown = np.array(May_Provincetown)
June_Provincetown = np.array(June_Provincetown)
July_Provincetown = np.array(July_Provincetown)
August_Provincetown = np.array(August_Provincetown)
September_Provincetown = np.array(September_Provincetown)
October_Provincetown = np.array(October_Provincetown)
November_Provincetown = np.array(November_Provincetown)
December_Provincetown = np.array(December_Provincetown)

January_Provincetown_mean = January_Provincetown.mean()
February_Provincetown_mean = February_Provincetown.mean()
March_Provincetown_mean = March_Provincetown.mean()
April_Provincetown_mean = April_Provincetown.mean()
May_Provincetown_mean = May_Provincetown.mean()
June_Provincetown_mean = June_Provincetown.mean()
July_Provincetown_mean = July_Provincetown.mean()
August_Provincetown_mean = August_Provincetown.mean()
September_Provincetown_mean = September_Provincetown.mean()
October_Provincetown_mean = October_Provincetown.mean()
November_Provincetown_mean = November_Provincetown.mean()
December_Provincetown_mean = December_Provincetown.mean()

y_Provincetown_monthly_means = np.array([January_Provincetown_mean,February_Provincetown_mean, March_Provincetown_mean, April_Provincetown_mean, May_Provincetown_mean, 
                          June_Provincetown_mean, July_Provincetown_mean, August_Provincetown_mean, September_Provincetown_mean, October_Provincetown_mean, 
                          November_Provincetown_mean, December_Provincetown_mean])
    
months_str = calendar.month_name
x_months = np.array(months_str[1:])

# Plots
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid(True, which='both')
ax1.plot(x_months, y_Provincetown_monthly_means, label='Provincetown Monthly Means, 2015-present')
ax1.legend(loc='best', bbox_to_anchor=(0.3, 0.1, 0.1, 0.3))
ax1.set_xlabel(r"Month")
ax1.set_ylabel(r"Mean Sea Level (m)")
ax1.set_xticklabels(x_months, rotation=55, horizontalalignment='right')
ax1.margins(0)
plt.tight_layout()
plt.show()

#%% Boston NOAA Tide Gauge

# Monthly MSL (mean sea level), 2015 - present, feet

with open(os.path.join(data_dir,"Water Level Data","NOAA 8443970 Boston MA","CO-OPS_Boston_8443970_wl_monthly.csv")) as f:
    reader = csv.reader(f, delimiter=",")
    Boston_monthly_all_levels = list(reader)
    
Boston_monthly_levels = []
for line in range(len(Boston_monthly_all_levels)-1):
    Boston_monthly_levels.append([Boston_monthly_all_levels[line+1][0] + " " + Boston_monthly_all_levels[line+1][1],Boston_monthly_all_levels[line+1][5]]) 
Boston_monthly_levels = np.array(Boston_monthly_levels)

x_Boston_monthly, y_Boston_monthly = Boston_monthly_levels.T

dates_Boston_monthly = [dateutil.parser.parse(x) for x in x_Boston_monthly]
x_Boston_monthly_datenum = mdates.date2num(dates_Boston_monthly)
y_Boston_monthly[np.where(y_Boston_monthly == '')] = np.nan
y_Boston_monthly = y_Boston_monthly.astype(np.float)*0.3048 # feet to meters

January_Boston = []
February_Boston = []
March_Boston = []
April_Boston = []
May_Boston = []
June_Boston = []
July_Boston = []
August_Boston = []
September_Boston = []
October_Boston = []
November_Boston = []
December_Boston = []
for i in range(len(dates_Boston_monthly)):
    if dates_Boston_monthly[i].month == 1: 
        January_Boston.append(y_Boston_monthly[i])
    if dates_Boston_monthly[i].month == 2: 
        February_Boston.append(y_Boston_monthly[i])
    if dates_Boston_monthly[i].month == 3: 
        March_Boston.append(y_Boston_monthly[i])
    if dates_Boston_monthly[i].month == 4: 
        April_Boston.append(y_Boston_monthly[i])
    if dates_Boston_monthly[i].month == 5: 
        May_Boston.append(y_Boston_monthly[i])
    if dates_Boston_monthly[i].month == 6:
        June_Boston.append(y_Boston_monthly[i])
    if dates_Boston_monthly[i].month == 7:
        July_Boston.append(y_Boston_monthly[i])
    if dates_Boston_monthly[i].month == 8: 
        August_Boston.append(y_Boston_monthly[i])
    if dates_Boston_monthly[i].month == 9: 
        September_Boston.append(y_Boston_monthly[i])
    if dates_Boston_monthly[i].month == 10: 
        October_Boston.append(y_Boston_monthly[i])
    if dates_Boston_monthly[i].month == 11: 
        November_Boston.append(y_Boston_monthly[i])
    if dates_Boston_monthly[i].month == 12: 
        December_Boston.append(y_Boston_monthly[i])
January_Boston = np.array(January_Boston)
February_Boston = np.array(February_Boston)
March_Boston = np.array(March_Boston)
April_Boston = np.array(April_Boston)
May_Boston = np.array(May_Boston)
June_Boston = np.array(June_Boston)
July_Boston = np.array(July_Boston)
August_Boston = np.array(August_Boston)
September_Boston = np.array(September_Boston)
October_Boston = np.array(October_Boston)
November_Boston = np.array(November_Boston)
December_Boston = np.array(December_Boston)

January_Boston_mean = January_Boston.mean()
February_Boston_mean = February_Boston.mean()
March_Boston_mean = March_Boston.mean()
April_Boston_mean = April_Boston.mean()
May_Boston_mean = May_Boston.mean()
June_Boston_mean = June_Boston.mean()
July_Boston_mean = July_Boston.mean()
August_Boston_mean = August_Boston.mean()
September_Boston_mean = September_Boston.mean()
October_Boston_mean = October_Boston.mean()
November_Boston_mean = November_Boston.mean()
December_Boston_mean = December_Boston.mean()

y_Boston_monthly_means = np.array([January_Boston_mean,February_Boston_mean, March_Boston_mean, April_Boston_mean, May_Boston_mean, 
                          June_Boston_mean, July_Boston_mean, August_Boston_mean, September_Boston_mean, October_Boston_mean, 
                          November_Boston_mean, December_Boston_mean])
    
# Plots
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid(True, which='both')
ax1.plot(x_months, y_Boston_monthly_means, label='Boston Monthly Means, 2015-present')
ax1.legend(loc='best', bbox_to_anchor=(0.3, 0.1, 0.1, 0.3))
ax1.set_xlabel(r"Month")
ax1.set_ylabel(r"Mean Sea Level (m)")
ax1.set_xticklabels(x_months, rotation=55, horizontalalignment='right')
ax1.margins(0)
plt.tight_layout()
plt.show()

# Hourly Verified Levels, 2015 - present, feet    

with open(os.path.join(data_dir,"Water Level Data","NOAA 8443970 Boston MA","CO-OPS_Boston_8443970_wl_hourly.csv")) as f:
    reader = csv.reader(f, delimiter=",")
    Boston_hourly_all_levels = list(reader)

Boston_hourly_levels = []
for line in range(len(Boston_hourly_all_levels)-1):
    Boston_hourly_levels.append([Boston_hourly_all_levels[line+1][0] + " " + Boston_hourly_all_levels[line+1][1],Boston_hourly_all_levels[line+1][4]]) 
Boston_hourly_levels = np.array(Boston_hourly_levels)

x_Boston_hourly, y_Boston_hourly = Boston_hourly_levels.T

dates_Boston_hourly = [dateutil.parser.parse(x) for x in x_Boston_hourly]
x_Boston_hourly_datenum = mdates.date2num(dates_Boston_hourly)
y_Boston_hourly[np.where(y_Boston_hourly == '-')] = np.nan
y_Boston_hourly = y_Boston_hourly.astype(np.float)*0.3048 # feet to meters

plt.figure()
pylab.plot(x_Boston_hourly_datenum, y_Boston_hourly, 'o', markersize=1)


#%% Chatham NOAA Tide Gauge, Monthly

# Monthly MSL (mean sea level), 2015 - present, feet

with open(os.path.join(data_dir,"Water Level Data","NOAA 8447435 Chatham_Lydia Cove MA","CO-OPS_Chatham_8447435_wl_monthly.csv")) as f:
    reader = csv.reader(f, delimiter=",")
    Chatham_monthly_all_levels = list(reader)
    
Chatham_monthly_levels = []
for line in range(len(Chatham_monthly_all_levels)-1):
    Chatham_monthly_levels.append([Chatham_monthly_all_levels[line+1][0] + " " + Chatham_monthly_all_levels[line+1][1],Chatham_monthly_all_levels[line+1][5]]) 
Chatham_monthly_levels = np.array(Chatham_monthly_levels)

x_Chatham_monthly, y_Chatham_monthly = Chatham_monthly_levels.T

dates_Chatham_monthly = [dateutil.parser.parse(x) for x in x_Chatham_monthly]
x_Chatham_monthly_datenum = mdates.date2num(dates_Chatham_monthly)
y_Chatham_monthly[np.where(y_Chatham_monthly == '')] = np.nan
y_Chatham_monthly = y_Chatham_monthly.astype(np.float)*0.3048 # feet to meters

January_Chatham = []
February_Chatham = []
March_Chatham = []
April_Chatham = []
May_Chatham = []
June_Chatham = []
July_Chatham = []
August_Chatham = []
September_Chatham = []
October_Chatham = []
November_Chatham = []
December_Chatham = []
for i in range(len(dates_Chatham_monthly)):
    if dates_Chatham_monthly[i].month == 1: 
        January_Chatham.append(y_Chatham_monthly[i])
    if dates_Chatham_monthly[i].month == 2: 
        February_Chatham.append(y_Chatham_monthly[i])
    if dates_Chatham_monthly[i].month == 3: 
        March_Chatham.append(y_Chatham_monthly[i])
    if dates_Chatham_monthly[i].month == 4: 
        April_Chatham.append(y_Chatham_monthly[i])
    if dates_Chatham_monthly[i].month == 5: 
        May_Chatham.append(y_Chatham_monthly[i])
    if dates_Chatham_monthly[i].month == 6:
        June_Chatham.append(y_Chatham_monthly[i])
    if dates_Chatham_monthly[i].month == 7:
        July_Chatham.append(y_Chatham_monthly[i])
    if dates_Chatham_monthly[i].month == 8: 
        August_Chatham.append(y_Chatham_monthly[i])
    if dates_Chatham_monthly[i].month == 9: 
        September_Chatham.append(y_Chatham_monthly[i])
    if dates_Chatham_monthly[i].month == 10: 
        October_Chatham.append(y_Chatham_monthly[i])
    if dates_Chatham_monthly[i].month == 11: 
        November_Chatham.append(y_Chatham_monthly[i])
    if dates_Chatham_monthly[i].month == 12: 
        December_Chatham.append(y_Chatham_monthly[i])
January_Chatham = np.array(January_Chatham)
February_Chatham = np.array(February_Chatham)
March_Chatham = np.array(March_Chatham)
April_Chatham = np.array(April_Chatham)
May_Chatham = np.array(May_Chatham)
June_Chatham = np.array(June_Chatham)
July_Chatham = np.array(July_Chatham)
August_Chatham = np.array(August_Chatham)
September_Chatham = np.array(September_Chatham)
October_Chatham = np.array(October_Chatham)
November_Chatham = np.array(November_Chatham)
December_Chatham = np.array(December_Chatham)

January_Chatham_mean = January_Chatham.mean()
February_Chatham_mean = February_Chatham.mean()
March_Chatham_mean = March_Chatham.mean()
April_Chatham_mean = April_Chatham.mean()
May_Chatham_mean = May_Chatham.mean()
June_Chatham_mean = June_Chatham.mean()
July_Chatham_mean = July_Chatham.mean()
August_Chatham_mean = August_Chatham.mean()
September_Chatham_mean = September_Chatham.mean()
October_Chatham_mean = October_Chatham.mean()
November_Chatham_mean = November_Chatham.mean()
December_Chatham_mean = December_Chatham.mean()

y_Chatham_monthly_means = np.array([January_Chatham_mean,February_Chatham_mean, March_Chatham_mean, April_Chatham_mean, May_Chatham_mean, 
                          June_Chatham_mean, July_Chatham_mean, August_Chatham_mean, September_Chatham_mean, October_Chatham_mean, 
                          November_Chatham_mean, December_Chatham_mean])
    
# Plots
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid(True, which='both')
ax1.plot(x_months, y_Chatham_monthly_means, label='Chatham Monthly Means, 2015-present')
ax1.legend(loc='best', bbox_to_anchor=(0.3, 0.1, 0.1, 0.3))
ax1.set_xlabel(r"Month")
ax1.set_ylabel(r"Mean Sea Level (m)")
ax1.set_xticklabels(x_months, rotation=55, horizontalalignment='right')
ax1.margins(0)
plt.tight_layout()
plt.show()



#%% Low Levels & Dike Invert at end of century

# Convert January 1, 2100, 00:00:00 to number
end_of_century = mdates.date2num(datetime(2100, 1, 1, 0, 0))
# 10 year increments
year_2020 = mdates.date2num(datetime(2020, 1, 1, 0, 0))
year_2030 = mdates.date2num(datetime(2030, 1, 1, 0, 0))
year_2040 = mdates.date2num(datetime(2040, 1, 1, 0, 0))
year_2050 = mdates.date2num(datetime(2050, 1, 1, 0, 0))
year_2060 = mdates.date2num(datetime(2060, 1, 1, 0, 0))
year_2070 = mdates.date2num(datetime(2070, 1, 1, 0, 0))
year_2080 = mdates.date2num(datetime(2080, 1, 1, 0, 0))
year_2090 = mdates.date2num(datetime(2090, 1, 1, 0, 0))

# Mean and standard deviation of mins
y_oceanside_meanmins = np.nanmean(y_oceanside_mins) # meters
y_HRside_meanmins = np.nanmean(y_HRside_mins) # meters
y_oceanside_stdmins = np.nanstd(y_oceanside_mins) # meters
y_HRside_stdmins = np.nanstd(y_HRside_mins) # meters
x_oceanside_meandatetime = np.nanmean(x_oceanside_datenum_hourly)
x_HRside_meandatetime = np.nanmean(x_HRside_rangedates)

# Mean and standard deviation of HR maxes
y_HRside_meanmaxes = np.nanmean(y_HRside_maxes) # meters
y_HRside_stdmaxes = np.nanstd(y_HRside_maxes) # meters
    
# Min Plots w/ Trendlines
plt.figure()
pylab.plot(x_oceanside_rangedates, y_oceanside_mins, 'o', markersize=1)
pylab.plot(x_HRside_rangedates, y_HRside_mins, 'o', markersize=1)

idx_oceanside_mins = np.isfinite(x_oceanside_rangedates) & np.isfinite(y_oceanside_mins)
z_oceanside_mins = np.polyfit(x_oceanside_rangedates[idx_oceanside_mins], y_oceanside_mins[idx_oceanside_mins], 1)
p_oceanside_mins = np.poly1d(z_oceanside_mins)
idx_HRside_mins = np.isfinite(x_HRside_rangedates) & np.isfinite(y_HRside_mins)
z_HRside_mins = np.polyfit(x_HRside_rangedates[idx_HRside_mins], y_HRside_mins[idx_HRside_mins], 1)
p_HRside_mins = np.poly1d(z_HRside_mins)

polyX_oceanside_mins = np.linspace(x_oceanside_rangedates.min(), end_of_century, 100)
pylab.plot(polyX_oceanside_mins,p_oceanside_mins(polyX_oceanside_mins),"c", label='Minimum Sea Level, 2018 Trend')
polyX_HRside_mins = np.linspace(x_HRside_rangedates.min(), end_of_century, 100)
pylab.plot(polyX_HRside_mins,p_HRside_mins(polyX_HRside_mins),"green", label='Minimum River Level, 2016-2018 Trend')
plt.axhline(y=-0.6, color='g', linestyle='-')
# the line equations:
print("y=%.6fx+(%.6f)"%(z_oceanside_mins[0],z_oceanside_mins[1]))
print("y=%.6fx+(%.6f)"%(z_HRside_mins[0],z_HRside_mins[1]))

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=18)
plt.ylabel('Elevation (m)', fontsize=16)
plt.legend()

pylab.show()

# Min Plots w/ Projected Sea Level Rise from USGS RadForc inputs SL Sandwich 1 and 1.5 m GMSL

# date arrays
x_oceanside_dates_to2100 = np.array([x_oceanside_meandatetime, year_2020, year_2030, year_2040, year_2050, year_2060, year_2070, 
                                     year_2080, year_2090, end_of_century])
x_HRside_dates_to2100 = np.array([x_HRside_meandatetime, year_2020, year_2030, year_2040, year_2050, year_2060, year_2070, 
                                     year_2080, year_2090, end_of_century])
    

y_oceanside_meanmins_to2100_1mGMSL = np.array([y_oceanside_meanmins, -1.01, -0.91, -0.81, -0.68, -0.54, -0.39, -0.24, -0.07, 0.10])    
y_oceanside_meanmins_to2100_1mGMSL_plusstd = y_oceanside_meanmins_to2100_1mGMSL + 0.5*y_oceanside_stdmins
y_oceanside_meanmins_to2100_1mGMSL_minusstd = y_oceanside_meanmins_to2100_1mGMSL - 0.5*y_oceanside_stdmins
    
y_oceanside_meanmins_to2100_1_5mGMSL = np.array([y_oceanside_meanmins, -0.99, -0.86, -0.70, -0.52, -0.32, -0.09, 0.17, 0.44, 0.74])
y_oceanside_meanmins_to2100_1_5mGMSL_plusstd = y_oceanside_meanmins_to2100_1_5mGMSL + 0.5*y_oceanside_stdmins
y_oceanside_meanmins_to2100_1_5mGMSL_minusstd = y_oceanside_meanmins_to2100_1_5mGMSL - 0.5*y_oceanside_stdmins

y_HRside_meanmins_to2100_1mGMSL = np.array([y_HRside_meanmins, -0.64, -0.54, -0.44, -0.31, -0.17, -0.02, 0.13, 0.30, 0.47])
y_HRside_meanmins_to2100_1mGMSL_plusstd = y_HRside_meanmins_to2100_1mGMSL + 0.5*y_HRside_stdmins
y_HRside_meanmins_to2100_1mGMSL_minusstd = y_HRside_meanmins_to2100_1mGMSL - 0.5*y_HRside_stdmins

y_HRside_meanmins_to2100_1_5mGMSL = np.array([y_HRside_meanmins, -0.62, -0.49, -0.33, -0.15, 0.05, 0.28, 0.54, 0.81, 1.11])
y_HRside_meanmins_to2100_1_5mGMSL_plusstd = y_HRside_meanmins_to2100_1_5mGMSL + 0.5*y_HRside_stdmins
y_HRside_meanmins_to2100_1_5mGMSL_minusstd = y_HRside_meanmins_to2100_1_5mGMSL - 0.5*y_HRside_stdmins
   
# Min Plots w/ Standard Deviations
# Vertical Lines from https://vdatum.noaa.gov/vdatumweb/vdatumweb?a=165105320190814 and using construction docs.
plt.figure()
plt.grid(True, which='both')
plt.axhline(y=-0.62, color='m', linestyle='-', label='Approximate Present Height of Sluice Gate')
plt.axhline(y=-0.59, color='m', linestyle='-')
plt.axhline(y=0.46, color='black', linestyle='-', label='Maximum Height of Sluice Gate to Culvert Top')
plt.axhline(y=0.38, color='black', linestyle='-')
plt.axhline(y=-0.91, color='red', linestyle='-', label='Dike Invert Elevation, HR (high) to WF Harbor (low)')
plt.axhline(y=-1.06, color='red', linestyle='-')
plt.axhline(y=-1.37, color='red', linestyle='-')

pylab.plot(x_oceanside_dates_to2100, y_oceanside_meanmins_to2100_1mGMSL, label='Low Tide Sea Level & 1 sigma stdev, 1m GMSL')
plt.fill_between(x_oceanside_dates_to2100, y_oceanside_meanmins_to2100_1mGMSL_plusstd, y_oceanside_meanmins_to2100_1mGMSL_minusstd, facecolor='blue', alpha=0.5)
pylab.plot(x_HRside_dates_to2100, y_HRside_meanmins_to2100_1mGMSL, label='Low Tide River Level & 1 sigma stdev, 1m GMSL')
plt.fill_between(x_HRside_dates_to2100, y_HRside_meanmins_to2100_1mGMSL_plusstd, y_HRside_meanmins_to2100_1mGMSL_minusstd, facecolor='orange', alpha=0.5)

pylab.plot(x_oceanside_dates_to2100, y_oceanside_meanmins_to2100_1_5mGMSL, label='Low Tide Sea Level & 1 sigma stdev, 1.5m GMSL')
plt.fill_between(x_oceanside_dates_to2100, y_oceanside_meanmins_to2100_1_5mGMSL_plusstd, y_oceanside_meanmins_to2100_1_5mGMSL_minusstd, facecolor='green', alpha=0.5)
pylab.plot(x_HRside_dates_to2100, y_HRside_meanmins_to2100_1_5mGMSL, label='Low Tide River Level & 1 sigma stdev, 1.5m GMSL')
plt.fill_between(x_HRside_dates_to2100, y_HRside_meanmins_to2100_1_5mGMSL_plusstd, y_HRside_meanmins_to2100_1_5mGMSL_minusstd, facecolor='red', alpha=0.5)

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=18)
plt.ylabel('Elevation (m)', fontsize=16)
plt.legend()

pylab.show()
    
#%% Well Data Comparisons, with Ranges (maxval-minval for each 12hr 25min interval)

raw_datafname = os.path.join(data_dir,'Well Data', 'Shallow', 'HerringRiver_WaterLevel_2015_2016_2017_2018.mat')
rawdict = loadmat(raw_datafname)
treat_names = [i[0][0] for i in rawdict['HR']['treatment'][0][0]]
wl_navd88 = rawdict['HR']['WL_NAVD88'][0][0].ravel()
# September 2015 to December 2018
date_time = rawdict['HR']['date_time'][0][0]
dbs = [i[0] for i in rawdict['HR']['dbs'][0][0].astype(np.float)]
data_df = pd.DataFrame(list(zip(treat_names,wl_navd88,date_time,dbs)),columns=['treatment','wl_navd88','datetime','dbs'])
data_df['datetime'] = pd.to_datetime(data_df['datetime'])
data_df = data_df[data_df['datetime']>datetime(2017,4,1)].copy()

datagrp_df = data_df.groupby('treatment')

for t1, df in datagrp_df:
    if t1 == "Dry Forest":
        x_Dry_Forest = np.array(df['datetime'])
        y_Dry_Forest = np.array(df['wl_navd88'])
    if t1 == "Phragmites Wetland":
        x_Phrag_Wetland = np.array(df['datetime'])
        y_Phrag_Wetland = np.array(df['wl_navd88'])
    if t1 == "Typha Wetland":
        x_Typha_Wetland = np.array(df['datetime'])
        y_Typha_Wetland = np.array(df['wl_navd88'])
    if t1 == "Wet Shrub":
        x_Wet_Shrub = np.array(df['datetime'])
        y_Wet_Shrub = np.array(df['wl_navd88'])
x_Dry_Forest_datenum = mdates.date2num(x_Dry_Forest)
x_Phrag_Wetland_datenum = mdates.date2num(x_Phrag_Wetland)
x_Typha_Wetland_datenum = mdates.date2num(x_Typha_Wetland)
x_Wet_Shrub_datenum = mdates.date2num(x_Wet_Shrub)

# Dry_Forest Well (Plot w/ High Toss, Wet Shrub, and Dog Leg)
bin_start = 0
x_Dry_Forest_rangedates = []    
y_Dry_Forest_mins = []
y_Dry_Forest_maxes = []
for bin_index in range(len(x_Dry_Forest_datenum)):
    datestart = x_Dry_Forest_datenum[bin_start]
    dateend = datestart + (x_Dry_Forest_datenum[bin_index] - x_Dry_Forest_datenum[bin_start])
    date_interval = dateend - datestart
    bin_end = bin_index
    if (date_interval >= tidal_peaktopeak_interval):
            x_Dry_Forest_rangedates.append(x_Dry_Forest_datenum[int((bin_start+bin_end)/2)])
            y_Dry_Forest_mins.append(np.nanmin(y_Dry_Forest[bin_start:bin_end]))
            y_Dry_Forest_maxes.append(np.nanmax(y_Dry_Forest[bin_start:bin_end]))
            bin_start = bin_end
x_Dry_Forest_rangedates = np.array(x_Dry_Forest_rangedates)
y_Dry_Forest_mins = np.array(y_Dry_Forest_mins)
y_Dry_Forest_maxes = np.array(y_Dry_Forest_maxes)
# y_Dry_Forest_mins[y_Dry_Forest_mins > np.nanmean(y_Dry_Forest_maxes)] = np.nan
# y_Dry_Forest_maxes[y_Dry_Forest_maxes < np.nanmean(y_Dry_Forest_mins)] = np.nan
y_Dry_Forest_ranges = y_Dry_Forest_maxes - y_Dry_Forest_mins

# Phragmites Wetland Well (Plot with CNR U/S and Dog Leg)
bin_start = 0
x_Phrag_Wetland_rangedates = []    
y_Phrag_Wetland_mins = []
y_Phrag_Wetland_maxes = []
for bin_index in range(len(x_Phrag_Wetland_datenum)):
    datestart = x_Phrag_Wetland_datenum[bin_start]
    dateend = datestart + (x_Phrag_Wetland_datenum[bin_index] - x_Phrag_Wetland_datenum[bin_start])
    date_interval = dateend - datestart
    bin_end = bin_index
    if (date_interval >= tidal_peaktopeak_interval):
            x_Phrag_Wetland_rangedates.append(x_Phrag_Wetland_datenum[int((bin_start+bin_end)/2)])
            y_Phrag_Wetland_mins.append(np.nanmin(y_Phrag_Wetland[bin_start:bin_end]))
            y_Phrag_Wetland_maxes.append(np.nanmax(y_Phrag_Wetland[bin_start:bin_end]))
            bin_start = bin_end
x_Phrag_Wetland_rangedates = np.array(x_Phrag_Wetland_rangedates)
y_Phrag_Wetland_mins = np.array(y_Phrag_Wetland_mins)
y_Phrag_Wetland_maxes = np.array(y_Phrag_Wetland_maxes)
# y_Phrag_Wetland_mins[y_Phrag_Wetland_mins > np.nanmean(y_Phrag_Wetland_maxes)] = np.nan
# y_Phrag_Wetland_maxes[y_Phrag_Wetland_maxes < np.nanmean(y_Phrag_Wetland_mins)] = np.nan
y_Phrag_Wetland_ranges = y_Phrag_Wetland_maxes - y_Phrag_Wetland_mins

# Wet Shrub Well
bin_start = 0
x_Wet_Shrub_rangedates = []    
y_Wet_Shrub_mins = []
y_Wet_Shrub_maxes = []
for bin_index in range(len(x_Wet_Shrub_datenum)):
    datestart = x_Wet_Shrub_datenum[bin_start]
    dateend = datestart + (x_Wet_Shrub_datenum[bin_index] - x_Wet_Shrub_datenum[bin_start])
    date_interval = dateend - datestart
    bin_end = bin_index
    if (date_interval >= tidal_peaktopeak_interval):
            x_Wet_Shrub_rangedates.append(x_Wet_Shrub_datenum[int((bin_start+bin_end)/2)])
            y_Wet_Shrub_mins.append(np.nanmin(y_Wet_Shrub[bin_start:bin_end]))
            y_Wet_Shrub_maxes.append(np.nanmax(y_Wet_Shrub[bin_start:bin_end]))
            bin_start = bin_end
x_Wet_Shrub_rangedates = np.array(x_Wet_Shrub_rangedates)
y_Wet_Shrub_mins = np.array(y_Wet_Shrub_mins)
y_Wet_Shrub_maxes = np.array(y_Wet_Shrub_maxes)
# y_Wet_Shrub_mins[y_Wet_Shrub_mins > np.nanmean(y_Wet_Shrub_maxes)] = np.nan
# y_Wet_Shrub_maxes[y_Wet_Shrub_maxes < np.nanmean(y_Wet_Shrub_mins)] = np.nan
y_Wet_Shrub_ranges = y_Wet_Shrub_maxes - y_Wet_Shrub_mins

"""
Monthly Bins for Well Data: Phrag Wetland
"""
# Need to remove nan vals and reduce measurement frequency
nan_indices_Phrag_Wetland = []
for i in range(len(y_Phrag_Wetland)):
    if np.isnan(y_Phrag_Wetland[i]):
        nan_indices_Phrag_Wetland.append(i)

y_Phrag_Wetland_nonans = y_Phrag_Wetland.tolist()
x_Phrag_Wetland_datenum_nonans = x_Phrag_Wetland_datenum.tolist()
for index in sorted(nan_indices_Phrag_Wetland, reverse=True):
    del y_Phrag_Wetland_nonans[index]
    del x_Phrag_Wetland_datenum_nonans[index]
y_Phrag_Wetland_nonans = np.array(y_Phrag_Wetland_nonans)
x_Phrag_Wetland_datenum_nonans = np.array(x_Phrag_Wetland_datenum_nonans) 

# convert numbered datetime back to standard (allows determination of minutes)
dates_Phrag_Wetland_nonans = mdates.num2date(x_Phrag_Wetland_datenum_nonans)

# Phrag_Wetland Monthly Bins
January_Phrag_Wetland = []
February_Phrag_Wetland = []
March_Phrag_Wetland = []
April_Phrag_Wetland = []
May_Phrag_Wetland = []
June_Phrag_Wetland = []
July_Phrag_Wetland = []
August_Phrag_Wetland = []
September_Phrag_Wetland = []
October_Phrag_Wetland = []
November_Phrag_Wetland = []
December_Phrag_Wetland = []
for i in range(len(dates_Phrag_Wetland_nonans)):
    if dates_Phrag_Wetland_nonans[i].month == 1: 
        January_Phrag_Wetland.append(y_Phrag_Wetland_nonans[i])
    if dates_Phrag_Wetland_nonans[i].month == 2: 
        February_Phrag_Wetland.append(y_Phrag_Wetland_nonans[i])
    if dates_Phrag_Wetland_nonans[i].month == 3: 
        March_Phrag_Wetland.append(y_Phrag_Wetland_nonans[i])
    if dates_Phrag_Wetland_nonans[i].month == 4: 
        April_Phrag_Wetland.append(y_Phrag_Wetland_nonans[i])
    if dates_Phrag_Wetland_nonans[i].month == 5: 
        May_Phrag_Wetland.append(y_Phrag_Wetland_nonans[i])
    if dates_Phrag_Wetland_nonans[i].month == 6:
        June_Phrag_Wetland.append(y_Phrag_Wetland_nonans[i])
    if dates_Phrag_Wetland_nonans[i].month == 7:
        July_Phrag_Wetland.append(y_Phrag_Wetland_nonans[i])
    if dates_Phrag_Wetland_nonans[i].month == 8: 
        August_Phrag_Wetland.append(y_Phrag_Wetland_nonans[i])
    if dates_Phrag_Wetland_nonans[i].month == 9: 
        September_Phrag_Wetland.append(y_Phrag_Wetland_nonans[i])
    if dates_Phrag_Wetland_nonans[i].month == 10: 
        October_Phrag_Wetland.append(y_Phrag_Wetland_nonans[i])
    if dates_Phrag_Wetland_nonans[i].month == 11: 
        November_Phrag_Wetland.append(y_Phrag_Wetland_nonans[i])
    if dates_Phrag_Wetland_nonans[i].month == 12: 
        December_Phrag_Wetland.append(y_Phrag_Wetland_nonans[i])
January_Phrag_Wetland = np.array(January_Phrag_Wetland)
February_Phrag_Wetland = np.array(February_Phrag_Wetland)
March_Phrag_Wetland = np.array(March_Phrag_Wetland)
April_Phrag_Wetland = np.array(April_Phrag_Wetland)
May_Phrag_Wetland = np.array(May_Phrag_Wetland)
June_Phrag_Wetland = np.array(June_Phrag_Wetland)
July_Phrag_Wetland = np.array(July_Phrag_Wetland)
August_Phrag_Wetland = np.array(August_Phrag_Wetland)
September_Phrag_Wetland = np.array(September_Phrag_Wetland)
October_Phrag_Wetland = np.array(October_Phrag_Wetland)
November_Phrag_Wetland = np.array(November_Phrag_Wetland)
December_Phrag_Wetland = np.array(December_Phrag_Wetland)

January_Phrag_Wetland_mean = January_Phrag_Wetland.mean()
February_Phrag_Wetland_mean = February_Phrag_Wetland.mean()
March_Phrag_Wetland_mean = March_Phrag_Wetland.mean()
April_Phrag_Wetland_mean = April_Phrag_Wetland.mean()
May_Phrag_Wetland_mean = May_Phrag_Wetland.mean()
June_Phrag_Wetland_mean = June_Phrag_Wetland.mean()
July_Phrag_Wetland_mean = July_Phrag_Wetland.mean()
August_Phrag_Wetland_mean = August_Phrag_Wetland.mean()
September_Phrag_Wetland_mean = September_Phrag_Wetland.mean()
October_Phrag_Wetland_mean = October_Phrag_Wetland.mean()
November_Phrag_Wetland_mean = November_Phrag_Wetland.mean()
December_Phrag_Wetland_mean = December_Phrag_Wetland.mean()

y_Phrag_Wetland_monthly_means = np.array([January_Phrag_Wetland_mean,February_Phrag_Wetland_mean, March_Phrag_Wetland_mean, April_Phrag_Wetland_mean, May_Phrag_Wetland_mean, 
                          June_Phrag_Wetland_mean, July_Phrag_Wetland_mean, August_Phrag_Wetland_mean, September_Phrag_Wetland_mean, October_Phrag_Wetland_mean, 
                          November_Phrag_Wetland_mean, December_Phrag_Wetland_mean])

"""
Monthly Bins for Well Data: Wet Shrub
"""
# Need to remove nan vals and reduce measurement frequency
nan_indices_Wet_Shrub = []
for i in range(len(y_Wet_Shrub)):
    if np.isnan(y_Wet_Shrub[i]):
        nan_indices_Wet_Shrub.append(i)

y_Wet_Shrub_nonans = y_Wet_Shrub.tolist()
x_Wet_Shrub_datenum_nonans = x_Wet_Shrub_datenum.tolist()
for index in sorted(nan_indices_Wet_Shrub, reverse=True):
    del y_Wet_Shrub_nonans[index]
    del x_Wet_Shrub_datenum_nonans[index]
y_Wet_Shrub_nonans = np.array(y_Wet_Shrub_nonans)
x_Wet_Shrub_datenum_nonans = np.array(x_Wet_Shrub_datenum_nonans) 

# convert numbered datetime back to standard (allows determination of minutes)
dates_Wet_Shrub_nonans = mdates.num2date(x_Wet_Shrub_datenum_nonans)

# Wet_Shrub Monthly Bins
January_Wet_Shrub = []
February_Wet_Shrub = []
March_Wet_Shrub = []
April_Wet_Shrub = []
May_Wet_Shrub = []
June_Wet_Shrub = []
July_Wet_Shrub = []
August_Wet_Shrub = []
September_Wet_Shrub = []
October_Wet_Shrub = []
November_Wet_Shrub = []
December_Wet_Shrub = []
for i in range(len(dates_Wet_Shrub_nonans)):
    if dates_Wet_Shrub_nonans[i].month == 1: 
        January_Wet_Shrub.append(y_Wet_Shrub_nonans[i])
    if dates_Wet_Shrub_nonans[i].month == 2: 
        February_Wet_Shrub.append(y_Wet_Shrub_nonans[i])
    if dates_Wet_Shrub_nonans[i].month == 3: 
        March_Wet_Shrub.append(y_Wet_Shrub_nonans[i])
    if dates_Wet_Shrub_nonans[i].month == 4: 
        April_Wet_Shrub.append(y_Wet_Shrub_nonans[i])
    if dates_Wet_Shrub_nonans[i].month == 5: 
        May_Wet_Shrub.append(y_Wet_Shrub_nonans[i])
    if dates_Wet_Shrub_nonans[i].month == 6:
        June_Wet_Shrub.append(y_Wet_Shrub_nonans[i])
    if dates_Wet_Shrub_nonans[i].month == 7:
        July_Wet_Shrub.append(y_Wet_Shrub_nonans[i])
    if dates_Wet_Shrub_nonans[i].month == 8: 
        August_Wet_Shrub.append(y_Wet_Shrub_nonans[i])
    if dates_Wet_Shrub_nonans[i].month == 9: 
        September_Wet_Shrub.append(y_Wet_Shrub_nonans[i])
    if dates_Wet_Shrub_nonans[i].month == 10: 
        October_Wet_Shrub.append(y_Wet_Shrub_nonans[i])
    if dates_Wet_Shrub_nonans[i].month == 11: 
        November_Wet_Shrub.append(y_Wet_Shrub_nonans[i])
    if dates_Wet_Shrub_nonans[i].month == 12: 
        December_Wet_Shrub.append(y_Wet_Shrub_nonans[i])
January_Wet_Shrub = np.array(January_Wet_Shrub)
February_Wet_Shrub = np.array(February_Wet_Shrub)
March_Wet_Shrub = np.array(March_Wet_Shrub)
April_Wet_Shrub = np.array(April_Wet_Shrub)
May_Wet_Shrub = np.array(May_Wet_Shrub)
June_Wet_Shrub = np.array(June_Wet_Shrub)
July_Wet_Shrub = np.array(July_Wet_Shrub)
August_Wet_Shrub = np.array(August_Wet_Shrub)
September_Wet_Shrub = np.array(September_Wet_Shrub)
October_Wet_Shrub = np.array(October_Wet_Shrub)
November_Wet_Shrub = np.array(November_Wet_Shrub)
December_Wet_Shrub = np.array(December_Wet_Shrub)

January_Wet_Shrub_mean = January_Wet_Shrub.mean()
February_Wet_Shrub_mean = February_Wet_Shrub.mean()
March_Wet_Shrub_mean = March_Wet_Shrub.mean()
April_Wet_Shrub_mean = April_Wet_Shrub.mean()
May_Wet_Shrub_mean = May_Wet_Shrub.mean()
June_Wet_Shrub_mean = June_Wet_Shrub.mean()
July_Wet_Shrub_mean = July_Wet_Shrub.mean()
August_Wet_Shrub_mean = August_Wet_Shrub.mean()
September_Wet_Shrub_mean = September_Wet_Shrub.mean()
October_Wet_Shrub_mean = October_Wet_Shrub.mean()
November_Wet_Shrub_mean = November_Wet_Shrub.mean()
December_Wet_Shrub_mean = December_Wet_Shrub.mean()

y_Wet_Shrub_monthly_means = np.array([January_Wet_Shrub_mean,February_Wet_Shrub_mean, March_Wet_Shrub_mean, April_Wet_Shrub_mean, May_Wet_Shrub_mean, 
                          June_Wet_Shrub_mean, July_Wet_Shrub_mean, August_Wet_Shrub_mean, September_Wet_Shrub_mean, October_Wet_Shrub_mean, 
                          November_Wet_Shrub_mean, December_Wet_Shrub_mean])
    
"""
Monthly Bins for Well Data: Dry Forest
"""
# Need to remove nan vals and reduce measurement frequency
nan_indices_Dry_Forest = []
for i in range(len(y_Dry_Forest)):
    if np.isnan(y_Dry_Forest[i]):
        nan_indices_Dry_Forest.append(i)

y_Dry_Forest_nonans = y_Dry_Forest.tolist()
x_Dry_Forest_datenum_nonans = x_Dry_Forest_datenum.tolist()
for index in sorted(nan_indices_Dry_Forest, reverse=True):
    del y_Dry_Forest_nonans[index]
    del x_Dry_Forest_datenum_nonans[index]
y_Dry_Forest_nonans = np.array(y_Dry_Forest_nonans)
x_Dry_Forest_datenum_nonans = np.array(x_Dry_Forest_datenum_nonans) 

# convert numbered datetime back to standard (allows determination of minutes)
dates_Dry_Forest_nonans = mdates.num2date(x_Dry_Forest_datenum_nonans)

# Dry_Forest Monthly Bins
January_Dry_Forest = []
February_Dry_Forest = []
March_Dry_Forest = []
April_Dry_Forest = []
May_Dry_Forest = []
June_Dry_Forest = []
July_Dry_Forest = []
August_Dry_Forest = []
September_Dry_Forest = []
October_Dry_Forest = []
November_Dry_Forest = []
December_Dry_Forest = []
for i in range(len(dates_Dry_Forest_nonans)):
    if dates_Dry_Forest_nonans[i].month == 1: 
        January_Dry_Forest.append(y_Dry_Forest_nonans[i])
    if dates_Dry_Forest_nonans[i].month == 2: 
        February_Dry_Forest.append(y_Dry_Forest_nonans[i])
    if dates_Dry_Forest_nonans[i].month == 3: 
        March_Dry_Forest.append(y_Dry_Forest_nonans[i])
    if dates_Dry_Forest_nonans[i].month == 4: 
        April_Dry_Forest.append(y_Dry_Forest_nonans[i])
    if dates_Dry_Forest_nonans[i].month == 5: 
        May_Dry_Forest.append(y_Dry_Forest_nonans[i])
    if dates_Dry_Forest_nonans[i].month == 6:
        June_Dry_Forest.append(y_Dry_Forest_nonans[i])
    if dates_Dry_Forest_nonans[i].month == 7:
        July_Dry_Forest.append(y_Dry_Forest_nonans[i])
    if dates_Dry_Forest_nonans[i].month == 8: 
        August_Dry_Forest.append(y_Dry_Forest_nonans[i])
    if dates_Dry_Forest_nonans[i].month == 9: 
        September_Dry_Forest.append(y_Dry_Forest_nonans[i])
    if dates_Dry_Forest_nonans[i].month == 10: 
        October_Dry_Forest.append(y_Dry_Forest_nonans[i])
    if dates_Dry_Forest_nonans[i].month == 11: 
        November_Dry_Forest.append(y_Dry_Forest_nonans[i])
    if dates_Dry_Forest_nonans[i].month == 12: 
        December_Dry_Forest.append(y_Dry_Forest_nonans[i])
January_Dry_Forest = np.array(January_Dry_Forest)
February_Dry_Forest = np.array(February_Dry_Forest)
March_Dry_Forest = np.array(March_Dry_Forest)
April_Dry_Forest = np.array(April_Dry_Forest)
May_Dry_Forest = np.array(May_Dry_Forest)
June_Dry_Forest = np.array(June_Dry_Forest)
July_Dry_Forest = np.array(July_Dry_Forest)
August_Dry_Forest = np.array(August_Dry_Forest)
September_Dry_Forest = np.array(September_Dry_Forest)
October_Dry_Forest = np.array(October_Dry_Forest)
November_Dry_Forest = np.array(November_Dry_Forest)
December_Dry_Forest = np.array(December_Dry_Forest)

January_Dry_Forest_mean = January_Dry_Forest.mean()
February_Dry_Forest_mean = February_Dry_Forest.mean()
March_Dry_Forest_mean = March_Dry_Forest.mean()
April_Dry_Forest_mean = April_Dry_Forest.mean()
May_Dry_Forest_mean = May_Dry_Forest.mean()
June_Dry_Forest_mean = June_Dry_Forest.mean()
July_Dry_Forest_mean = July_Dry_Forest.mean()
August_Dry_Forest_mean = August_Dry_Forest.mean()
September_Dry_Forest_mean = September_Dry_Forest.mean()
October_Dry_Forest_mean = October_Dry_Forest.mean()
November_Dry_Forest_mean = November_Dry_Forest.mean()
December_Dry_Forest_mean = December_Dry_Forest.mean()

y_Dry_Forest_monthly_means = np.array([January_Dry_Forest_mean,February_Dry_Forest_mean, March_Dry_Forest_mean, April_Dry_Forest_mean, May_Dry_Forest_mean, 
                          June_Dry_Forest_mean, July_Dry_Forest_mean, August_Dry_Forest_mean, September_Dry_Forest_mean, October_Dry_Forest_mean, 
                          November_Dry_Forest_mean, December_Dry_Forest_mean])
    
# Plots
months_str = calendar.month_name
x_months = np.array(months_str[1:])
    
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid(True, which='both')
ax1.plot(x_months, y_Phrag_Wetland_monthly_means, label='Phrag_Wetland Monthly Means, 2017-present')
ax1.plot(x_months, y_Wet_Shrub_monthly_means, label='Wet_Shrub Monthly Means, 2017-present')
ax1.plot(x_months, y_Dry_Forest_monthly_means, label='Dry_Forest Monthly Means, 2017-present')
ax1.legend(loc='best', bbox_to_anchor=(0.3, 0.1, 0.1, 0.3))
ax1.set_xlabel(r"Month")
ax1.set_ylabel(r"Mean Well Level, NAVD88 (m)")
ax1.set_xticklabels(x_months, rotation=55, horizontalalignment='right')
ax1.margins(0)
plt.tight_layout()
plt.show()

#%% Plots, CNR U/S to High Toss (run previous cells)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.plot(x_CNRUS_datenum, y_CNRUS, label='CNR U/S Levels')
ax1.plot(x_DogLeg_datenum, y_DogLeg, label='Dog Leg Levels')
ax1.plot(x_Phrag_Wetland_datenum, y_Phrag_Wetland, label='Phragmites Wetland Levels')

idx_Phrag_Wetland = np.isfinite(x_Phrag_Wetland_datenum) & np.isfinite(y_Phrag_Wetland)
z_Phrag_Wetland = np.polyfit(x_Phrag_Wetland_datenum[idx_Phrag_Wetland], y_Phrag_Wetland[idx_Phrag_Wetland], 1)
p_Phrag_Wetland = np.poly1d(z_Phrag_Wetland)

ax1.plot(polyX_CNRUS,p_CNRUS(polyX_CNRUS),"blue", label='CNR U/S Level Trend')
polyX_Phrag_Wetland = np.linspace(x_Phrag_Wetland_datenum.min(), x_Phrag_Wetland_datenum.max(), 100)
ax1.plot(polyX_Phrag_Wetland,p_Phrag_Wetland(polyX_Phrag_Wetland),"green", label='Phrag Wetland Level Trend')
ax1.plot(polyX_DogLeg,p_DogLeg(polyX_DogLeg),"orange", label='Dog Leg Level Trend')
# the line equations:
print("y=%.6fx+(%.6f)"%(z_CNRUS[0],z_CNRUS[1]))
print("y=%.6fx+(%.6f)"%(z_Phrag_Wetland[0],z_Phrag_Wetland[1]))
print("y=%.6fx+(%.6f)"%(z_DogLeg[0],z_DogLeg[1]))

# # Max Plots, CNR U/S to Dog Leg - use all data for wells (not tidally influenced)
# ax2.plot(x_CNRUS_rangedates, y_CNRUS_maxes, 'o', markersize=1)
# ax2.plot(x_DogLeg_rangedates, y_DogLeg_maxes, 'o', markersize=1)
# ax2.plot(x_Phrag_Wetland_datenum, y_Phrag_Wetland)

# ax2.plot(polyX_CNRUS_max,p_CNRUS_max(polyX_CNRUS_max),"blue", label='CNR U/S Level Highs Trend')
# ax2.plot(polyX_Phrag_Wetland,p_Phrag_Wetland(polyX_Phrag_Wetland),"green", label='Phrag Wetland Level Highs Trend')
# ax2.plot(polyX_DogLeg_max,p_DogLeg_max(polyX_DogLeg_max),"orange", label='Dog Leg Level Highs Trend')
# # the line equations:
# print("y=%.6fx+(%.6f)"%(z_CNRUS_max[0],z_CNRUS_max[1]))
# print("y=%.6fx+(%.6f)"%(z_Phrag_Wetland[0],z_Phrag_Wetland[1]))
# print("y=%.6fx+(%.6f)"%(z_DogLeg_max[0],z_DogLeg_max[1]))

# # Min Plots, CNR U/S to Dog Leg - use all data for wells (not tidally influenced)
# ax3.plot(x_CNRUS_rangedates, y_CNRUS_mins, 'o', markersize=1)
# ax3.plot(x_DogLeg_rangedates, y_DogLeg_mins, 'o', markersize=1)
# ax3.plot(x_Phrag_Wetland_datenum, y_Phrag_Wetland)

# ax3.plot(polyX_CNRUS_min,p_CNRUS_min(polyX_CNRUS_min),"blue", label='CNR U/S Level Lows Trend')
# ax3.plot(polyX_Phrag_Wetland,p_Phrag_Wetland(polyX_Phrag_Wetland),"green", label='Phrag Wetland Level Lows Trend')
# ax3.plot(polyX_DogLeg_min,p_DogLeg_min(polyX_DogLeg_min),"orange", label='Dog Leg Level Lows Trend')
# # the line equations:
# print("y=%.6fx+(%.6f)"%(z_CNRUS_min[0],z_CNRUS_min[1]))
# print("y=%.6fx+(%.6f)"%(z_Phrag_Wetland[0],z_Phrag_Wetland[1]))
# print("y=%.6fx+(%.6f)"%(z_DogLeg_min[0],z_DogLeg_min[1]))

ax2.plot(x_DogLeg_datenum, y_DogLeg, label='Dog Leg Levels')
ax2.plot(x_HighToss_datenum, y_HighToss, label='High Toss Levels')
ax2.plot(x_Wet_Shrub_datenum, y_Wet_Shrub, label='Wet Shrub Levels')
ax2.plot(x_Dry_Forest_datenum, y_Dry_Forest, label='Dry Forest Levels')

idx_Wet_Shrub = np.isfinite(x_Wet_Shrub_datenum) & np.isfinite(y_Wet_Shrub)
z_Wet_Shrub = np.polyfit(x_Wet_Shrub_datenum[idx_Wet_Shrub], y_Wet_Shrub[idx_Wet_Shrub], 1)
p_Wet_Shrub = np.poly1d(z_Wet_Shrub)
idx_Dry_Forest = np.isfinite(x_Dry_Forest_datenum) & np.isfinite(y_Dry_Forest)
z_Dry_Forest = np.polyfit(x_Dry_Forest_datenum[idx_Dry_Forest], y_Dry_Forest[idx_Dry_Forest], 1)
p_Dry_Forest = np.poly1d(z_Dry_Forest)

ax2.plot(polyX_DogLeg,p_DogLeg(polyX_DogLeg),"blue", label='Dog Leg Level Trend')
polyX_Wet_Shrub = np.linspace(x_Wet_Shrub_datenum.min(), x_Wet_Shrub_datenum.max(), 100)
ax2.plot(polyX_Wet_Shrub,p_Wet_Shrub(polyX_Wet_Shrub),"green", label='Wet Shrub Level Trend')
polyX_Dry_Forest = np.linspace(x_Dry_Forest_datenum.min(), x_Dry_Forest_datenum.max(), 100)
ax2.plot(polyX_Dry_Forest,p_Dry_Forest(polyX_Dry_Forest),"red", label='Dry Forest Level Trend')
ax2.plot(polyX_HighToss,p_HighToss(polyX_HighToss),"orange", label='High Toss Level Trend')
# the line equations:
print("y=%.6fx+(%.6f)"%(z_DogLeg[0],z_DogLeg[1]))
print("y=%.6fx+(%.6f)"%(z_Wet_Shrub[0],z_Wet_Shrub[1]))
print("y=%.6fx+(%.6f)"%(z_Dry_Forest[0],z_Dry_Forest[1]))
print("y=%.6fx+(%.6f)"%(z_HighToss[0],z_HighToss[1]))

# # Max Plots, Dog Leg to High Toss - use all data for wells (not tidally influenced)
# pylab.plot(x_DogLeg_rangedates, y_DogLeg_maxes, label='Dog Leg Highs')
# pylab.plot(x_Wet_Shrub_datenum, y_Wet_Shrub, label='Wet Shrub')
# pylab.plot(x_Dry_Forest_datenum, y_Dry_Forest, label='Dry Forest')
# pylab.plot(x_HighToss_rangedates, y_HighToss_maxes, label='High Toss Highs')

# # Min Plots, Dog Leg to High Toss - use all data for wells (not tidally influenced)
# pylab.plot(x_DogLeg_rangedates, y_DogLeg_mins, label='Dog Leg Lows')
# pylab.plot(x_Wet_Shrub_datenum, y_Wet_Shrub, label='Wet Shrub')
# pylab.plot(x_Dry_Forest_datenum, y_Dry_Forest, label='Dry Forest')
# pylab.plot(x_HighToss_rangedates, y_HighToss_mins, label='High Toss Lows')

# Show X-axis major tick marks as dates
loc= mdates.AutoDateLocator()
fig.gca().xaxis.set_major_locator(loc)
fig.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
fig.autofmt_xdate()
# ax2.set_xlabel('Date', fontsize=18)
fig.text(0.5, 0.1, 'Date', fontsize=18, ha='center', va='center')
fig.text(0.08, 0.5, 'Elevation (m)', fontsize=18, ha='center', va='center', rotation='vertical')
ax1.legend()
ax2.legend()

#%% SLR affecting wells? July 2017 to December 2018 trend

CNRUSlev_july2017 = z_CNRUS[0]*x_HighToss_datenum.min()+z_CNRUS[1]
CNRUSlev_today = z_CNRUS[0]*x_Dry_Forest_datenum.max()+z_CNRUS[1]
CNRUSlr_daterange = CNRUSlev_today-CNRUSlev_july2017
CNRUSlr_oneyear = CNRUSlr_daterange/(x_Dry_Forest_datenum.max()-x_HighToss_datenum.min())*365

Phrag_Wetlandlev_july2017 = z_Phrag_Wetland[0]*x_HighToss_datenum.min()+z_Phrag_Wetland[1]
Phrag_Wetlandlev_today = z_Phrag_Wetland[0]*x_Dry_Forest_datenum.max()+z_Phrag_Wetland[1]
Phrag_Wetlandlr_daterange = Phrag_Wetlandlev_today-Phrag_Wetlandlev_july2017
Phrag_Wetlandlr_oneyear = Phrag_Wetlandlr_daterange/(x_Dry_Forest_datenum.max()-x_HighToss_datenum.min())*365

DogLeglev_july2017 = z_DogLeg[0]*x_HighToss_datenum.min()+z_DogLeg[1]
DogLeglev_today = z_DogLeg[0]*x_Dry_Forest_datenum.max()+z_DogLeg[1]
DogLeglr_daterange = DogLeglev_today-DogLeglev_july2017
DogLeglr_oneyear = DogLeglr_daterange/(x_Dry_Forest_datenum.max()-x_HighToss_datenum.min())*365

Wet_Shrublev_july2017 = z_Wet_Shrub[0]*x_HighToss_datenum.min()+z_Wet_Shrub[1]
Wet_Shrublev_today = z_Wet_Shrub[0]*x_Dry_Forest_datenum.max()+z_Wet_Shrub[1]
Wet_Shrublr_daterange = Wet_Shrublev_today-Wet_Shrublev_july2017
Wet_Shrublr_oneyear = Wet_Shrublr_daterange/(x_Dry_Forest_datenum.max()-x_HighToss_datenum.min())*365

Dry_Forestlev_july2017 = z_Dry_Forest[0]*x_HighToss_datenum.min()+z_Dry_Forest[1]
Dry_Forestlev_today = z_Dry_Forest[0]*x_Dry_Forest_datenum.max()+z_Dry_Forest[1]
Dry_Forestlr_daterange = Dry_Forestlev_today-Dry_Forestlev_july2017
Dry_Forestlr_oneyear = Dry_Forestlr_daterange/(x_Dry_Forest_datenum.max()-x_HighToss_datenum.min())*365

HighTosslev_july2017 = z_HighToss[0]*x_HighToss_datenum.min()+z_HighToss[1]
HighTosslev_today = z_HighToss[0]*x_Dry_Forest_datenum.max()+z_HighToss[1]
HighTosslr_daterange = HighTosslev_today-HighTosslev_july2017
HighTosslr_oneyear = HighTosslr_daterange/(x_Dry_Forest_datenum.max()-x_HighToss_datenum.min())*365

print("One year water level elevation change is as follows:")
print("CNR U/S = ", ("%.3f"%(CNRUSlr_oneyear)), "m.")
print("Phrag_Wetland = ", ("%.3f"%(Phrag_Wetlandlr_oneyear)), "m.")
print("Dog Leg = ", ("%.3f"%(DogLeglr_oneyear)), "m.")
print("Wet Shrub = ", ("%.3f"%(Wet_Shrublr_oneyear)), "m.")
print("Dry Forest = ", ("%.3f"%(Dry_Forestlr_oneyear)), "m.")
print("High Toss = ", ("%.3f"%(HighTosslr_oneyear)), "m.")

#%% Percent of time estuary levels are above well levels

# Need to remove nan vals and reduce measurement frequency
nan_indices_Phrag_Wetland = []
for i in range(len(y_Phrag_Wetland)):
    if np.isnan(y_Phrag_Wetland[i]):
        nan_indices_Phrag_Wetland.append(i)

y_Phrag_Wetland_nonans = y_Phrag_Wetland.tolist()
x_Phrag_Wetland_datenum_nonans = x_Phrag_Wetland_datenum.tolist()
for index in sorted(nan_indices_Phrag_Wetland, reverse=True):
    del y_Phrag_Wetland_nonans[index]
    del x_Phrag_Wetland_datenum_nonans[index]
y_Phrag_Wetland_nonans = np.array(y_Phrag_Wetland_nonans)
x_Phrag_Wetland_datenum_nonans = np.array(x_Phrag_Wetland_datenum_nonans) 

# convert numbered datetime back to standard (allows determination of minutes)
dates_Phrag_Wetland_nonans = mdates.num2date(x_Phrag_Wetland_datenum_nonans)

hourly_indices_Phrag_Wetland = []
for i in range(len(dates_Phrag_Wetland_nonans)):
    if dates_Phrag_Wetland_nonans[i].minute == 0: # minute is only zero on the hour
        hourly_indices_Phrag_Wetland.append(i)
        
y_Phrag_Wetland_hourly = []
x_Phrag_Wetland_datenum_hourly = []
for index in sorted(hourly_indices_Phrag_Wetland):
    y_Phrag_Wetland_hourly.append(y_Phrag_Wetland_nonans[index])
    x_Phrag_Wetland_datenum_hourly.append(x_Phrag_Wetland_datenum_nonans[index])
y_Phrag_Wetland_hourly = np.array(y_Phrag_Wetland_hourly)
x_Phrag_Wetland_datenum_hourly = np.array(x_Phrag_Wetland_datenum_hourly)

# Need to remove nan vals and reduce measurement frequency
nan_indices_CNRUS = []
for i in range(len(y_CNRUS)):
    if np.isnan(y_CNRUS[i]):
        nan_indices_CNRUS.append(i)

y_CNRUS_nonans = y_CNRUS.tolist()
x_CNRUS_datenum_nonans = x_CNRUS_datenum.tolist()
for index in sorted(nan_indices_CNRUS, reverse=True):
    del y_CNRUS_nonans[index]
    del x_CNRUS_datenum_nonans[index]
y_CNRUS_nonans = np.array(y_CNRUS_nonans)
x_CNRUS_datenum_nonans = np.array(x_CNRUS_datenum_nonans) 

# convert numbered datetime back to standard (allows determination of minutes)
dates_CNRUS_nonans = mdates.num2date(x_CNRUS_datenum_nonans)

hourly_indices_CNRUS = []
for i in range(len(dates_CNRUS_nonans)):
    if dates_CNRUS_nonans[i].minute == 0: # minute is only zero on the hour
        hourly_indices_CNRUS.append(i)
        
y_CNRUS_hourly = []
x_CNRUS_datenum_hourly = []
for index in sorted(hourly_indices_CNRUS):
    y_CNRUS_hourly.append(y_CNRUS_nonans[index])
    x_CNRUS_datenum_hourly.append(x_CNRUS_datenum_nonans[index])
y_CNRUS_hourly = np.array(y_CNRUS_hourly)
x_CNRUS_datenum_hourly = np.array(x_CNRUS_datenum_hourly)

#%% Distances and Coordinates

Phrag_Wetland_to_HR = 150. # meters
Dry_Forest_to_HR = 45. # meters
Wet_Shrub_to_HR = 350. # meters

# LatLong UTM in meters
CNRUS_E_N = (411883.34, 4642659.23)
Phrag_Wetland_E_N = (412505.44, 4643367.90)
DogLeg_E_N = (412671.25, 4643616.72)
Wet_Shrub_E_N = (412881.77, 4643996.02)
Dry_Forest_E_N = (412407.84, 4644033.50)
HighToss_E_N = (412353.97, 4644108.12)

Dry_Forest_to_Wet_Shrub = math.hypot(Wet_Shrub_E_N[0]-Dry_Forest_E_N[0], Wet_Shrub_E_N[1]-Dry_Forest_E_N[1])

#%% Line-color cycling

# Matplotlib makes it really simple to use evenly-spaced intervals of a colormap: you just call the colormap with 
# evenly-spaced values between 0 and 1. For example, let's plot a sinusoidal curve with different phase shifts 
# and use colors from the "cool" colormap to color each curve:
n_lines = 5
x = np.linspace(0, 10)
phase_shift = np.linspace(0, np.pi, n_lines)

color_idx = np.linspace(0, 1, n_lines)
for i, shift in zip(color_idx, phase_shift):
    plt.plot(x, np.sin(x - shift), color=plt.cm.cool(i), lw=3)

plt.show()

# Alternatively, you can set the color cycle of the plot axes:
ax = plt.axes()
ax.set_color_cycle([plt.cm.cool(i) for i in np.linspace(0, 1, n_lines)])
for shift in phase_shift:
    plt.plot(x, np.sin(x - shift), lw=3)
    
# I prefer this method because the loop definition is a bit simpler (i.e., no call to zip). I've added this method 
# to a utility package called mpltools:
# from mpltools import color # need to install mpltools

# ax = plt.axes()
# color.cycle_cmap(n_lines, cmap=plt.cm.cool, ax=ax)
# for shift in phase_shift:
#     plt.plot(x, np.sin(x - shift), lw=3)

# Below, I plot a sinusoidal curve with different rates of exponential decay and label those rates with different colors:
pvalues = np.logspace(-1, 0, 4)
pmin = pvalues[0]
pmax = pvalues[-1]

def norm(pval):
    return (pval - pmin) / float(pmax - pmin)

x = np.linspace(0, 10)
for pval in pvalues:
    y = np.sin(x) * np.exp(-pval * x)
    color = plt.cm.YlOrBr(norm(pval))
    plt.plot(x, y, 's', color=color)

leg = plt.legend(['%0.1f' % v for v in pvalues], ncol=2)
leg.set_title('decay rate')

plt.show()

# To simplify this process, I wrote a simple factory function (function that returns a function) called color_mapper:
# from mpltools import color

# pvalues = np.logspace(-1, 0, 4)
# prange = [pvalues[0], pvalues[-1]]
# map_color = color.color_mapper(prange, cmap='YlOrBr')

# x = np.linspace(0, 10)
# for pval in pvalues:
#     y = np.sin(x) * np.exp(-pval * x)
#     plt.plot(x, y, 's', color=map_color(pval))