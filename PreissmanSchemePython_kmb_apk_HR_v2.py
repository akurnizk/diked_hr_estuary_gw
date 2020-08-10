# -*- coding: utf-8 -*-
"""
FORTRAN code, translated
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

# !      COMPUTATION OF UNSTEADY, FREE-SURFACE FLOWS BY PREISSMANN
# !      IMPLICIT SCHEME IN A TRAPEZOIDAL CHANNEL
# !      CONSTANT FLOW DEPTH ALONG THE CHANNEL IS SPECIFIED AS
# !      INITIAL CONDITION.
# !      TRANSIENT CONDITIONS ARE PRODUCED BY THE SUDDEN CLOSURE
# !      OF A DOWNSTREAM GATE.
# !
# !    ************************* NOTATION ************************
# !
# !     ALPHA = WEIGHTING COEFFICIENT
# !     AR = STATEMENT FUNCTION FOR FLOW AREA
# !     B0 = CHANNEL BOTTOM WIDTH
# !     C = CELERITY
# !     CENTR = MOMENT OF FLOW AREA
# !     CMN = MANNING'S COEFFICIENT
# !     CHL = CHANNEL LENGTH
# !     G = ACCELERATION OF GRAVITY
# !     HR = STATEMENT FUNCTION FOR HYDRAULIC RADIUS
# !     IPRINT = COUNTER FOR PRINTING RESULTS
# !     MAXITER = MAXIMUM NUMBER OF ITERATIONS
# !     NSEC = NUMBER OF CHANNEL SECTIONS
# !     Q0 = INITIAL STEADY STATE DISCHARGE
# !     S = CHANNEL LATERAL SLOPE
# !     S0 = CHANNEL BOTTOM SLOPE
# !     TLAST = TIME FOR TRANSIENT FLOW COMPUTATION
# !     TOL = TOLERANCE FOR INTERATIONS
# !     TOP = STATEMENT FUNCTION FOR WATER TOP WIDTH
# !     V = FLOW VELOCITY
# !     Y = FLOW DEPTH
# !     UNITS = ALPHANUMERIC VARIABLE FOR UNITS SYSTEM
# !   ************************************************************
# !

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

HR_all_resam_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","HR_All_Data_Resampled_HourlyMeans_8272017-1212020.csv")) # Calculated
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
loc= mdates.AutoDateLocator()
plt.gca().xaxis.set_major_locator(loc)
plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
plt.gcf().autofmt_xdate()
plt.xlabel('Date', fontsize=22)
plt.ylabel('Elevation (m), Discharge (m^3/s)', fontsize=22)
plt.legend(loc='upper right')

#%% Take slices for local analysis.

HR_all_slice_start_i = HR_all_resam_df["Gage height, m, Ocean side"].first_valid_index()
HR_all_slice_end_i = HR_all_resam_df["Discharge, cms"].last_valid_index()

"""
Use Oceanside data start index up to end of measured discharge data.
"""

HR_all_resam_slice = HR_all_resam_df[HR_all_slice_start_i:HR_all_slice_end_i] # choose point to best fit final curve, calculated
HR_all_resam_slice.interpolate(method='index', limit=12, inplace=True)

x_cumtime_resam_slice = pd.to_timedelta(HR_all_resam_slice["datetime"]-HR_all_resam_slice["datetime"].iloc[0]).astype('timedelta64[s]')
cumtime_slice = x_cumtime_resam_slice.iloc[-1]

# nans, x = nan_helper(disch_onecycle)
# disch_onecycle[nans] = np.interp(x(nans), x(~nans), disch_onecycle[~nans])

plt.figure()
plt.scatter(x_cumtime_resam_slice, np.array(HR_all_resam_slice["Discharge, Dike Calc, cms"]), label = 'Delta Storage in HR')

tidecyc_hr = 89400/60/60
testtime_hr = cumtime_slice/60/60

#%% Mean levels and ranges for High Toss Boundary

# Mean Levels
elev_hightoss_avg = np.nanmean(HR_all_resam_df["High Toss Water Level, NAVD88"]) # + 0.06 # assume some dampening from culvert (causes measured avgs to be lower)
elev_cnrus_avg = np.nanmean(HR_all_resam_df["CNR U/S Water Level, NAVD88"])
elev_hrside_avg = np.nanmean(HR_all_resam_df["Gage height, m, HR side"])

# Ranges and Amplitudes
range_hightoss_avg = 0.52
amp_hightoss_avg = range_hightoss_avg/2
range_cnrus_avg = 0.71
amp_cnrus_avg = range_cnrus_avg/2

#%% Depth-Storage Curve

storage_slice = []
for row in range(len(HR_all_resam_slice)):
    wsp = np.linspace(HR_all_resam_slice["High Toss Water Level, NAVD88"].iloc[row],HR_all_resam_slice["CNR U/S Water Level, NAVD88"].iloc[row],40)
    # wsp = D + np.amin(elevs_interp,axis=1) # add depth to lowest point in channel
    wsp_mask = np.vstack([(elevs_interp[xsec,:] <= wsp[xsec]) for xsec in range(len(elevs_interp))])
    area_all = []
    for xsec in range(len(elevs_interp)):
        area_xsec = []
        for xsec_i in range(len(elevs_interp.T)-1):
            pt_to_pt_x = np.sqrt((out_x_stacked[xsec,xsec_i]-out_x_stacked[xsec,xsec_i+1])**2+(out_y_stacked[xsec,xsec_i]-out_y_stacked[xsec,xsec_i+1])**2)
            pt_to_pt_y = abs(elevs_interp[xsec,xsec_i]-elevs_interp[xsec,xsec_i+1])
            if (wsp_mask[xsec,xsec_i] != wsp_mask[xsec,xsec_i+1]):
                pt_to_pt_y = wsp[xsec]-min(elevs_interp[xsec,xsec_i],elevs_interp[xsec,xsec_i+1])
                pt_to_pt_x = pt_to_pt_x*pt_to_pt_y/abs(elevs_interp[xsec,xsec_i]-elevs_interp[xsec,xsec_i+1])
                wsp_mask[xsec,xsec_i] = True
            area_up = min(wsp[xsec]-elevs_interp[xsec,xsec_i],wsp[xsec]-elevs_interp[xsec,xsec_i+1])*pt_to_pt_x
            area_triang = pt_to_pt_y*pt_to_pt_x/2
            area_xsec.append(((area_up+area_triang)*wsp_mask[xsec,xsec_i]))
        area_all.append((np.nansum(area_xsec)))
    area_all_row = np.array(area_all)
    area_all_avg = (area_all_row[:-1]+area_all_row[1:])/2
    dx_storage = area_all_avg*min_dist_dx
    total_storage = np.nansum(dx_storage)
    storage_slice.append(total_storage)
storage_slice_arr = np.array(storage_slice)

plt.figure()
plt.scatter(np.array(HR_all_resam_slice["CNR U/S Water Level, NAVD88"]),storage_slice_arr,label="Change in HR Storage v. CNR")
plt.scatter(np.array(HR_all_resam_slice["Gage height, m, Ocean side"]),storage_slice_arr,label="Change in HR Storage v. Ocean")
plt.scatter(np.array(HR_all_resam_slice["High Toss Water Level, NAVD88"]),storage_slice_arr,label="Change in HR Storage v. HT")
plt.xlabel('Sensor Levels', fontsize=22)
plt.ylabel('Storage (m^3)', fontsize=22)
plt.legend()
plt.scatter(np.array(HR_all_resam_slice["CNR U/S Water Level, NAVD88"]),np.array(HR_all_resam_slice["High Toss Water Level, NAVD88"]),label="Change in HT elev")

#%% Curve fits of CTD Data

def HTfunc(t, a, b, c):
    return hightoss_mean + a*t + amp_hightoss_avg*np.exp(−b*t)*np.sin(c*math.pi*t/22350)
    
popt, pcov = curve_fit(func, xdata, ydata)

def CNRfunc(t, a, b, c):
    cnrus_mean + a*t + amp_cnrus_avg*np.exp(−b*t)*np.sin(c*math.pi*t/22350)

#%% Necessary functions, defined at end of Fortran code
def elev0(t, time_onecycle_elev, elev_hightoss_onecycle):
    """
    Statement function for upstream depth.
    When ready, include HRside measurements as input and use in formula?
    """
    f_elev0 = interp1d(time_onecycle_elev,elev_hightoss_onecycle)
    temp_elev0 = f_elev0(t)
    # return 0.86 + amp_hightoss_avg*math.sin(math.pi*t/22350)
    return np.mean(temp_elev0)
def elevD(t, time_onecycle_elev, elev_cnrus_onecycle):
    """
    Statement function for downstream depth.
    When ready, include HRside measurements as input and use in formula?
    """
    f_elevd = interp1d(time_onecycle_elev,elev_cnrus_onecycle)
    temp_elevd = f_elevd(t)
    # return 1.15 + amp_cnrus_avg*math.sin(math.pi*t/22350)
    return np.mean(temp_elevd)

def AR(D,B):
    """
    Satement function for flow area.
    """
    return (B+D*s)*D

def HR(D,B):
    """
    Satement function for hydraulic radius.
    """
    return (B+D*s)*D/(B+2*D*np.sqrt(1+s*s))

def TOP(D,B):
    """
    Satement function for water top width.
    """
    return B+2*D*s

def CENTR(D,B):
    """
    Satement function for moment of flow area.
    """
    return D*D*(B/2+D*s/3)

def DCENDY(D,B):
    """
    Satement function for derivative of moment of flow area with respect to depth.
    """
    return D*(B+D*s)

def BW(D,B): # not used?
    return B + 2*s*D

def make_sf(Y,V,B,cmn_squared):
    """
    Satement function for friction slope.
    """
    return abs(V)*V*cmn_squared/HR(Y,B)**1.333

def make_C2(Y,V,ARi,ARiP1,cmn_squared,s0,b0,grav):
    sf1 = make_sf(Y[:-1],V[:-1],b0[:-1],cmn_squared)
    sf2 = make_sf(Y[1:],V[1:],b0[1:],cmn_squared)
    term1 = -dt*(1-alpha)*(grav*ARiP1*(s0-sf2)+grav*ARi*(s0-sf1))
    term2 = -(V[:-1]*ARi+V[1:]*ARiP1)
    term3 = dtx2*(1-alpha)*((V[1:]**2)*ARiP1 + \
                  grav*CENTR(Y[1:],b0[1:]) - \
                  (V[:-1]**2)*ARi-grav*CENTR(Y[:-1],b0[:-1]))
    C2 = term1 + term2 + term3
    return C2

def MATSOL(N,A):
    """ Matrix solver.
    
    ******************************************************************
    !      SIMULTANEOUS SOLUTION OF THE SYSTEM OF EQUATIONS
    """

    X = np.zeros((N+1),dtype=float) # X.shape = N+1
    NROW = np.arange(0,N+1,dtype=int) # NROW.shape = N+1

    for i in np.arange(N): # loop through rows
        AMAX = np.max(np.abs(A[NROW[i:],i])) # max value for column, all later rows
        ip = np.argmax(np.abs(A[NROW[i:],i]))+i # index of above
        
        if(abs(AMAX) <= 1E-08):
            print('Singular matrix --> No unique solution exists')
            return X
        
        if(NROW[i] != NROW[ip]): # swap rows
            NC = NROW[i].copy()
            NROW[i] = NROW[ip].copy()
            NROW[ip] = NC.copy()
        
        
        COEF = A[NROW[i+1:],i]/A[NROW[i],i] # normalize column values by maximum magnitude value (AMAX > 0)
        A[NROW[i+1:],i+1:] = A[NROW[i+1:],i+1:] - np.dot(COEF[:,None],A[NROW[i],i+1:][None,:]) # normalize/reduce matrix
        
    
    if(abs(A[NROW[N],N]) <= 1E-08):
        print('Singular matrix --> No unique solution exists')
        return X
    
    X[N] = A[NROW[N],N+1]/A[NROW[N],N] # downstream edge
    i = N-1
    while (i >= 0):
#         SUMM = 0.0
#         j = i+1
        
         SUMM = np.sum(A[NROW[i],i+1:N+1]*X[i+1:N+1]) # do not include final column
        
#         while (j <= N-1):
#             SUMM = A[NROW[i],j]*X[j] + SUMM
#             j = j+1
        # print(SUMM,SUMM2)
        
         X[i] = (A[NROW[i],N+1] - SUMM)/A[NROW[i],i]
         i = i-1
    return X

def NORMAL_D(YNORM,Q,CMAN,B0,S,S0):
    """ Computation of normal depth.
    
    !====================================================================
    !             NORMAL DEPTH
    !====================================================================
    """
  
    if (Q < 0.):
        YNORM = 0.
        return
  
    C1 = (CMAN*Q)/np.sqrt(S0)
    C2 = 2*np.sqrt(1 + S*S)
    YNORM = (CMAN**2*(Q/B0)**2/S0)**0.3333
    for i in range(999):
        FY = AR(YNORM)*HR(YNORM)**0.6667 - C1
        DFDY = 1.6667*BW(YNORM)*HR(YNORM)**0.6667 - 0.6667*HR(YNORM)**1.6667*C2
        YNEW = YNORM - FY/DFDY
        ERR = abs((YNEW - YNORM)/YNEW)
        YNORM = YNEW.copy()
        if (ERR < 1.0E-06):
            return
    return

#%% Initialize arrays below variables based on nsec

grav = 9.81 # m/s^2
nsec = len(min_dist_dx) # number of spaces between xsecs
np11 = 2*nsec + 2
tlast = 89400*1 # time for transient flow computation (measurements are at 5 min (300s) intervals - use these?)
# tlast = 89400/4 # time for testing (mean to high to mean)
iprint = 1 # counter for printing results
chl = np.nansum(min_dist_dx) # channel length
s = 2 # channel lateral slope (assumed for now)

# minimum elevations for each xsec
min_elevs = np.amin(elevs_interp,axis=1)
# use mean levels up & down and assumed linear profile to set starting depths
# y_act = np.linspace(elev_hightoss_avg,elev_cnrus_avg,40)-min_elevs # starting depths
y_act = np.linspace(elev_hightoss_avg,elev_cnrus_avg,40)-min_elevs

"""
Use actual depths to find actual water surface profile.
"""
# wsp should be np.linspace(elev_hightoss_avg,elev_cnrus_avg,40)
wsp = y_act + min_elevs # add depth to lowest point in channel
wsp_mask = np.vstack([(elevs_interp[xsec,:] <= wsp[xsec]) for xsec in range(len(elevs_interp))])

"""
Calculating b0 from top widths
"""
# top_width_all = [] # use channel geometry and starting elevs to find top widths
# for xsec in range(len(elevs_interp)):
#     top_width_xsec = []
#     for xsec_i in range(len(elevs_interp.T)-1):
#         pt_to_pt_x = np.sqrt((out_x_stacked[xsec,xsec_i]-out_x_stacked[xsec,xsec_i+1])**2+(out_y_stacked[xsec,xsec_i]-out_y_stacked[xsec,xsec_i+1])**2)
#         pt_to_pt_y = abs(elevs_interp[xsec,xsec_i]-elevs_interp[xsec,xsec_i+1])
#         if (wsp_mask[xsec,xsec_i] != wsp_mask[xsec,xsec_i+1]):
#             y_edge = wsp[xsec]-min(elevs_interp[xsec,xsec_i],elevs_interp[xsec,xsec_i+1])
#             pt_to_pt_x = pt_to_pt_x*y_edge/pt_to_pt_y
#             wsp_mask[xsec,xsec_i] = True
#         top_width_xsec.append((pt_to_pt_x*wsp_mask[xsec,xsec_i]))
#     top_width_all.append((np.nansum(top_width_xsec)))
# top_widths = np.array(top_width_all)

# b0_act = []
# for xsec in range(nsec+1):
#     b0_act_temp = top_widths[xsec]-2*y_act[xsec]*s # solve for average bottom width (given y0_act means, top-width, and s)
#     b0_act.append(b0_act_temp)
# b0_act = np.array(b0_act)

"""
Calculating b0 from areas
"""
area_all = []
for xsec in range(len(elevs_interp)):
    area_xsec = []
    for xsec_i in range(len(elevs_interp.T)-1):
        pt_to_pt_x = np.sqrt((out_x_stacked[xsec,xsec_i]-out_x_stacked[xsec,xsec_i+1])**2+(out_y_stacked[xsec,xsec_i]-out_y_stacked[xsec,xsec_i+1])**2)
        pt_to_pt_y = abs(elevs_interp[xsec,xsec_i]-elevs_interp[xsec,xsec_i+1])
        if (wsp_mask[xsec,xsec_i] != wsp_mask[xsec,xsec_i+1]):
            pt_to_pt_y = wsp[xsec]-min(elevs_interp[xsec,xsec_i],elevs_interp[xsec,xsec_i+1])
            pt_to_pt_x = pt_to_pt_x*pt_to_pt_y/abs(elevs_interp[xsec,xsec_i]-elevs_interp[xsec,xsec_i+1])
            wsp_mask[xsec,xsec_i] = True
        area_up = min(wsp[xsec]-elevs_interp[xsec,xsec_i],wsp[xsec]-elevs_interp[xsec,xsec_i+1])*pt_to_pt_x
        area_triang = pt_to_pt_y*pt_to_pt_x/2
        area_xsec.append(((area_up+area_triang)*wsp_mask[xsec,xsec_i]))
    area_all.append((np.nansum(area_xsec)))
area_all = np.array(area_all)

b0_act = []
for xsec in range(nsec+1): # find base widths of trapezoidal channel with same area.
    b0_act_temp = area_all[xsec]/y_act[xsec] - y_act[xsec]*s # solve for average bottom width (given y0_act means, top-width, and s)
    b0_act.append(b0_act_temp)
b0_act = np.array(b0_act)

xsec_loc = np.cumsum(min_dist_dx)
xsec_loc = np.append([0],xsec_loc)
plt.plot(xsec_loc,b0_act,label='b0_act data')
"""
Linear function that fits the bottom widths for an average bottom width change
"""
idx_b0 = np.isfinite(xsec_loc) & np.isfinite(b0_act)
# idx_b0[-2:] = False
z_b0 = np.polyfit(xsec_loc[idx_b0], b0_act[idx_b0], 1)
p_b0 = np.poly1d(z_b0)
polyX_b0 = np.linspace(xsec_loc.min(), xsec_loc.max(), 100)
pylab.plot(polyX_b0,p_b0(polyX_b0),"green", label='Average Base Width of HR, HT to Dike')

b0 = z_b0[0]*xsec_loc + z_b0[1] # channel bottom widths
# b0_theor = np.ones_like(b0)*50 # no width change tester
b0_theor = np.linspace(48,52,40) # taper tester
z_b0_theor = np.polyfit(xsec_loc,b0_theor,1)
b0 = z_b0_theor[0]*xsec_loc + z_b0_theor[1] # channel bottom widths

plt.plot(xsec_loc,b0,label='tester')
plt.legend()

plt.figure()
plt.plot(xsec_loc,min_elevs,label='elev data')
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

cmn = 0.025 # manning's coefficient, original
cmn = 0.06 # tester (use spaulding & grilli?) influences discharge amplitude
# s0 = 0.001 # channel bottom slope - ignore. use dem to find xsec area
# q0 = 30 # initial steady state discharge
q0_avg = 0.6 # Average annual fw discharge values
q0 = disch_onecycle[0] # approximate uniform discharge at initial conditions
# y0 = 1.58 # upstream initial flow depth (high toss)
# y0 = elev_hightoss_avg - z_elevs[1] # high toss water level minus elevation (theoretical curve)
y0 = elev_hightoss_arr[0] - z_elevs[1] # actual data
# yd = 1.58 # flow depth at lower end (initial condition) - need to solve so that adjust so that Q_calc_channel_out = Q_dike
# yd = elev_cnrus_avg - (-s0*xsec_loc[-1] + z_elevs[1]) # theoretical curve
yd = elev_cnrus_arr[0] - (-s0*xsec_loc[-1] + z_elevs[1]) # actual data
alpha = 1 # weighting coefficient
tol = 0.0001 # tolerance for iterations
maxiter = 50 # maximum number of iterations

C1, C2 = ( np.array([np.nan]*(nsec)) for i in range(2))

T = 0 # steady state, initial time
cmn_squared = cmn**2 # gravity units are metric, so no changes are necessary.

# Z = s0 * chl - np.arange(nsec+1) * dx * s0 # bottom elev array
Z = z_elevs[0]*xsec_loc + z_elevs[1]
# Y = np.ones_like(Z)*y0 # depth array
# Y[(Z+y0) < yd] = yd - Z[(Z+y0) < yd] # Make starting depths positive.
# Y = np.linspace(elev_hightoss_avg,elev_cnrus_avg,40) - Z # theoretical
Y = np.linspace(elev_hightoss_arr[0],elev_cnrus_arr[0],40) - Z # actual
V = q0/AR(Y,b0)

# Steady state conditions
# c = np.sqrt(grav*AR(y0)/TOP(y0)) # celerity, original
c = np.sqrt(grav*np.nanmean(AR(Y,b0))/np.nanmean(TOP(Y,b0))) # mean celerity
# v0 = q0/AR(y0) # flow velocity, original
v0 = q0_avg/np.nanmean(AR(Y,b0)) # mean flow velocity
# dx = chl/nsec # original
dx = min_dist_dx

# dt = dx/(v0+c) # original time step length
dt = (chl/nsec)/(v0+c) # avgerage time step length
dtx2 = 2*dt/dx
yres = y0
i = 0
#np1 = nsec # remove for clarity. In fortran, np1 = nsec+1, python starts at 0 giving extra index

#%%

iflag = 0
ip = iprint
# WRITE(10,"('T=,',F8.3,',Z=',60(',',F6.2))")0.,(Z(I),I = 1,NP1)
PREISS_H_out = np.concatenate([['T=',T,'Z='],[float(x) for x in Z]])
PREISS_Q_out = PREISS_H_out.copy() # added this to have a header row with the bottom elevations for the discharge array.

# !
# !     COMPUTE TRANSIENT CONDITIONS
# !
H_out = []
Q_out = []
T_out = []

# Insert initial conditions
H_out.append(Y+Z)
Q_out.append(V*AR(Y,b0))
T_out.append(T)

plt.figure()
while (T <= tlast) & (iflag == 0): # time loop, ntimes = tlast/dt
    print("Model time = {0:3.2f} s".format(T))
    print("High Toss V = {0:3.2f} m/s".format(V[0]))
    print("High Toss AR = {0:3.2f} m^2".format(AR(Y,b0)[0]))
    ITER = 0
    if (iprint == ip):
        ip = 0
        PREISS_H_out = np.vstack((PREISS_H_out,np.concatenate([['T=',T,'H='],[float(x) for x in (Y+Z)]])))
        PREISS_Q_out = np.vstack((PREISS_Q_out,np.concatenate([['T=',T,'Q='],[float(x) for x in (V*AR(Y,b0))]])))
    T = T + dt # original
    # dt_shift = 90 # seconds of offset between levels at CNR U/S and High Toss for hypo curve
    yres = elev0(T, time_onecycle_elev, elev_hightoss_onecycle) - Z[0]
    yd = elevD(T, time_onecycle_elev, elev_cnrus_onecycle) - Z[-1] # for changing downstream levels
    
    # !
    # !     GENERATE SYSTEM OF EQUATIONS
    # !
    
    ARi = AR(Y[:-1],b0[:-1]) # calculate flow area at upstream section
    ARiP1 = AR(Y[1:],b0[1:]) # calculate flow area at downstream section
    C1 = dtx2*(1-alpha)*(ARiP1*V[1:]-ARi*V[:-1])-ARi-ARiP1
    C2 = make_C2(Y,V,ARi,ARiP1,cmn_squared,s0,b0,grav)
    
    SUMM = tol+10
    for L in range(1,1000):
        plt.plot(Y,".",label=L-1)
        if (SUMM > tol):
            EQN = np.zeros((np11,np11+1),dtype=float) # should generate the same array?
            ITER = ITER+1    
            # !    
            # !        INTERIOR NODES
            # !    
            
            ARi = AR(Y[:-1],b0[:-1]) # calculate flow area at upstream section
            ARiP1 = AR(Y[1:],b0[1:]) # calculate flow area at downstream section
            row_inds1 = 2*np.arange(nsec,dtype=int)+1 # every other row, starting at 1 (2nd row)
            EQN[row_inds1,np11]=-(ARi+ARiP1+dtx2*alpha*(V[1:]*ARiP1-V[:-1]*ARi)+C1) # sets last column
            
            sf1 = make_sf(Y[:-1],V[:-1],b0[:-1],cmn_squared)
            sf2 = make_sf(Y[1:],V[1:],b0[1:],cmn_squared)
            term1 = term1 = dtx2*alpha*((V[1:]**2)*ARiP1 + grav*CENTR(Y[1:],b0[1:])-(V[:-1]**2)*ARi-grav*CENTR(Y[:-1],b0[:-1]))
            term2 = -alpha*dt*grav*((s0-sf2)*ARiP1+(s0-sf1)*ARi)
            EQN[row_inds1+1,np11] = -(V[:-1]*ARi+V[1:]*ARiP1+term1+term2+C2) # every other row, starting at 2 (3rd row)
            
            daY1 = TOP(Y[:-1],b0[:-1])
            daY2 = TOP(Y[1:],b0[1:])
            EQN[row_inds1,row_inds1-1] = daY1*(1-dtx2*alpha*V[:-1])
            EQN[row_inds1,row_inds1] = -dtx2*alpha*ARi
            EQN[row_inds1,row_inds1+1] = daY2*(1+dtx2*alpha*V[1:])
            EQN[row_inds1,row_inds1+2] = dtx2*alpha*ARiP1
            
            dcdY1 = DCENDY(Y[:-1],b0[:-1])
            dcdY2 = DCENDY(Y[1:],b0[1:])
            dsdV1 = 2*V[:-1]*cmn_squared/HR(Y[:-1],b0[:-1])**1.333
            dsdV2 = 2*V[1:]*cmn_squared/HR(Y[1:],b0[1:])**1.333
            
            PERi = ARi/HR(Y[:-1],b0[:-1])
            term1 = 2*np.sqrt(1+s**2)*ARi - daY1*PERi
            term2 = HR(Y[:-1],b0[:-1])**0.333*ARi**2
            dsdY1 = 1.333*V[:-1]*abs(V[:-1])*cmn_squared*term1/term2
            
            PERiP1 = ARiP1/HR(Y[1:],b0[1:])
            term1 = 2*np.sqrt(1+s**2)*ARiP1-daY2*PERiP1
            term2 = (HR(Y[1:],b0[1:])**0.333)*(ARiP1**2)
            dsdY2 = 1.333*V[1:]*abs(V[1:])*cmn_squared*term1/term2
            
            term1 = -dtx2*alpha*((V[:-1]**2)*daY1 + grav*dcdY1)
            term2 = -grav*dt*alpha*(s0-sf1)*daY1
            EQN[row_inds1+1,row_inds1-1]=V[:-1]*daY1+term1+term2+grav*dt*alpha*ARi*dsdY1
            EQN[row_inds1+1,row_inds1]=ARi-dtx2*alpha*2*V[:-1]*ARi+grav*dt*alpha*ARi*dsdV1
        
            term1 = dtx2*alpha*((V[1:]**2)*daY2+grav*dcdY2)
            term2 = -grav*dt*grav*(s0-sf2)*daY2
            EQN[row_inds1+1,row_inds1+1] = V[1:]*daY2+term1+term2+alpha*dt*grav*ARiP1*dsdY2
            EQN[row_inds1+1,row_inds1+2] = ARiP1+dtx2*alpha*2*V[1:]*ARiP1+alpha*dt*grav*dsdV2*ARiP1
            
            
            # !
            # !         UPSTREAM END (Y given)
            # !
            EQN[0,0] = 1.0
            EQN[0,np11] = -(Y[0]-yres)
            # !
            # !         UPSTREAM END (V given)
            # !
            # EQN[0,0]    = 1.0
            # EQN[1,np11] = (V[0] - v0)                              # ! ok


            # !
            # !         DOWNSTREAM END (NO OUTFLOW)
            # !
            # EQN[-1,-2] = 1.
            # EQN[-1,np11] = 0. - V[-1]
            # !       
            # !         DOWNSTREAM END (Y given)
            # !
            EQN[-1,-2] = 1.
            EQN[-1,np11] =  Y[-1] - yd                         # ! ok
       
            # Run implicit solution 
            DF = MATSOL(np11-1,EQN)
            
            # Organize output
            SUMM = np.sum(DF)
            Y = Y + DF[::2]
            V = V + DF[1::2]
                
            #CHECK NUMBER OF ITERATIONS 
            if (ITER > maxiter):
                iflag = 1
                SUMM = tol
        else:
            break
    ip = ip+1
    H_out.append(Y+Z)
    Q_temp = V*AR(Y,b0)
    Q_out.append(Q_temp)
    T_out.append(T)

H_out = np.array(H_out)
Q_out = np.array(Q_out)
T_out = np.array(T_out)

plt.figure()
plt.scatter(time_onecycle_elev, elev_hightoss_onecycle, label = "Measured Elevation, m, High Toss")
plt.scatter(time_onecycle_elev, elev_cnrus_onecycle, label = "Measured Elevation, m, CNR U/S")
plt.scatter(time_onecycle_Qcalc, disch_onecycle, label = 'Measured Discharge, cms, through dike')
plt.scatter(T_out,Q_out[:,-1], label = "Calculated Downstream Discharge, cms, Preissmann")
plt.legend()

# IF(IFLAG.EQ.1) WRITE(6,"('MAXIMUM NUMBER OF ITERATIONS EXCEEDED')")
# STOP
# END

if (iflag == 1):
    print("Maximum number of iterations exceeded")
