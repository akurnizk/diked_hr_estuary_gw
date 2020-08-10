# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:23:12 2020

@author: akurnizk
"""

import os
import scipy
import hydroeval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.optimize import fsolve

import seaborn as sns; sns.set()

import matplotlib as mpl
mpl.rc('xtick', labelsize=22)     
mpl.rc('ytick', labelsize=22)
mpl.rcParams['pdf.fonttype'] = 42

map_dir = r'E:\Maps' # retrieved files from https://viewer.nationalmap.gov/basic/
data_dir = os.path.join('E:\Data')

#%% Load data to be used in analysis: Time series of levels just outside and inside of dike.

HR_all_resam_1hr_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","HR_All_Data_Resampled_HourlyMeans_8272017-1212020.csv")) # Calculated
data_cols = HR_all_resam_1hr_df.columns.drop("datetime")
HR_all_resam_1hr_df[data_cols] = HR_all_resam_1hr_df[data_cols].apply(pd.to_numeric, errors='coerce')
HR_all_resam_1hr_df["datetime"] = pd.to_datetime(HR_all_resam_1hr_df["datetime"])

# Hourly Predictions 1946 to 2100
pred_to_2100_dtHRocean_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","Dike_Data_HourlyPred_111946_12312100.csv")) # Calculated
pred_to_2100_dtHRocean_df["datetime"] = pd.to_datetime(pred_to_2100_dtHRocean_df["datetime"])

#%% Dike Geometry Constants (Obtained from construction documents)

# ALL METRIC (m, kg, s)
# Declare invert elevations - should be slightly higher on river side.
# Can also use a slope-distance method to determine one or the other
inv_el_open = -1.064
inv_el_HRside = -0.928

# Bottom elevation of a single sluice gate
# For more sluice gates, determine opening elevation of EACH.
sluice_bot_el = -0.579
y_sluice_open = sluice_bot_el-inv_el_open # opening height (diff of elevations)

# Width of sluice culvert (again, need multiple for multiple)
L_sluice_culv = 1.829
A_sluice_open = y_sluice_open*L_sluice_culv # area of sluice opening (for submerged flow comp)

# Width of tide culverts and tide gates.
L_flaps_in = 1.829
L_flaps_out = 2.057

# Initial angle of opening for flap gate. For use in moment analysis.
angle_init_flaps = 0.0872 # radians, ~ 5 degrees

# Density of local seawater (assume enough mixing for there to be negligible difference)
dens_seawater = 1018 # kg/m^3, average is roughly the same on both sides of the dike.

# Standard acceleration of gravity - should be okay since at sea level
grav = scipy.constants.g # m/s^2

# Weight of flap gates (assume same for all - could use better estimate)
W_gate = 2000 # Newtons -> see excel calculations using gate parts, volumes, and densities.

# Height of flap gate
h_gate = 2.317 # meters from flap gate bottom to hinge. Assume weight is uniformly distributed.

# Hinge to invert length (should be slightly less as long as gate covers opening)
d_hinge_to_inv = 2.286
hinge_el_open = inv_el_open+d_hinge_to_inv # elevation of hinge

# Headloss parameters, estimate from literature. These are from WHG report
HL_max = 0.9 # maximum flap headloss, meters
HLsluice_max = 1.0 # maximum sluice flood headgain, meters
D_HL = 0.4 # flap headloss parameter, meters
Dsluice_HL = 1.0 # sluice flood headloss parameter, meters

#%% Initialize input coefficients/flood conditions

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

"""
Coefficients from WHG Report (for comparison)
"""
C_one_flood = 1.375 # Discharge coefficient for supercritical b-c weir flow
C_two_flood = 1.375 # Dischrage coefficient for subcritical b-c weir flow
C_three_flood = 1.4 # Discharge coefficient for free sluice flow
C_four_flood = 1.35 # Discharge coefficient for submerged orifice flow
C_one_ebb = 1
C_two_ebb = 1
C_three_ebb = 0.6
C_four_ebb = 0.8 

"""
Ebb C_d means. Optimized by fitting to known discharge curve.
"""
C_d_ebb_free_mean = 0.8
C_d_ebb_transit_mean = 0.6
C_d_ebb_submer_or_mean = 0.9
C_d_ebb_subcrit_weir_mean = 0.7
C_d_ebb_supcrit_weir_mean = 1.0
C_d_ebb_flap_subcrit_weir_mean = 0.7
C_d_ebb_flap_supcrit_weir_mean = 0.85
"""
Flood C_d means.  Optimized by fitting to known discharge curve.
"""
C_d_flood_free_mean = 0.75
C_d_flood_transit_mean = 0.6
C_d_flood_submer_or_mean = 1.1

"""
Initialize measured/theoretical levels, empty array for discharge calculations 
"""
# # Levels relative to culvert invert at sluice/flaps.
# H_sea_lev = np.array(HR_all_resam_1hr_df["Gage height, m, Ocean side"] - inv_el_open)
# y_d_HR_lev = np.array(HR_all_resam_1hr_df["Gage height, m, HR side"] - inv_el_open)

# # Vertical distances from flap gate hinge to water levels.
# d_hinge_to_H = np.array(hinge_el_open - HR_all_resam_1hr_df["Gage height, m, Ocean side"])
# d_hinge_to_y_d = np.array(hinge_el_open - HR_all_resam_1hr_df["Gage height, m, HR side"])

# Levels relative to culvert invert at sluice/flaps.
H_sea_lev = np.array(pred_to_2100_dtHRocean_df["Gage height, m, Ocean side, Predicted"] - inv_el_open)
y_d_HR_lev = np.array(pred_to_2100_dtHRocean_df["Gage height, m, HR side, Predicted"] - inv_el_open)

# Vertical distances from flap gate hinge to water levels.
d_hinge_to_H = np.array(hinge_el_open - pred_to_2100_dtHRocean_df["Gage height, m, Ocean side, Predicted"])
d_hinge_to_y_d = np.array(hinge_el_open - pred_to_2100_dtHRocean_df["Gage height, m, HR side, Predicted"])



"""
If (H_sea_lev > y_d_HR_lev): # If sea level is greater than HR level -> Negative Flow (Flood Tide, Flap Gates Closed)
"""
# Avoiding RuntimeWarnings...

# # Flow type conditions
# flood_free_cond = empty_1d_datalen.copy()
# flood_submer_or_cond = empty_1d_datalen.copy()
# flood_transit_cond = empty_1d_datalen.copy()

# # If either side of dike has nan val, set condition to false
# flood_free_cond[np.isnan(y_d_HR_lev/H_sea_lev)] = False
# flood_submer_or_cond[np.isnan(y_d_HR_lev/H_sea_lev)] = False
# flood_transit_cond[np.isnan(y_d_HR_lev/H_sea_lev)] = False

# # Filter harbor and HR arrays where neither are nan.
# H_sea_lev_nonan = H_sea_lev[~np.isnan(y_d_HR_lev/H_sea_lev)]
# y_d_HR_lev_nonan = y_d_HR_lev[~np.isnan(y_d_HR_lev/H_sea_lev)]

# # Flood Free Sluice Condition
# flood_free_cond[~np.isnan(y_d_HR_lev/H_sea_lev)] = (H_sea_lev_nonan > y_d_HR_lev_nonan) & (y_d_HR_lev_nonan/H_sea_lev_nonan < (2/3))
# # Flood Submerged Orifice Condition
# flood_submer_or_cond[~np.isnan(y_d_HR_lev/H_sea_lev)] = (H_sea_lev_nonan > y_d_HR_lev_nonan) & (y_d_HR_lev_nonan/H_sea_lev_nonan > 0.8)
# # Flood Transitional Condition
# flood_transit_cond[~np.isnan(y_d_HR_lev/H_sea_lev)] = (H_sea_lev_nonan > y_d_HR_lev_nonan) & (y_d_HR_lev_nonan/H_sea_lev_nonan > (2/3)) & (y_d_HR_lev_nonan/H_sea_lev_nonan < 0.8)

# With RuntimeWarnings...
flood_free_cond = (H_sea_lev > y_d_HR_lev) & (y_d_HR_lev/H_sea_lev < (2/3))
flood_submer_or_cond = (H_sea_lev > y_d_HR_lev) & (y_d_HR_lev/H_sea_lev > 0.8)
flood_transit_cond = (H_sea_lev > y_d_HR_lev) & (y_d_HR_lev/H_sea_lev > (2/3)) & (y_d_HR_lev/H_sea_lev < 0.8)

#%% Analytical Estimation of Discharge Through Dike Using Water Levels, My Analysis (all SI)

"""
Test: Supercritical Broad-crested Weir/Free Sluice, Transitional, Subcritical Broad-crested Weir/Submerged Orifice
"""

# If sea level is greater than HR level -> Negative Flow (Flood Tide, Flap Gates Closed)
# Flood = discharge back into estuary (Sluice only! Assume no leaks)
# Sluice gate will always be submerged with flood condition

# Free Sluice Flow
HLsluice = HLsluice_max*(1-0.5*(y_d_HR_lev+H_sea_lev)/Dsluice_HL) # Equation from WHG Report - my own application
# This is from Free Flow Sluice-Gate C_d by Prabhata K. Swamee, for comparison against C_d_flood_free
# C_Swamee = flood_free_cond*0.611*((H_sea_lev-y_d_HR_lev)/(H_sea_lev+15*y_d_HR_lev))**0.072 
Q_flood_free = flood_free_cond*(-C_d_flood_free_mean*A_sluice_open*np.sqrt(2*grav*(H_sea_lev-HLsluice)))

# Submerged Orifice Flow
Q_flood_submer_or = flood_submer_or_cond*(-C_d_flood_submer_or_mean*A_sluice_open*np.sqrt(2*grav*(H_sea_lev-y_d_HR_lev)))

# Transitional Flow
Q_flood_transit = flood_transit_cond*(-C_d_flood_transit_mean*A_sluice_open*np.sqrt(2*grav*3*(H_sea_lev-y_d_HR_lev)))

# If sea level is less than HR level -> Positive Flow (Ebb Tide, Flap Gates Open)

# Center Flap Gate Calculations
A_center_flap_HRside = y_d_HR_lev*L_flaps_in
A_center_flap_oceanside = H_sea_lev*L_flaps_out

# Using SciPy fsolve
# Changing angle of flap gate based on moment
theta_ebb_flap_deg = np.empty(len(pred_to_2100_dtHRocean_df))
for i in range(len(pred_to_2100_dtHRocean_df)): # Optimize angle for all water level pairs.
    def f(theta): 
        """
        Equation for moment around hinge on flap gate.

        Parameters
        ----------
        theta : TYPE
            Angle of gate from initial.

        Returns
        -------
        TYPE
            Sum of moments around hinge.

        """
        return -W_gate*np.sin(theta+angle_init_flaps)*h_gate/dens_seawater/grav - L_flaps_out*(h_gate**2*
                                      np.cos(theta+angle_init_flaps)**2 - 2*h_gate*d_hinge_to_H[i]*np.cos(theta+angle_init_flaps) + d_hinge_to_H[i]**2/
                                      np.cos(theta+angle_init_flaps))*(h_gate-(1/3)*(h_gate-d_hinge_to_H[i]/
                                              np.cos(theta+angle_init_flaps))) + L_flaps_in*(h_gate**2*np.cos(theta+
                                                      angle_init_flaps)**2-2*h_gate*d_hinge_to_y_d[i]*np.cos(theta+angle_init_flaps) + d_hinge_to_y_d[i]**2/
                                              np.cos(theta+angle_init_flaps))*(h_gate-(1/3)*(h_gate - d_hinge_to_y_d[i]/
                                                                                             np.cos(theta+angle_init_flaps)))                                  
    root = float(fsolve(f, 0)) # use root finder to find angle closest to zero
    theta_ebb_flap_deg[i] = np.rad2deg(root)

# Changing conditions for flaps and sluice during ebb tide.

# Ebb Flap Supercritical Weir Condition
ebb_flap_supcrit_weir_cond = (H_sea_lev < y_d_HR_lev) & (H_sea_lev/y_d_HR_lev < (2/3)) & (theta_ebb_flap_deg > 0)
# Ebb Sluice Supercritical Weir Condtion
ebb_sluice_supcrit_weir_cond = (H_sea_lev < y_d_HR_lev) & (H_sea_lev/y_d_HR_lev < (2/3)) & (y_d_HR_lev < y_sluice_open)
# Ebb Sluice Free Sluice Condition
ebb_sluice_free_cond = (H_sea_lev < y_d_HR_lev) & (H_sea_lev/y_d_HR_lev < (2/3)) & (y_d_HR_lev > y_sluice_open)
# Ebb Flap Subcritical Weir Condition
ebb_flap_subcrit_weir_cond = (H_sea_lev < y_d_HR_lev) & (H_sea_lev/y_d_HR_lev > (2/3)) & (theta_ebb_flap_deg > 0)
# Ebb Sluice Subcritical Weir Condtion
ebb_sluice_subcrit_weir_cond = (H_sea_lev < y_d_HR_lev) & (H_sea_lev/y_d_HR_lev > (2/3)) & (y_d_HR_lev < y_sluice_open)
# Ebb Sluice Submerged Orifice Condtion
ebb_sluice_submer_or_cond = (H_sea_lev < y_d_HR_lev) & (H_sea_lev/y_d_HR_lev > (2/3)) & (y_d_HR_lev > y_sluice_open) & (H_sea_lev/y_d_HR_lev > 0.8)
# Ebb Sluice Transitional Condition
ebb_sluice_transit_cond = (H_sea_lev < y_d_HR_lev) & (H_sea_lev/y_d_HR_lev > (2/3)) & (y_d_HR_lev > y_sluice_open) & (H_sea_lev/y_d_HR_lev < 0.8)

"""
Test: Supercritical/Free Sluice, Transitional, Subcritical/Submerged Orifice
"""
# Ebb = flow into harbor

# Actual head loss as function of maximum head loss
HL = HL_max*(1-0.5*(y_d_HR_lev+H_sea_lev)/D_HL)

# Supercritical BC weir/free sluice
Q_ebb_flap_supcrit_weir = ebb_flap_supcrit_weir_cond*(C_d_ebb_flap_supcrit_weir_mean*(2/3)*(y_d_HR_lev+HL)*L_flaps_in*np.sqrt((2/3)*grav*(y_d_HR_lev+HL)))
        
# Supercritical Broad-crested Weir Flow
Q_ebb_supcrit_weir = ebb_sluice_supcrit_weir_cond*(C_d_ebb_supcrit_weir_mean*(2/3)*L_sluice_culv*y_d_HR_lev*np.sqrt((2/3)*grav*y_d_HR_lev))

# Free Sluice Flow
# This is from Free Flow Sluice-Gate C_d by Prabhata K. Swamee, for comparison against C_d_ebb_free
# C_Swamee = ebb_sluice_free_cond*0.611*((y_d_HR_lev-H_sea_lev)/(y_d_HR_lev+15*H_sea_lev))**0.072
Q_ebb_free = ebb_sluice_free_cond*(C_d_ebb_free_mean*A_sluice_open*np.sqrt(2*grav*y_d_HR_lev))

# Subcritical BC weir/submerged orifice
# Use area of water surface on harbor side of flap or HR side?
Q_ebb_flap_subcrit_weir = ebb_flap_subcrit_weir_cond*(C_d_ebb_flap_subcrit_weir_mean*A_center_flap_oceanside*np.sqrt(2*grav*((y_d_HR_lev+HL)-H_sea_lev)))

# Subcritical Broad-crested Weir Flow
Q_ebb_subcrit_weir = ebb_sluice_subcrit_weir_cond*(C_d_ebb_subcrit_weir_mean*L_sluice_culv*H_sea_lev*np.sqrt(2*grav*(y_d_HR_lev-H_sea_lev)))

# Submerged Orifice Flow
Q_ebb_submer_or = ebb_sluice_submer_or_cond*(C_d_ebb_submer_or_mean*A_sluice_open*np.sqrt(2*grav*(y_d_HR_lev-H_sea_lev)))

# Transitional Flow
Q_ebb_transit = ebb_sluice_transit_cond*(C_d_ebb_transit_mean*A_sluice_open*np.sqrt(2*grav*3*(y_d_HR_lev-H_sea_lev)))

"""
Coefficients from Swamee Paper for comparison against C_d_flood_free and C_d_ebb_free
"""
# C_Swamee_mean = np.nanmean(C_Swamee)
# C_Swamee_std = np.nanstd(C_Swamee) 

"""
Total Flow
"""
# Add Q to this array (Add at each index the different culvert Qs)
Q_dike_sluice_calc_flood = np.nansum((Q_flood_free,Q_flood_transit,Q_flood_submer_or),axis=0)
Q_dike_sluice_calc_ebb = np.nansum((Q_ebb_free,Q_ebb_transit,Q_ebb_submer_or),axis=0)
Q_dike_sluice_weir_calc_ebb = np.nansum((Q_ebb_subcrit_weir,Q_ebb_supcrit_weir),axis=0)

# Total flow through sluice and flap
Q_dike_sluice_calc = np.nansum((Q_dike_sluice_calc_flood,Q_dike_sluice_calc_ebb,Q_dike_sluice_weir_calc_ebb),axis=0)
Q_dike_centerflap_calc = np.nansum((Q_ebb_flap_subcrit_weir,Q_ebb_flap_supcrit_weir),axis=0)
# Left Flap Gate Has Same Conditions as Center (smaller culvert, but same gate size)
Q_dike_leftflap_calc = Q_dike_centerflap_calc.copy()

Q_total = np.nansum((Q_dike_leftflap_calc,Q_dike_centerflap_calc,Q_dike_sluice_calc),axis=0)

Q_total[Q_total==0] = np.nan

# Total flow through sluice and through flap.
flow_sluice_culv = np.nansum((Q_ebb_free,Q_ebb_transit,Q_ebb_submer_or,Q_ebb_supcrit_weir,Q_ebb_subcrit_weir),axis=0)
flow_flap_culv = np.nansum((Q_ebb_flap_supcrit_weir,Q_ebb_flap_subcrit_weir),axis=0)

# Fraction of flow through each culvert
flow_frac_sluice_culv = flow_sluice_culv/(flow_sluice_culv+2*flow_flap_culv)
flow_frac_sluice_culv[Q_total<0] = 1
flow_frac_center_culv = flow_flap_culv/(flow_sluice_culv+2*flow_flap_culv)
flow_frac_center_culv[Q_total<0] = 0
flow_frac_left_culv = flow_frac_center_culv # Assume same amount of flow through both flap gates

#%% Save Discharge Output
Q_dike_list = np.vstack((Q_total, Q_dike_sluice_calc, Q_dike_centerflap_calc, flow_frac_sluice_culv, flow_frac_center_culv)).T
Q_dike_df = pd.DataFrame(data=Q_dike_list, columns=["Q Total","Q Sluice","Q Flap","Q_frac Sluice","Q_frac Flap"])
Q_dike_df.insert(0,"datetime",pred_to_2100_dtHRocean_df["datetime"])

# Q_dike_df.to_csv(os.path.join(data_dir, 'General Dike Data', 'Dike_Discharge_Calc_HourlyPred_111946_12312100.csv'), index = False)

#%% Load Q
Q_dike_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","Dike_Discharge_Calc_HourlyPred_111946_12312100.csv"))
data_cols = Q_dike_df.columns.drop("datetime")
Q_dike_df[data_cols] = Q_dike_df[data_cols].apply(pd.to_numeric, errors='coerce')
Q_dike_df["datetime"] = pd.to_datetime(Q_dike_df["datetime"])

Q_total = np.array(Q_dike_df["Q Total"])

#%% Q Plots

pred_to_2100_dikelevdisch_df = pred_to_2100_dtHRocean_df.copy()
pred_to_2100_dikelevdisch_df["Discharge, cms, Predicted"] = Q_total

disch_calc_pred_all_df = pd.merge(HR_all_resam_1hr_df,pred_to_2100_dikelevdisch_df)
disch_calc_pred_all_df = disch_calc_pred_all_df[['datetime','Discharge, cms','Discharge, Dike Calc, cms','Discharge, cms, Predicted']]
index_calcstart = disch_calc_pred_all_df['Discharge, Dike Calc, cms'].first_valid_index()
index_measend = disch_calc_pred_all_df['Discharge, cms'].last_valid_index()

disch_meas_calc_pred_overlap_df = disch_calc_pred_all_df.iloc[index_calcstart:index_measend]
disch_meas_calc_pred_overlap_df.reset_index(drop=True, inplace=True)

"""
Plots
"""
# Overlap of Measured, Calculated with Measured, Calculated with Predicted
plt.figure()
plt.plot(mdates.date2num(disch_meas_calc_pred_overlap_df["datetime"]), disch_meas_calc_pred_overlap_df["Discharge, cms"], color='Turquoise', marker="1", label = 'Measured Discharge')
plt.plot(mdates.date2num(disch_meas_calc_pred_overlap_df["datetime"]), disch_meas_calc_pred_overlap_df["Discharge, Dike Calc, cms"], color='Magenta', marker="2", label = 'Calculated Discharge with Measured Levels')
plt.plot(mdates.date2num(disch_meas_calc_pred_overlap_df["datetime"]), disch_meas_calc_pred_overlap_df["Discharge, cms, Predicted"], color='Olive', marker="3", label = 'Calculated Discharge with Predicted Levels')

# Show X-axis major tick marks as dates
def DateAxisFmt(yax_label):
    loc = mdates.AutoDateLocator()
    plt.gca().xaxis.set_major_locator(loc)
    plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date', fontsize=22)
    plt.ylabel(yax_label, fontsize=22)

ylabel_disch = r'Discharge $\left[\frac{m^3}{s}\right]$'
DateAxisFmt(ylabel_disch)
plt.ylim(-9.5,7)
plt.xlabel(r'Date, 2017 $\left[MM-DD\/\/HH\right]$', fontsize=22)
plt.legend(loc='lower center', fontsize=22)

# Histograms of Residuals
# Observed-Modeled (with Observed)
disch_obsmod_residuals = pd.DataFrame(disch_meas_calc_pred_overlap_df["Discharge, cms"]-disch_meas_calc_pred_overlap_df["Discharge, Dike Calc, cms"])
disch_obsmod_residuals.rename(columns={0:'Residuals, cms, Discharge Calc'}, inplace=True)
ax = disch_obsmod_residuals.plot.hist(bins=20)
plt.xlabel(r'Residuals of Discharge Calculated with Observed Levels $\left[\frac{m^3}{s}\right]$', fontsize=22)
plt.ylabel('Count', fontsize=22)
plt.legend()
ax.get_legend().remove()

# Observed-Modeled (With Modeled)
disch_modmod_residuals = pd.DataFrame(disch_meas_calc_pred_overlap_df["Discharge, cms"]-disch_meas_calc_pred_overlap_df["Discharge, cms, Predicted"])
disch_modmod_residuals.rename(columns={0:'Residuals, cms, Discharge Pred'}, inplace=True)
ax = disch_modmod_residuals.plot.hist(bins=20)
plt.xlabel(r'Residuals of Discharge Calculated with Predicted Levels $\left[\frac{m^3}{s}\right]$', fontsize=22)
plt.ylabel('Count', fontsize=22)
plt.legend()
ax.get_legend().remove()

# Calculated with Measured, Calculated with Predicted
index_calcend = disch_calc_pred_all_df['Discharge, Dike Calc, cms'].last_valid_index()
disch_calc_pred_overlap_df = disch_calc_pred_all_df.iloc[index_calcstart:index_calcend]
disch_calc_pred_overlap_df.reset_index(drop=True, inplace=True)

# Rolling Average
disch_calc_pred_overlap_df["Discharge, Dike Calc, cms, Rolling Average"] = disch_calc_pred_overlap_df["Discharge, Dike Calc, cms"].rolling(window=4383, min_periods=2000).mean() # 709 hrs/Julian month 
disch_calc_pred_overlap_df["Discharge, cms, Predicted, Rolling Average"] = disch_calc_pred_overlap_df["Discharge, cms, Predicted"].rolling(window=4383, min_periods=2000).mean()

plt.figure()
plt.plot(mdates.date2num(disch_calc_pred_overlap_df["datetime"]), disch_calc_pred_overlap_df["Discharge, Dike Calc, cms, Rolling Average"], color='Turquoise', marker="1", label = 'Calculated Discharge with Measured Levels')
plt.plot(mdates.date2num(disch_calc_pred_overlap_df["datetime"]), disch_calc_pred_overlap_df["Discharge, cms, Predicted, Rolling Average"], color='Magenta', marker="2", label = 'Calculated Discharge with Predicted Levels')

ylabel_disch = r'Rolling Average (6 Month) of Discharge $\left[\frac{m^3}{s}\right]$'
DateAxisFmt(ylabel_disch)
plt.xlabel(r'Date $\left[YYYY-MM\right]$', fontsize=22)
plt.legend(loc='best', bbox_to_anchor=(0.18,0.8), fontsize=22)

#%% Rolling Calculation of Moving Average

pred_to_2100_dikelevdisch_df["Ocean side, Predicted, Rolling Average"] = pred_to_2100_dikelevdisch_df["Gage height, m, Ocean side, Predicted"].rolling(window=8766, min_periods=8000).mean() # 8000 hours removed for convergence
pred_to_2100_dikelevdisch_df["HR side, Predicted, Rolling Average"] = pred_to_2100_dikelevdisch_df["Gage height, m, HR side, Predicted"].rolling(window=8766, min_periods=8000).mean() # 8766 hrs/Julian year
pred_to_2100_dikelevdisch_df["Discharge, cms, Predicted, Rolling Average"] = pred_to_2100_dikelevdisch_df["Discharge, cms, Predicted"].rolling(window=8766, min_periods=8000).mean() # 709 hrs/Julian month 

# Predicted
plt.scatter(mdates.date2num(pred_to_2100_dtHRocean_df["datetime"]), pred_to_2100_dikelevdisch_df["Ocean side, Predicted, Rolling Average"], marker='.', label = 'WF Harbor Near-Dike Levels')
plt.scatter(mdates.date2num(pred_to_2100_dtHRocean_df["datetime"]), pred_to_2100_dikelevdisch_df["HR side, Predicted, Rolling Average"], marker='.', label = 'HR Near-Dike Levels')
plt.scatter(mdates.date2num(pred_to_2100_dtHRocean_df["datetime"]), pred_to_2100_dikelevdisch_df["Discharge, cms, Predicted, Rolling Average"], marker='.', label = 'Calculated Discharge')
# plt.ylim(-1.5,1)

ylabel_disch = 'Rolling Average (Annual), Predictions, \n' r'Elevation $\left[m\/\/NAVD88\right]$, Discharge $\left[\frac{m^3}{s}\right]$'
DateAxisFmt(ylabel_disch)
plt.legend(loc='best', fontsize=22)

#%% Calculation for any number of flaps and sluices with these geometries

num_sluices = 1
num_flaps = 2

Q_total_anynum = np.nansum((num_flaps*Q_dike_centerflap_calc,num_sluices*Q_dike_sluice_calc),axis=0)

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

#%% Max and min vals for discharge, through dike

dike_discharge_calc = pd.DataFrame(Q_total)[0]
dike_discharge_meas = HR_all_resam_1hr_df["Discharge, cms"]
dates = HR_all_resam_1hr_df["datetime"]

min_dates_dikeQ_calc, y_dikeQ_mins_calc, max_dates_dikeQ_calc, y_dikeQ_maxes_calc = MaxMinLevels(dates, dike_discharge_calc)
min_dates_dikeQ_meas, y_dikeQ_mins_meas, max_dates_dikeQ_meas, y_dikeQ_maxes_meas = MaxMinLevels(dates, dike_discharge_meas)

pd.merge(Q_dike_df)

HR_dike_Qmeas_resam_df = pd.read_csv(os.path.join(data_dir,"General Dike Data","HR_Meas_Discharge_Data_Resampled.csv"))
HR_dike_Qmeas_resam_df["datetime"] = pd.to_datetime(HR_dike_Qmeas_resam_df["datetime"])

min_dates_dikeQ_5m_meas, y_dikeQ_mins_5m_meas, max_dates_dikeQ_5m_meas, y_dikeQ_maxes_5m_meas = MaxMinLevels(HR_dike_Qmeas_resam_df["datetime"], HR_dike_Qmeas_resam_df["Discharge, cms"])

plt.scatter(mdates.date2num(max_dates_dikeQ_5m_meas), y_dikeQ_maxes_5m_meas, label= 'Ebb, 5 Minute Measured')
plt.scatter(mdates.date2num(min_dates_dikeQ_5m_meas), y_dikeQ_mins_5m_meas, label= 'Flood, 5 Minute Measured')

plt.scatter(mdates.date2num(max_dates_dikeQ_meas), y_dikeQ_maxes_meas, marker='.', label= 'Ebb, Sept. 2017, Hourly Means')
plt.scatter(mdates.date2num(min_dates_dikeQ_meas), y_dikeQ_mins_meas, marker='.', label= 'Flood, Sept. 2017, Hourly Means')

ylabel_disch = r'Maximum Discharge $\left[\frac{m^3}{s}\right]$'
DateAxisFmt(ylabel_disch)
plt.xlabel(r'Date $\left[YYYY-MM\right]$', fontsize=22)
plt.legend(loc='center', fontsize=22)
