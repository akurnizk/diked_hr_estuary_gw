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
from scipy import interpolate
from shapely.geometry import Point
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, curve_fit
from datetime import datetime, time, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

work_dir = os.path.join('E:\Herring Models\Seasonal')
data_dir = os.path.join('E:\Data')
map_dir = r'E:\Maps' # retrieved files from https://viewer.nationalmap.gov/basic/

mean_sea_level = 0.843 # Datum in meters at closest NOAA station (8447435), Chatham, Lydia Cove MA
# https://tidesandcurrents.noaa.gov/datums.html?units=1&epoch=0&id=8447435&name=Chatham%2C+Lydia+Cove&state=MA

import affine
import geopandas as gpd

import statistics
from statistics import mode

cgw_code_dir = 'E:\Python KMB - CGW' # Location of BitBucket folder containing cgw folder
sys.path.insert(0,cgw_code_dir)

from cgw.utils import general_utils as genu
from cgw.utils import feature_utils as shpu
from cgw.utils import raster_utils as rastu

import rasterio
from rasterio import mask
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.io import MemoryFile
from rasterio.enums import Resampling

months_str = calendar.month_name
x_months = np.array(months_str[1:]) # array of the months of the year
x_ss_months = np.append(np.array(['Steady State']), x_months)

#%% To Do

# Compare sea level measurements at Boston, Provincetown, and outside dike.

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

#%% Using rasterio

def xy_from_affine(tform=None,nx=None,ny=None):
    X,Y = np.meshgrid(np.arange(nx)+0.5,np.arange(ny)+0.5)*tform
    return X,Y

#%% Loading in shapefiles and DEM

area_df_HR = gpd.read_file(os.path.join(map_dir,'Herring River_Diked.shp'))

# Making fake raster for model domain
temp_crs = area_df_HR.crs # same as polygon

minx,miny,maxx,maxy = area_df_HR.bounds.values.T
leftbound,topbound = minx.min(),maxy.max() # top left corner of model domain
xres, yres = 1, 1 # grid resolution

"""
Chequesset Region
"""

# Using SHPU to find rotation angle
shp_fname = os.path.join(map_dir,'Chequesset_Model_Area_UTM.shp')
cell_spacing = 1. # model grid cell spacing in meters
# Define inputs for shp_to_grid function
shp_to_grid_dict = {'shp':shp_fname,'cell_spacing':cell_spacing}
grid_outputs = shpu.shp_to_grid(**shp_to_grid_dict)
# Pop out all of the outputs into individual variables
[X_nodes,Y_nodes],model_polygon,[out_proj,[xshift,yshift],min_angle] = grid_outputs
# min angle found...
grid_transform = [out_proj,[xshift,yshift],min_angle] # make transform list
# Can calculate cell centers (where heads are calculated), in different coordinates
cc,cc_proj,cc_ll = shpu.nodes_to_cc([X_nodes,Y_nodes],grid_transform)

def make_geodata(X=None,Y=None,dx=None,dy=None,rot_xy=None):
    return [X[0,0]-np.cos(-rot_xy)*dx/2.+np.sin(-rot_xy)*dx/2.,
                   np.cos(-rot_xy)*dx,
                   -np.sin(-rot_xy)*dx,
                   Y[0,0]-np.cos(-rot_xy)*dy/2.-np.sin(-rot_xy)*dy/2.,
                   np.sin(-rot_xy)*dy,
                   np.cos(-rot_xy)*dy]

geodata = make_geodata(X=cc_proj[0],Y=cc_proj[1],dx=xres,dy=-yres,rot_xy=-min_angle)

# model_transform = affine.Affine(xres,0.0,leftbound,0.0,-yres,topbound)
model_transform = affine.Affine.from_gdal(*geodata)

model_width = cc[0].shape[1] 
model_height = cc[0].shape[0]
nodata_val = -9999

X,Y = xy_from_affine(model_transform,model_width,model_height)

# For writing to file
model_profile = {'driver':'GTiff','crs':temp_crs,'count':1,
                'height': model_height,'dtype':rasterio.float64,
                'width': model_width,'nodata':nodata_val,'compress':'lzw',
                'transform':model_transform}
# For loading other rasters
vrt_options = {'resampling': Resampling.bilinear,
                'transform': model_transform,
                'crs':temp_crs,
                'height': model_height,
                'width': model_width,'nodata':nodata_val}

# Load lidar dem to model domain - does this exist?
ascifname_lidar = os.path.join(map_dir,'hr_2011las2rast_clip_asci\hr_2011las2rast_clip_asci.txt')
with rasterio.open(ascifname_lidar) as src:
    with WarpedVRT(src,**vrt_options) as vrt:
        dem_model_lidar = vrt.read()[0] # [0] selects first band
        dem_model_lidar[dem_model_lidar==vrt.nodata] = np.nan
        # Should be dem_model.shape = [model_height,model_width]

# Load unfilled 1mx1m dem to model domain
demfname_cheq = os.path.join(map_dir,'USGS_NED_Chequesset_one_meter_Combined.tif')
with rasterio.open(demfname_cheq) as src:
    with WarpedVRT(src,**vrt_options) as vrt:
        dem_model_cheq = vrt.read()[0] # [0] selects first band
        dem_model_cheq[dem_model_cheq==vrt.nodata] = np.nan
        # Should be dem_model.shape = [model_height,model_width]

# Load bathymetry info from WHG
matcont_hr = loadmat(os.path.join(map_dir, 'GUI_version2\GUI_info_new.mat'))
xl,yl,belv = matcont_hr['xl'], matcont_hr['yl'], matcont_hr['belv']
        
dem_X,dem_Y,dem_da = rastu.load_geotif(ascifname_lidar) # dem_da is an xarray data array

#%% Example of making a MODFLOW-like grid from a shapefile

# Create masked areas for model domain
mask_array_HR = np.zeros([model_height,model_width])

# Make temporary raster in memory
with MemoryFile() as memfile:
    with memfile.open(**model_profile) as dataset:
        tempdataset = np.ones([1,model_height,model_width]) # make array of all one value
        dataset.write(tempdataset)
        for igeom,feature in enumerate(area_df_HR.geometry.values):      # loop through features in HR shp
            mask_rast_HR,tform_HR = mask.mask(dataset,[feature],crop=False,all_touched=True)
            mask_rast_HR = mask_rast_HR.squeeze()
            mask_array_HR[mask_rast_HR==1] = igeom+1 # start at 1
    
#%% Land Surface Elevation and HR Plots

# HR Mask
fig,ax = genu.plt.subplots(1,2)
ax[0].set_xlabel('column #')
ax[0].set_ylabel('row #')
genu.quick_plot(np.ma.masked_array(dem_model_lidar,mask_array_HR!=1),vmin=-2,vmax=5,ax=ax[0])
#genu.quick_plot(mask_array.astype(int),ax=ax[0]) # in row, column space
c1=ax[1].pcolormesh(cc[0],cc[1],mask_array_HR.astype(int)) # in model coordinates
genu.plt.colorbar(c1,ax=ax[1],orientation='horizontal')
fig.gca().set_aspect('equal', adjustable='box')
ax[1].set_xlabel('X [m]')
ax[1].set_ylabel('Y [m]')

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
HL_max = 0.9 # maximum flap headloss, meters
HLsluice_max = 1.0 # maximum sluice flood headgain, meters
D_HL = 0.4 # flap headloss parameter, meters
Dsluice_HL = 1.0 # sluice flood headloss parameter, meters

# Initialize Discharge Arrays and set to nans
Q_flood_free = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_flood_transit = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_flood_submer_or = np.zeros_like(HR_dike_lev_disch_m["datenum"])
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
Q_ebb_free[:] = np.nan
Q_ebb_transit[:] = np.nan
Q_ebb_submer_or[:] = np.nan
Q_ebb_subcrit_weir[:] = np.nan
Q_ebb_supcrit_weir[:] = np.nan
Q_ebb_flap_subcrit_weir[:] = np.nan
Q_ebb_flap_supcrit_weir[:] = np.nan

Q_ebb_flap_supcrit_weir_act = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_supcrit_weir_act = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_free_act = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_flap_subcrit_weir_act = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_subcrit_weir_act = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_submer_or_act = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_transit_act = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_ebb_flap_supcrit_weir_act[:] = np.nan
Q_ebb_supcrit_weir_act[:] = np.nan
Q_ebb_free_act[:] = np.nan
Q_ebb_flap_subcrit_weir_act[:] = np.nan
Q_ebb_subcrit_weir_act[:] = np.nan
Q_ebb_submer_or_act[:] = np.nan
Q_ebb_transit_act[:] = np.nan

# Initialize Discharge Coefficient Arrays and set to nans
C_d_ebb_free = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_ebb_transit = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_ebb_submer_or = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_ebb_subcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_ebb_supcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_ebb_flap_subcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
C_d_ebb_flap_supcrit_weir = np.zeros_like(HR_dike_lev_disch_m["datenum"])
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

C_Swamee = np.zeros_like(HR_dike_lev_disch_m["datenum"]) # This is from Free Flow Sluice-Gate C_d by Prabhata K. Swamee

"""
Ebb C_d means and stdevs.
"""
C_d_ebb_free_mean = 0.8
C_d_ebb_transit_mean = 0.6
C_d_ebb_submer_or_mean = 0.9
C_d_ebb_subcrit_weir_mean = 0.7
C_d_ebb_supcrit_weir_mean = 1.0
C_d_ebb_flap_subcrit_weir_mean = 0.7
C_d_ebb_flap_supcrit_weir_mean = 0.85
"""
Flood C_d means and stdevs.
"""
C_d_flood_free_mean = 0.75
C_d_flood_transit_mean = 0.6
C_d_flood_submer_or_mean = 1.1

"""
Using arrays
"""
# for i in range(len(HR_dike_lev_disch_m)): # (FOR WHEN IT WAS A LOOP - NEED [i] AFTER ARRAYS IN LOOP)
# Levels relative to culvert invert at sluice/flaps.
H_sea_lev = np.array(HR_dike_lev_disch_m["Gage height, m, Ocean side"] - inv_el_open)
y_d_HR_lev = np.array(HR_dike_lev_disch_m["Gage height, m, HR side"] - inv_el_open)
# Through-dike discharge to be calculated
Q_disch_arr = np.zeros_like(HR_dike_lev_disch_m["datenum"])
Q_disch_arr[:] = np.nan
# Vertical distances from flap gate hinge to water levels.
d_hinge_to_H = np.array(hinge_el_open - HR_dike_lev_disch_m["Gage height, m, Ocean side"])
d_hinge_to_y_d = np.array(hinge_el_open - HR_dike_lev_disch_m["Gage height, m, HR side"])

# Flood Free Sluice Condition
flood_free_cond = (H_sea_lev > y_d_HR_lev) & (y_d_HR_lev/H_sea_lev < (2/3))
# Flood Submerged Orifice Condition
flood_submer_or_cond = (H_sea_lev > y_d_HR_lev) & (y_d_HR_lev/H_sea_lev > 0.8)
# Flood Transitional Condition
flood_transit_cond = (H_sea_lev > y_d_HR_lev) & (y_d_HR_lev/H_sea_lev > (2/3)) & (y_d_HR_lev/H_sea_lev < 0.8)

# if (H_sea_lev > y_d_HR_lev): # If sea level is greater than HR level -> Negative Flow (Flood Tide, Flap Gates Closed)
"""
Test: Supercritical Broad-crested Weir/Free Sluice, Transitional, Subcritical Broad-crested Weir/Submerged Orifice
"""
    # if (y_d_HR_lev/H_sea_lev < (2/3)): # Free Sluice Flow
HLsluice = HLsluice_max*(1-0.5*(y_d_HR_lev+H_sea_lev)/Dsluice_HL)
C_Swamee = 0.611*((H_sea_lev-y_d_HR_lev)/(H_sea_lev+15*y_d_HR_lev))**0.072
Q_flood_free = flood_free_cond*(-C_d_flood_free_mean*A_sluice_open*np.sqrt(2*grav*(H_sea_lev-HLsluice)))
    # else:
    #     if (y_d_HR_lev/H_sea_lev > 0.8): # Submerged Orifice Flow
Q_flood_submer_or = flood_submer_or_cond*(-C_d_flood_submer_or_mean*A_sluice_open*np.sqrt(2*grav*(H_sea_lev-y_d_HR_lev)))
        # else: # Transitional Flow
Q_flood_transit = flood_transit_cond*(-C_d_flood_transit_mean*A_sluice_open*np.sqrt(2*grav*3*(H_sea_lev-y_d_HR_lev)))

# else: # If sea level is less than HR level -> Positive Flow (Ebb Tide, Flap Gates Open)
# Center Flap Gate Calculations
A_center_flap_HRside = y_d_HR_lev*L_flaps_in
A_center_flap_oceanside = H_sea_lev*L_flaps_out # Should L change?
# Using SciPy fsolve
for i in range(len(HR_dike_lev_disch_m)):
    def f(theta): 
        return -W_gate*np.sin(theta+angle_init_flaps)*h_gate/dens_seawater/grav - L_flaps_out*(h_gate**2*
                                      np.cos(theta+angle_init_flaps)**2 - 2*h_gate*d_hinge_to_H[i]*np.cos(theta+angle_init_flaps) + d_hinge_to_H[i]**2/
                                      np.cos(theta+angle_init_flaps))*(h_gate-(1/3)*(h_gate-d_hinge_to_H[i]/
                                              np.cos(theta+angle_init_flaps))) + L_flaps_in*(h_gate**2*np.cos(theta+
                                                      angle_init_flaps)**2-2*h_gate*d_hinge_to_y_d[i]*np.cos(theta+angle_init_flaps) + d_hinge_to_y_d[i]**2/
                                              np.cos(theta+angle_init_flaps))*(h_gate-(1/3)*(h_gate - d_hinge_to_y_d[i]/
                                                                                             np.cos(theta+angle_init_flaps)))                                  
    root = float(fsolve(f, 0)) # use root finder to find angle closest to zero
    theta_ebb_flap_deg[i] = np.rad2deg(root)
# Flow fractions of total measured discharge through each culvert (NEED TO OPTIMIZE)
    
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
    # if (H_sea_lev/y_d_HR_lev < (2/3)): # supercritical BC weir/free sluice - OPTIMIZE COEFFIENT BETWEEN FLAPS AND SLUICE!
    #     if (root > 0):
HL = HL_max*(1-0.5*(y_d_HR_lev+H_sea_lev)/D_HL)
Q_ebb_flap_supcrit_weir = ebb_flap_supcrit_weir_cond*(C_d_ebb_flap_supcrit_weir_mean*(2/3)*(y_d_HR_lev+HL)*L_flaps_in*np.sqrt((2/3)*grav*(y_d_HR_lev+HL)))
        # if (y_d_HR_lev < y_sluice_open): # Supercritical Broad-crested Weir Flow
Q_ebb_supcrit_weir = ebb_sluice_supcrit_weir_cond*(C_d_ebb_supcrit_weir_mean*(2/3)*L_sluice_culv*y_d_HR_lev*np.sqrt((2/3)*grav*y_d_HR_lev))
        # else: # Free Sluice Flow
C_Swamee = 0.611*((y_d_HR_lev-H_sea_lev)/(y_d_HR_lev+15*H_sea_lev))**0.072
Q_ebb_free = ebb_sluice_free_cond*(C_d_ebb_free_mean*A_sluice_open*np.sqrt(2*grav*y_d_HR_lev))
    # else: #  subcritical BC weir/submerged orifice - OPTIMIZE COEFFIENT BETWEEN FLAPS AND SLUICE!
    #     if (root > 0):
# HL = HL_max*(1-0.5*(y_d_HR_lev+H_sea_lev)/D_HL)
Q_ebb_flap_subcrit_weir = ebb_flap_subcrit_weir_cond*(C_d_ebb_flap_subcrit_weir_mean*A_center_flap_oceanside*np.sqrt(2*grav*((y_d_HR_lev+HL)-H_sea_lev)))
        # if (y_d_HR_lev < y_sluice_open): # Subcritical Broad-crested Weir Flow
Q_ebb_subcrit_weir = ebb_sluice_subcrit_weir_cond*(C_d_ebb_subcrit_weir_mean*L_sluice_culv*H_sea_lev*np.sqrt(2*grav*(y_d_HR_lev-H_sea_lev)))
        # elif (H_sea_lev/y_d_HR_lev > 0.8): # Submerged Orifice Flow
Q_ebb_submer_or = ebb_sluice_submer_or_cond*(C_d_ebb_submer_or_mean*A_sluice_open*np.sqrt(2*grav*(y_d_HR_lev-H_sea_lev)))
        # else: # Transitional Flow
Q_ebb_transit = ebb_sluice_transit_cond*(C_d_ebb_transit_mean*A_sluice_open*np.sqrt(2*grav*3*(y_d_HR_lev-H_sea_lev)))

flow_sluice_culv = np.nansum((Q_ebb_free,Q_ebb_transit,Q_ebb_submer_or,Q_ebb_supcrit_weir,Q_ebb_subcrit_weir),axis=0)
flow_flap_culv = np.nansum((Q_ebb_flap_supcrit_weir,Q_ebb_flap_subcrit_weir),axis=0)
flow_frac_sluice_culv = flow_sluice_culv/(flow_sluice_culv+2*flow_flap_culv)
flow_frac_center_culv = flow_flap_culv/(flow_sluice_culv+2*flow_flap_culv)
flow_frac_left_culv = flow_frac_center_culv

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
# Add Q to this array (Add at each index the different culvert Qs)
Q_dike_sluice_calc_flood = np.nansum((Q_flood_free,Q_flood_transit,Q_flood_submer_or),axis=0)
Q_dike_sluice_calc_ebb = np.nansum((Q_ebb_free,Q_ebb_transit,Q_ebb_submer_or),axis=0)
Q_dike_sluice_weir_calc_ebb = np.nansum((Q_ebb_subcrit_weir,Q_ebb_supcrit_weir),axis=0)

Q_dike_sluice_calc = np.nansum((Q_dike_sluice_calc_flood,Q_dike_sluice_calc_ebb,Q_dike_sluice_weir_calc_ebb),axis=0)

Q_dike_centerflap_calc = np.nansum((Q_ebb_flap_subcrit_weir,Q_ebb_flap_supcrit_weir),axis=0)
    
# Left Flap Gate Has Same Conditions as Center (smaller culvert, but same gate size)
Q_dike_leftflap_calc = Q_dike_centerflap_calc.copy()

Q_total = np.nansum((Q_dike_leftflap_calc,Q_dike_centerflap_calc,Q_dike_sluice_calc),axis=0)

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

#%% Translating Open-Channel Flow Project (WSP Preissmann Implicit (alpha = 1))

range_HRside_avg = 0.766
range_oceanside_avg = 2.535 # note that low tide is not well represented given the river discharge
tide_amp_out = range_oceanside_avg/2
tide_amp_in = range_HRside_avg/2
meanmins_oceanside = -1.02
meanmins_HRside = -0.65

# River mouth depth

def tidal_cycle(time_sec):
        return 1 + tide_amp_in*math.sin(math.pi*time_sec/22350)

tide_times = np.arange(0,89700,300)
tide_heights = []
for x in tide_times:
    tide_heights = np.append(tide_heights,tidal_cycle(x))

# Plot theoretical water level curve
fig, ax = plt.subplots()
ax.plot(tide_times,tide_heights)
ax.set_xlim(0,89400)
ax.xaxis.set_ticks(np.arange(0, 104300, 14900))
ax.set(xlabel='Time (s)', ylabel='Water Depth Outside Dike (m NAVD88)')
ax.grid()

"""
LOAD HR GEOMETRY  # make top of array the upstream-most section
"""
out_x_stacked = np.loadtxt(os.path.join(map_dir, 'HR_XsecLines','HR_xsec_all_xcoords.csv'), delimiter=',')
out_x_stacked = np.flip(out_x_stacked,axis=0)
out_y_stacked = np.loadtxt(os.path.join(map_dir, 'HR_XsecLines','HR_xsec_all_ycoords.csv'), delimiter=',')
out_y_stacked = np.flip(out_y_stacked,axis=0)
elevs_interp = np.loadtxt(os.path.join(map_dir, 'HR_XsecLines','HR_xsec_all_elevs.csv'), delimiter=',')
elevs_interp = np.flip(elevs_interp,axis=0)
intersect_newxy = np.loadtxt(os.path.join(map_dir, 'HR_XsecLines','HR_xsec_all_inscts.csv'), delimiter=',')
intersect_newxy = np.flip(intersect_newxy,axis=0)
min_dist_dx = np.loadtxt(os.path.join(map_dir, 'HR_XsecLines','HR_xsec_all_dx.csv'), delimiter=',')
min_dist_dx = np.flip(min_dist_dx,axis=0)

# !      COMPUTATION OF UNSTEADY, FREE-SURFACE FLOWS BY PREISSMANN IMPLICIT SCHEME IN A TRAPEZOIDAL CHANNEL.
# !      CONSTANT FLOW DEPTH ALONG THE CHANNEL IS SPECIFIED AS INITIAL CONDITION.
# !      TRANSIENT CONDITIONS ARE PRODUCED BY CHANGING DISCHARGES AT UPSTREAM AND DOWNSTREAM ENDS
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
#%% Necessary functions, defined at end of Fortran code
"""
D is depth in deepest part of xsec in each.
"""
# Make matrix of changing depths, discretized.
y_range_start = np.linspace(0,6,100,endpoint=False)
y_range_end = np.linspace(0,6,100,endpoint=False)
y_sample_allxsec = np.empty((40,100))
y_sample_allxsec[0] = y_range_start
y_sample_allxsec[-1] = y_range_end
for j in range(len(y_range_start)):
    y_sample_allxsec[:,j] = np.linspace(y_range_start[j],y_range_end[j],40)

def AR(D, elevs_interp=None, out_x_stacked=None, out_y_stacked=None): 
    """
    Satement function for flow area.
    """
    # return (b0+D*s)*D # original
    wsp = D + np.amin(elevs_interp,axis=1) # add depth to lowest point in channel
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
    return np.array(area_all)

def HR(D, elevs_interp=None, out_x_stacked=None, out_y_stacked=None): # area/w_perim
    """
    Satement function for hydraulic radius.
    """
    # return (b0+D*s)*D/(b0+2*D*np.sqrt(1+s*s)) # original
    wsp = D + np.amin(elevs_interp,axis=1) # add depth to lowest point in channel
    wsp_mask = np.vstack([(elevs_interp[xsec,:] <= wsp[xsec]) for xsec in range(len(elevs_interp))])
    w_perim_all = []
    for xsec in range(len(elevs_interp)):
        w_perim_xsec = []
        for xsec_i in range(len(elevs_interp.T)-1):
            pt_to_pt_x = np.sqrt((out_x_stacked[xsec,xsec_i]-out_x_stacked[xsec,xsec_i+1])**2+(out_y_stacked[xsec,xsec_i]-out_y_stacked[xsec,xsec_i+1])**2)
            pt_to_pt_y = abs(elevs_interp[xsec,xsec_i]-elevs_interp[xsec,xsec_i+1])
            pt_to_pt_hyp = np.sqrt(pt_to_pt_x**2 + pt_to_pt_y**2)
            if (wsp_mask[xsec,xsec_i] != wsp_mask[xsec,xsec_i+1]):
                y_edge = wsp[xsec]-min(elevs_interp[xsec,xsec_i],elevs_interp[xsec,xsec_i+1])
                pt_to_pt_hyp = pt_to_pt_hyp*y_edge/pt_to_pt_y
                wsp_mask[xsec,xsec_i] = True
            w_perim_xsec.append((pt_to_pt_hyp*wsp_mask[xsec,xsec_i]))
        w_perim_all.append((np.nansum(w_perim_xsec)))
    return AR(D,elevs_interp,out_x_stacked,out_y_stacked)/np.array(w_perim_all)

def TOP(D, elevs_interp=None, out_x_stacked=None, out_y_stacked=None):
    """
    Satement function for water top width.
    """
    # return b0+2*D*s
    wsp = D + np.amin(elevs_interp,axis=1) # add depth to lowest point in channel
    wsp_mask = np.vstack([(elevs_interp[xsec,:] <= wsp[xsec]) for xsec in range(len(elevs_interp))])
    top_width_all = []
    for xsec in range(len(elevs_interp)):
        top_width_xsec = []
        for xsec_i in range(len(elevs_interp.T)-1):
            pt_to_pt_x = np.sqrt((out_x_stacked[xsec,xsec_i]-out_x_stacked[xsec,xsec_i+1])**2+(out_y_stacked[xsec,xsec_i]-out_y_stacked[xsec,xsec_i+1])**2)
            pt_to_pt_y = abs(elevs_interp[xsec,xsec_i]-elevs_interp[xsec,xsec_i+1])
            if (wsp_mask[xsec,xsec_i] != wsp_mask[xsec,xsec_i+1]):
                y_edge = wsp[xsec]-min(elevs_interp[xsec,xsec_i],elevs_interp[xsec,xsec_i+1])
                pt_to_pt_x = pt_to_pt_x*y_edge/pt_to_pt_y
                wsp_mask[xsec,xsec_i] = True
            top_width_xsec.append((pt_to_pt_x*wsp_mask[xsec,xsec_i]))
        top_width_all.append((np.nansum(top_width_xsec)))
    return np.array(top_width_all)

def CENTR(D, elevs_interp=None, out_x_stacked=None, out_y_stacked=None):
    """
    Satement function for moment of flow area.
    """
    # return D*D*(b0/2+D*s/3)
    wsp = D + np.amin(elevs_interp,axis=1) # add depth to lowest point in channel
    wsp_mask = np.vstack([(elevs_interp[xsec,:] <= wsp[xsec]) for xsec in range(len(elevs_interp))])
    moment_all = []
    for xsec in range(len(elevs_interp)):
        moment_xsec = []
        for xsec_i in range(len(elevs_interp.T)-1):
            pt_to_pt_x = np.sqrt((out_x_stacked[xsec,xsec_i]-out_x_stacked[xsec,xsec_i+1])**2+(out_y_stacked[xsec,xsec_i]-out_y_stacked[xsec,xsec_i+1])**2)
            pt_to_pt_y = abs(elevs_interp[xsec,xsec_i]-elevs_interp[xsec,xsec_i+1])
            if (wsp_mask[xsec,xsec_i] != wsp_mask[xsec,xsec_i+1]):
                pt_to_pt_y = wsp[xsec]-min(elevs_interp[xsec,xsec_i],elevs_interp[xsec,xsec_i+1])
                pt_to_pt_x = pt_to_pt_x*pt_to_pt_y/abs(elevs_interp[xsec,xsec_i]-elevs_interp[xsec,xsec_i+1])
                wsp_mask[xsec,xsec_i] = True
            y_c_up = 0.5*min(wsp[xsec]-elevs_interp[xsec,xsec_i],wsp[xsec]-elevs_interp[xsec,xsec_i+1])
            y_c_triang = (1/3)*pt_to_pt_y + min(wsp[xsec]-elevs_interp[xsec,xsec_i],wsp[xsec]-elevs_interp[xsec,xsec_i+1])
            area_up = min(wsp[xsec]-elevs_interp[xsec,xsec_i],wsp[xsec]-elevs_interp[xsec,xsec_i+1])*pt_to_pt_x
            area_triang = pt_to_pt_y*pt_to_pt_x/2
            moment_up = y_c_up*area_up
            moment_triang = y_c_triang*area_triang
            moment_xsec.append(((moment_up+moment_triang)*wsp_mask[xsec,xsec_i]))
        moment_all.append((np.nansum(moment_xsec)))
    return np.array(moment_all)

"""
Make matrix of dcendy derivatives to pull from.
"""
cen_matrix = np.empty((40,100))
for j in range(len(y_range_start)):
    cen_matrix[:,j] = CENTR(y_sample_allxsec[:,j],elevs_interp,out_x_stacked,out_y_stacked)
dcendy_matrix = np.diff(cen_matrix)/np.diff(y_sample_allxsec)
y_sample_midpoints = (y_sample_allxsec[:,1:]+y_sample_allxsec[:,:-1])/2

def DCENDY(D, dcendy_matrix=None, y_sample_midpoints=None): # D is Y
    """
    Satement function for derivative of moment of flow area with respect to depth.
    """
    # return D*(b0+D*s)
    dcendy_act = []
    for row in range(len(y_sample_midpoints)):
        interp_func = interp1d(y_sample_midpoints[row],dcendy_matrix[row])
        dcendy_act.append(interp_func([D[row]])[0])
    dcendy_all = np.array(dcendy_act)
    # def norm_func(x,a,b,c): # is this crazy? might want to adjust if actual y values aren't represented here
    #     return a*np.exp(-((x-b)**2)/c)
    # popt, pcov = curve_fit(norm_func, D, cen)
    # cen_smoothed = norm_func(D,*popt)
    # dcendy_all = -2*(D-popt[1])/popt[2]*popt[0]*np.exp(-((D-popt[1])**2)/popt[2]) # derivative of curve fit function
    return dcendy_all

def make_sf(Y,V,cmn_squared):
    """
    Satement function for friction slope.
    """
    return abs(V)*V*cmn_squared/HR(Y,elevs_interp,out_x_stacked,out_y_stacked)**1.333

def make_C2(Y,V,ARi,ARiP1,cmn_squared,s0,grav):
    sf1 = make_sf(Y,V,cmn_squared)[:-1]
    sf2 = make_sf(Y,V,cmn_squared)[1:]
    term1 = -dt*(1-alpha)*(grav*ARiP1*(s0-sf2)+grav*ARi*(s0-sf1))
    term2 = -(V[:-1]*ARi+V[1:]*ARiP1)
    term3 = dtx2*(1-alpha)*((V[1:]**2)*ARiP1 + \
                  grav*CENTR(Y,elevs_interp,out_x_stacked,out_y_stacked)[1:] - \
                  (V[:-1]**2)*ARi-grav*CENTR(Y,elevs_interp,out_x_stacked,out_y_stacked)[:-1])
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
        FY = AR(YNORM,elevs_interp,out_x_stacked,out_y_stacked)*HR(YNORM)**0.6667 - C1
        DFDY = 1.6667*TOP(YNORM,elevs_interp,out_x_stacked,out_y_stacked)*HR(YNORM,elevs_interp,out_x_stacked,
                         out_y_stacked)**0.6667 - 0.6667*HR(YNORM,elevs_interp,out_x_stacked,out_y_stacked)**1.6667*C2
        YNEW = YNORM - FY/DFDY
        ERR = abs((YNEW - YNORM)/YNEW)
        YNORM = YNEW.copy()
        if (ERR < 1.0E-06):
            return
    return

#%% Initialize arrays below variables based on nsec

grav = 9.81 # m/s^2
nsec = len(min_dist_dx) # number of sections from the dike to high toss
np11 = 2*nsec + 2
tlast = 89400 # time for transient flow computation (measurements are at 5 min (300s) intervals - use these?)
iprint = 1 # counter for printing results
# READ(20,*)
# READ(20,*) CHL,B0,S,CMN,S0,Q0,Y0,YD,ALPHA,TOL,MAXITER
chl = np.nansum(min_dist_dx) # channel length (should be equal to nsec is grid is 1m)
# b0 = 10 # channel bottom width - ignore. use dem to find xsec area
s = 2 # channel lateral slope - ignore. use dem to find xsec area
cmn = 0.07 # manning's coefficient (use spaulding & grilli?)
# s0 = 0.001 # channel bottom slope - ignore. use dem to find xsec area
min_elevs = np.amin(elevs_interp,axis=1)
xsec_loc = np.cumsum(min_dist_dx)
xsec_loc = np.append([0],xsec_loc)
plt.plot(xsec_loc,min_elevs,label='data')
"""
Should I find a linear function that fits the elevations for an average slope?
"""
idx_elevs = np.isfinite(xsec_loc) & np.isfinite(min_elevs)
z_elevs = np.polyfit(xsec_loc[idx_elevs], min_elevs[idx_elevs], 1)
p_elevs = np.poly1d(z_elevs)
polyX_elevs = np.linspace(xsec_loc.min(), xsec_loc.max(), 100)
pylab.plot(polyX_elevs,p_elevs(polyX_elevs),"red", label='Average Slope of HR, HT to Dike')

# s0 = -z_elevs[0]
"""
Or a curve fit?
"""
# def poly1_func(x,a,b,c): # is this crazy? might want to adjust if actual y values aren't represented here
#     return a*(x)**b+c
# popt, pcov = curve_fit(poly1_func, xsec_loc, min_elevs)
# chl_smoothed = poly1_func(xsec_loc,*popt)
# plt.plot(xsec_loc,chl_smoothed,label='function')
# chl_slope = popt[1]*popt[0]*xsec_loc**(popt[1]-1) # derivative of curve fit function
# chl_slope[0] = chl_slope[1]
# plt.plot(xsec_loc,chl_slope,label='function slope')
# plt.legend()
# s0 = chl_slope
"""
Or find the slope at each point based on its neighbors? Make sure Preissmann can handle negative slopes
"""
slopes_HR = -np.diff(min_elevs)/min_dist_dx
# slopes1 = slopes_HR[:-1]
# slopes2 = slopes_HR[1:]
# slopes_HR_mean = np.array([(a + b) / 2 for a, b in zip(slopes1, slopes2)])
# slopes_HR_combined = np.concatenate((np.array([slopes_HR[0]]),slopes_HR_mean,np.array([slopes_HR[-1]])))
# plt.plot(xsec_loc,slopes_HR_combined,label='discretized slopes')
# plt.legend()
# s0 = slopes_HR_combined
"""
Or is it just the slopes between the cross-sections?
"""
s0 = slopes_HR

q0 = 0.6 # initial steady state discharge (use monthly-averaged fw discharge values)
# y0 = 1 # uniform flow depth (starting condition?) - may have to make high toss a transient boundary
elev_hightoss_avg = np.nanmean(HR_CTD_all_df["High Toss Water Level, NAVD88"])
elev_cnrus_avg = np.nanmean(HR_CTD_all_df["CNR U/S Water Level, NAVD88"])
y0 = np.linspace(elev_hightoss_avg,elev_cnrus_avg,40)-min_elevs # starting depths
yd = y0[-1] # flow depth at lower end (initial condition) - need to solve so that adjust so that Q_calc_channel_out = Q_dike
alpha = 1 # weighting coefficient (between 0.55 and 1)
tol = 0.0001 # tolerance for iterations
maxiter = 50 # maximum number of iterations

C1, C2 = (np.array([np.nan]*(nsec)) for i in range(2))

T = 0 # steady state, initial time
cmn_squared = cmn**2 # gravity units are metric, so no changes are necessary.

# Steady state conditions
"""
Should I be taking the average of the celerity and velocity for each section?
"""
c = np.sqrt(grav*AR(y0,elevs_interp,out_x_stacked,out_y_stacked)/TOP(y0,elevs_interp,out_x_stacked,out_y_stacked)) # celerity
v0 = q0/AR(y0,elevs_interp,out_x_stacked,out_y_stacked) # flow velocity
dx = min_dist_dx
# dx = cell_spacing # assuming channel length and number of sections are the same
dt = min_dist_dx/(v0[1:]+c[1:]) # time step length
dtx2 = 2*dt/min_dist_dx
yres = y0
i = 0
#np1 = nsec # remove for clarity. In fortran, np1 = nsec+1, python starts at 0 giving extra index

Z = min_elevs # bottom elev array
Y = y0 # depth array
# Y = np.ones_like(Z)*y0 # depth array # make sure this goes from High Toss (first val) to the dike (last val)
# Y[(Z+y0) < yd] = yd - Z[(Z+y0) < yd] # where elev + initial depth is less than downstream depth, make positive?
V = q0/AR(Y,elevs_interp,out_x_stacked,out_y_stacked)

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

# Insert initial conditions
H_out.append(Y+Z)
Q_out.append(V*AR(Y,elevs_interp,out_x_stacked,out_y_stacked))

while (T <= tlast) & (iflag == 0): # time loop, ntimes = tlast/dt
    print("Model time = {0:3.2f} s".format(T))
    print("High Toss V = {0:3.2f} m/s".format(V[0]))
    print("High Toss AR = {0:3.2f} m^2".format(AR(Y,elevs_interp,out_x_stacked,out_y_stacked)[0]))
    ITER = 0
    if (iprint == ip):
        ip = 0
        PREISS_H_out = np.vstack((PREISS_H_out,np.concatenate([['T=',T,'H='],[float(x) for x in (Y+Z)]])))
        PREISS_Q_out = np.vstack((PREISS_Q_out,
                                  np.concatenate([['T=',T,'Q='],
                                                  [float(x) for x in (V*AR(Y,elevs_interp,
                                                   out_x_stacked,out_y_stacked))]])))
    T = T + dt[ITER]
    # yd = 1.58 + 0.4*np.sin(math.pi*T/22350) # for changing boundary condition
    # !
    # !     GENERATE SYSTEM OF EQUATIONS
    # !
    
    ARi = AR(Y,elevs_interp,out_x_stacked,out_y_stacked)[:-1] # calculate flow area at upstream section
    ARiP1 = AR(Y,elevs_interp,out_x_stacked,out_y_stacked)[1:] # calculate flow area at downstream section
    C1 = dtx2*(1-alpha)*(ARiP1*V[1:]-ARi*V[:-1])-ARi-ARiP1
    C2 = make_C2(Y,V,ARi,ARiP1,cmn_squared,s0,grav)
      
    SUMM = tol+10
    for L in range(1,1000):
        plt.plot(Y,".",label=L-1)
        plt.plot(V,".",label=L-1)
        plt.legend()
        if (SUMM > tol):
            EQN = np.zeros((np11,np11+1),dtype=float) # should generate the same array?
            ITER = ITER+1    
            # !    
            # !        INTERIOR NODES
            # !    
            
            ARi = AR(Y,elevs_interp,out_x_stacked,out_y_stacked)[:-1] # calculate flow area at upstream section
            ARiP1 = AR(Y,elevs_interp,out_x_stacked,out_y_stacked)[1:] # calculate flow area at downstream section
            row_inds1 = 2*np.arange(nsec,dtype=int)+1 # every other row, starting at 1 (2nd row)
            EQN[row_inds1,np11]=-(ARi+ARiP1+dtx2*alpha*(V[1:]*ARiP1-V[:-1]*ARi)+C1) # sets last column
            
            sf1 = make_sf(Y,V,cmn_squared)[:-1]
            sf2 = make_sf(Y,V,cmn_squared)[1:]
            term1 = dtx2*alpha*((V[1:]**2)*ARiP1 + grav*CENTR(Y,elevs_interp,out_x_stacked,out_y_stacked)[1:]-
                                (V[:-1]**2)*ARi-grav*CENTR(Y,elevs_interp,out_x_stacked,out_y_stacked)[:-1])
            term2 = -alpha*dt*grav*((s0-sf2)*ARiP1+(s0-sf1)*ARi)
            EQN[row_inds1+1,np11] = -(V[:-1]*ARi+V[1:]*ARiP1+term1+term2+C2) # every other row, starting at 2 (3rd row)
            
            daY1 = TOP(Y,elevs_interp,out_x_stacked,out_y_stacked)[:-1]
            daY2 = TOP(Y,elevs_interp,out_x_stacked,out_y_stacked)[1:]
            EQN[row_inds1,row_inds1-1] = daY1*(1-dtx2*alpha*V[:-1])
            EQN[row_inds1,row_inds1] = -dtx2*alpha*ARi
            EQN[row_inds1,row_inds1+1] = daY2*(1+dtx2*alpha*V[1:])
            EQN[row_inds1,row_inds1+2] = dtx2*alpha*ARiP1
            
            dcdY1 = DCENDY(Y,dcendy_matrix,y_sample_midpoints)[:-1]
            dcdY2 = DCENDY(Y,dcendy_matrix,y_sample_midpoints)[1:]
            dsdV1 = 2*V[:-1]*cmn_squared/HR(Y,elevs_interp,out_x_stacked,out_y_stacked)[:-1]**1.333
            dsdV2 = 2*V[1:]*cmn_squared/HR(Y,elevs_interp,out_x_stacked,out_y_stacked)[1:]**1.333
            
            PERi = ARi/HR(Y,elevs_interp,out_x_stacked,out_y_stacked)[:-1]
            
            """
            # Change in PER (wetted perimeter) with respect to change in y (normal depth) (p. 104 Open Channel Flow)
            """
            PER_matrix = np.empty((40,100))
            for j in range(len(y_range_start)):
                PER_matrix[:,j] = AR(y_sample_allxsec[:,j],elevs_interp,out_x_stacked,
                          out_y_stacked)/HR(y_sample_allxsec[:,j],elevs_interp,out_x_stacked,out_y_stacked)
            PER_matrix[:,0] = 0 # When depth is zero, so is the wetted perimeter
            dPERdy_matrix = np.diff(PER_matrix)/np.diff(y_sample_allxsec)
            y_sample_midpoints = (y_sample_allxsec[:,1:]+y_sample_allxsec[:,:-1])/2
            dPERdy_act = []
            for row in range(len(y_sample_midpoints)):
                interp_func = interp1d(y_sample_midpoints[row],dPERdy_matrix[row])
                dPERdy_act.append(interp_func([Y[row]])[0])
            dPERdy_all = np.array(dPERdy_act)
            
            """
            # Linear fit of previous, dPERdy is slope
            """
            # PER_matrix[:,1:] = PER_matrix[:,1:]*[PER_matrix[:,:-1] != PER_matrix[:,1:]]
            # PER_matrix[PER_matrix == 0] = np.nan
            # PER_matrix[:,0] = 0 # When depth is zero, so is the wetted perimeter
            # idx_PER = np.isfinite(y_sample_allxsec) & np.isfinite(PER_matrix)
            # y_sample_finite = y_sample_allxsec*idx_PER
            # PER_matrix_finite = PER_matrix*idx_PER
            # z_PER_all = []
            # for row in range(len(PER_matrix)):
            #     z_PER = np.polyfit(y_sample_allxsec[row][idx_PER[row]],PER_matrix[row][idx_PER[row]], 1)
            #     z_PER_all.append(z_PER)
            # dPERdy_all = np.array(z_PER_all)[:,0]
            
            term1 = dPERdy_all[:-1]*ARi - daY1*PERi
            # term1 = 2*np.sqrt(1+s**2)*ARi - daY1*PERi # change based on lateral slope
            term2 = HR(Y,elevs_interp,out_x_stacked,out_y_stacked)[:-1]**0.333*ARi**2
            dsdY1 = 1.333*V[:-1]*abs(V)[:-1]*cmn_squared*term1/term2
            
            PERiP1 = ARiP1/HR(Y,elevs_interp,out_x_stacked,out_y_stacked)[1:]
            
            term1 = dPERdy_all[1:]*ARiP1-daY2*PERiP1
            # term1 = 2*np.sqrt(1+s**2)*ARiP1-daY2*PERiP1 # change based on lateral slope
            term2 = (HR(Y,elevs_interp,out_x_stacked,out_y_stacked)[1:]**0.333)*(ARiP1**2)
            dsdY2 = 1.333*V[1:]*abs(V)[1:]*cmn_squared*term1/term2
            
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
            # EQN[0,0] = 1.0
            # EQN[0,np11] = -(Y[0]-yres)
            # !
            # !         UPSTREAM END (V given)
            # !
            EQN[0,0]    = 1.0
            EQN[1,np11] = (V[0] - v0[ITER])                              # ! ok


            # !
            # !         DOWNSTREAM END (NO OUTFLOW)
            # !
            # EQN[-1,-2] = 1.
            # EQN[-1,np11] = 0. - V[-1]
            # DS END, STEADY OUTFLOW?
            # EQN[-1,np11] = v0[ITER] - V[-1] # fix
            # !       
            # !         DOWNSTREAM END (Y given)
            # !
            EQN[-1,-2] = 1.
            EQN[-1,np11] =  Y[-1] - yd                         # ! ok
            # !         DOWNSTREAM END (V given)
            # !
            # EQN[-1,-2]    = 1.0
            # EQN[-1,np11] = (V[-1] - v0[ITER])                              # ! ok
       
            # Run implicit solution 
            DF = MATSOL(np11-1,EQN)
            
            # Organize output
            SUMM = np.sum(DF)
            Y = Y + DF[::2]
            V = V + DF[1::2]
            
            #CHECK NUMBER OF ITERATIONS 
            if (ITER > maxiter):
                iflag = 1
                SUMM = tol.copy()
        else:
            break
    ip = ip+1
    H_out.append(Y+Z)
    Q_out.append(V*AR(Y,elevs_interp,out_x_stacked,out_y_stacked))

H_out = np.array(H_out)
Q_out = np.array(Q_out)

# IF(IFLAG.EQ.1) WRITE(6,"('MAXIMUM NUMBER OF ITERATIONS EXCEEDED')")
# STOP
# END

if (iflag == 1):
    print("Maximum number of iterations exceeded")

#%% Analytical Estimation of HRside Levels, My Analysis (all SI)
# Using WF Harbor Levels and HR level guess to get Discharge Through Dike.
# Using HR level guess and FW discharge to get all levels in HR.
# Using "next guess" to get "next volume" - compare average discharge over two guesses with change in V over time. Optimize.



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







