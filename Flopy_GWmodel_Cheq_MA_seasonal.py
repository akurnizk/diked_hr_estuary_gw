# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:47:36 2018

@author: akurnizk
"""

import utm
import csv
import math
import flopy
import sys,os
import affine
import calendar
import dateutil
import numpy as np
import pandas as pd
import geopandas as gpd
import flopy.utils.binaryfile as bf

from shapely.geometry import Point

import statistics
from statistics import mode

import moviepy.editor as mpy
from moviepy.editor import *

import matplotlib as mpl
mpl.rc('xtick', labelsize=22)     
mpl.rc('ytick', labelsize=22)
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import pylab

cgw_code_dir = 'E:\Python KMB - CGW' # Location of BitBucket folder containing cgw folder
sys.path.insert(0,cgw_code_dir)

from cgw.utils import general_utils as genu
from cgw.utils import feature_utils as shpu
from cgw.utils import raster_utils as rastu
from mpl_toolkits.axes_grid1 import make_axes_locatable

import rasterio
from rasterio import mask
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.io import MemoryFile
from rasterio.enums import Resampling

# Assign name and create modflow model object

modelname = 'CheqModel2'
work_dir = os.path.join('E:\Herring Models\Seasonal')
map_dir = r'E:\Maps' # retrieved files from https://viewer.nationalmap.gov/basic/
mfexe = "MODFLOW-NWT.exe"
# mf = flopy.modflow.Modflow(modelname, exe_name='mf2005',model_ws=work_dir)
mf = flopy.modflow.Modflow(modelname, exe_name=mfexe,model_ws=work_dir, version='mfnwt') # fix me - playing with obtaining NWT solver
swt = flopy.seawat.Seawat(modelname, exe_name='swtv4')
print(swt.namefile)

# Monthly arrays generated from NOAA NAVD88 monthly average data from Aug 2015 to March 2019 for Chatham.
# Monthly arrays generated from USGS NAVD88 monthly average data from Jan 2015 to Aug 2019 for Provincetown.
months_str = calendar.month_name
x_months = np.array(months_str[1:]) # array of the months of the year
x_ss_months = np.append(np.array(['Steady State']), x_months)

#%%
# NOAA station 8447435, Chatham, Lydia Cove MA (use these values for the right boundary)
mean_sea_level_monthly_east = np.array([0.142037, 0.077267, 0.091643, 0.146914, 0.148031, 0.164287, 0.123342, 0.114757, 0.148209, 0.163525, 0.134798, 0.088697]) 
# https://tidesandcurrents.noaa.gov/datums.html?units=1&epoch=0&id=8447435&name=Chatham%2C+Lydia+Cove&state=MA
mean_sea_level_annual_east = np.array([mean_sea_level_monthly_east.mean()])

"""
CHANGE FOR SLR
"""
# With SLR (https://tidesandcurrents.noaa.gov/publications/techrpt83_Global_and_Regional_SLR_Scenarios_for_the_US_final.pdf)
# mean_sea_level_annual_east = mean_sea_level_annual_east + 0.5 # half meter of SLR (Int-Low)
# mean_sea_level_annual_east = mean_sea_level_annual_east + 1.5 # one and a half meter of SLR (Int-High)
# mean_sea_level_annual_east = mean_sea_level_annual_east + 2.5 # two and a half meter of SLR (Extreme)

msl_ss_monthly_east = np.append(mean_sea_level_annual_east, mean_sea_level_monthly_east)

#%%
# Provincetown USGS 420259070105600 MSL Monthly averages from Daily data (for left boundary)
mean_sea_level_monthly_west = np.array([0.011995, -0.036057, -0.000885, 0.004470, 0.013647, 0.041189, 0.049098, 0.043139, 0.060604, 0.070694, 0.029235, -0.000951])
# https://waterdata.usgs.gov/nwis/inventory/?site_no=420259070105600&agency_cd=USGS
mean_sea_level_annual_west = np.array([mean_sea_level_monthly_west.mean()])

"""
CHANGE FOR SLR
"""
# With SLR (https://tidesandcurrents.noaa.gov/publications/techrpt83_Global_and_Regional_SLR_Scenarios_for_the_US_final.pdf)
# mean_sea_level_annual_west = mean_sea_level_annual_west + 0.5 # half meter of SLR (Int-Low)
# mean_sea_level_annual_west = mean_sea_level_annual_west + 1.5 # one and a half meter of SLR (Int-High)
# mean_sea_level_annual_west = mean_sea_level_annual_west + 2.5 # two and a half meter of SLR (Extreme)

msl_ss_monthly_west = np.append(mean_sea_level_annual_west, mean_sea_level_monthly_west)

# SLR determined from NOAA SLR Viewer at Sandwich Marina, MA out to 2100
# https://coast.noaa.gov/slr/#/layer/sce/0/-7855694.840194072/5156340.373175255/10/satellite/26/0.8/2100/interHigh/midAccretion
# https://tidesandcurrents.noaa.gov/map/index.html
# Taking 2.5 m is int-high/high, 1.5 m is int/int-high, and 0.5 m is low/int-low, assuming viewer has relative SLR and not absolute levels

#%% Using rasterio

def xy_from_affine(tform=None,nx=None,ny=None):
    X,Y = np.meshgrid(np.arange(nx)+0.5,np.arange(ny)+0.5)*tform
    return X,Y

#%% Loading in shapefiles and DEM

area_df = gpd.read_file(os.path.join(map_dir,'Chequesset_Model_Area_UTM.shp'))
area_df_HR = gpd.read_file(os.path.join(map_dir,'Herring River_Diked.shp'))
area_df_CheqWest = gpd.read_file(os.path.join(map_dir,'Chequesset_West.shp'))
area_df_CheqEast = gpd.read_file(os.path.join(map_dir,'Chequesset_East.shp'))
area_df_PoorConv = gpd.read_file(os.path.join(map_dir,'Poor_Convergence_11_11_19.shp')) # this area was converging poorly
area_df_SLAMM = gpd.read_file(os.path.join(map_dir,'SLAMMinputareas_NPSwells','SLAMM_inputs_kmb_14Aug19.shp')) # this area was converging poorly
# area_df_NHDPonds = gpd.read_file(os.path.join(map_dir, 'NHDPonds.shp'))

# Making fake raster for model domain
temp_crs = area_df.crs # same as polygon

minx,miny,maxx,maxy = area_df.bounds.values.T
leftbound,topbound = minx.min(),maxy.max() # top left corner of model domain
xres, yres = 10, 10 # grid resolution

"""
Chequesset Region
"""

# Using SHPU to find rotation angle
shp_fname = os.path.join(map_dir,'Chequesset_Model_Area_UTM.shp')
cell_spacing = 10. # model grid cell spacing in meters
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

# Load filled dem to model domain
demfname = os.path.join(map_dir,'Cheq10mx10m_UTM_Combined_Dike_HydroFill.tif')
with rasterio.open(demfname) as src:
    with WarpedVRT(src,**vrt_options) as vrt:
        dem_model = vrt.read()[0] # [0] selects first band
        dem_model[dem_model==vrt.nodata] = np.nan
        # Should be dem_model.shape = [model_height,model_width]

# Load unfilled dem to model domain
demfname_nofill = os.path.join(map_dir,'Cheq10mx10m_UTM_Combined.tif')
with rasterio.open(demfname_nofill) as src:
    with WarpedVRT(src,**vrt_options) as vrt:
        dem_model_nofill = vrt.read()[0] # [0] selects first band
        dem_model_nofill[dem_model_nofill==vrt.nodata] = np.nan
        # Should be dem_model.shape = [model_height,model_width]

# Generate grid and mask of places that were hydrofilled (greater than 5 cm)
dem_model_hydrofill = dem_model - dem_model_nofill
dem_model_hydrofill[dem_model_hydrofill<0] = 0 # hydrofill should not drop the elevation        
dem_model_hydrofill_mask = dem_model_hydrofill.copy()
dem_model_hydrofill_mask[dem_model_hydrofill_mask>=0.05] = 1 # consider hydrofill only significant greater than 5 cm.
dem_model_hydrofill_mask[dem_model_hydrofill_mask<1] = 0
        
dem_X,dem_Y,dem_da = rastu.load_geotif(demfname) # dem_da is an xarray data array
# Experimental example of loading DEM data for that area (outdated)
# dem_vals = dem_da.values.squeeze()
# decimate_by_ncells = 1 # by every n cells (can decimate dem to save time)
# dem_vals[dem_vals==dem_da.nodatavals[0]] = genu.np.nan # Set no-data value to nan
# # Transform dem to model coordinates with linear interpolation
# trans_dict = {'orig_xy':[dem_X,dem_Y],'orig_val':dem_vals,'active_method':'linear','new_xy':cc_proj} # if dem in same projection as model boundary shp
# dem_trans = rastu.subsection_griddata(**trans_dict)
# dem_trans[dem_trans<-1000] = genu.np.nan
# genu.quick_plot(dem_trans, vmin=0, vmax=60) # plots elevations (in meters), with only active cells
# # plot needs x, y labels and colorbar label (elevation)


#%% Example of making a MODFLOW-like grid from a shapefile

# Use shpu with model_polygon to define active cells in the model (outdated)
# Cheq region
# ir,ic,_ = shpu.gridpts_in_shp(model_polygon,cc)
# active_cells = genu.define_mask(cc,[ir,ic])
# HR region
# ir_HR,ic_HR,_ = shpu.gridpts_in_shp(model_polygon_HR,cc_HR)
# active_cells_HR = genu.define_mask(cc_HR,[ir_HR,ic_HR])

# Create masked areas for model domain using area_df
mask_array = np.zeros([model_height,model_width])
mask_array_HR = np.zeros([model_height,model_width])
mask_array_CheqWest = np.zeros([model_height,model_width])
mask_array_CheqEast = np.zeros([model_height,model_width])
mask_array_PoorConv = np.zeros([model_height,model_width])
mask_array_SLAMM = np.zeros([model_height,model_width])
# mask_array_NHDPonds = np.zeros([model_height,model_width])


# Make temporary raster in memory
with MemoryFile() as memfile:
    with memfile.open(**model_profile) as dataset:
        tempdataset = np.ones([1,model_height,model_width]) # make array of all one value
        dataset.write(tempdataset)
        for igeom,feature in enumerate(area_df.geometry.values):      # loop through features in Cheq shp
            mask_rast,tform = mask.mask(dataset,[feature],crop=False,all_touched=True)
            mask_rast = mask_rast.squeeze()
            mask_array[mask_rast==1] = igeom+1 # start at 1
        for igeom,feature in enumerate(area_df_HR.geometry.values):      # loop through features in HR shp
            mask_rast_HR,tform_HR = mask.mask(dataset,[feature],crop=False,all_touched=True)
            mask_rast_HR = mask_rast_HR.squeeze()
            mask_array_HR[mask_rast_HR==1] = igeom+1 # start at 1
        for igeom,feature in enumerate(area_df_CheqWest.geometry.values):      # loop through features in CheqWest shp
            mask_rast_CheqWest,tform_CheqWest = mask.mask(dataset,[feature],crop=False,all_touched=True)
            mask_rast_CheqWest = mask_rast_CheqWest.squeeze()
            mask_array_CheqWest[mask_rast_CheqWest==1] = igeom+1 # start at 1
        for igeom,feature in enumerate(area_df_CheqEast.geometry.values):      # loop through features in CheqEast shp
            mask_rast_CheqEast,tform_CheqEast = mask.mask(dataset,[feature],crop=False,all_touched=True)
            mask_rast_CheqEast = mask_rast_CheqEast.squeeze()
            mask_array_CheqEast[mask_rast_CheqEast==1] = igeom+1 # start at 1
        for igeom,feature in enumerate(area_df_PoorConv.geometry.values):      # loop through features in CheqEast shp
            mask_rast_PoorConv,tform_PoorConv = mask.mask(dataset,[feature],crop=False,all_touched=True)
            mask_rast_PoorConv = mask_rast_PoorConv.squeeze()
            mask_array_PoorConv[mask_rast_PoorConv==1] = igeom+1 # start at 1
        for igeom,feature in enumerate(area_df_SLAMM.geometry.values):      # loop through features in CheqEast shp
            mask_rast_SLAMM,tform_SLAMM = mask.mask(dataset,[feature],crop=False,all_touched=True)
            mask_rast_SLAMM = mask_rast_SLAMM.squeeze()
            mask_array_SLAMM[mask_rast_SLAMM==1] = igeom+1 # start at 1
        # for igeom,feature in enumerate(area_df_NHDPonds.geometry.values):
        #     mask_rast_NHDPonds,tform_NHDPonds = mask.mask(dataset,[feature],crop=False,all_touched=True)
        #     mask_rast_NHDPonds = mask_rast_NHDPonds.squeeze()
        #     mask_array_NHDPonds[mask_rast_NHDPonds==1] = igeom+1

# When set active, cell at col 873 and row 1034 has trouble converging, so set inactive (not in new shapefile)
# mask_array[1034,873] = 0

# Bottom of first layer based on fill depths in HR (visual inspection of plot)
channel_depth_HR = 3.21 # meters

"""
Mill Creek Area can also be included (Needs to be diked when HR dike is removed)
"""

#%% Land Surface Elevation Plots (Active, HR, West, and East)
"""
Plot active cells
"""
# Mask Array
fig,ax = genu.plt.subplots(1,2)
ax[0].set_xlabel('column #')
ax[0].set_ylabel('row #')
genu.quick_plot(np.ma.masked_array(dem_model,mask_array!=1),vmin=0,vmax=60,ax=ax[0])
#genu.quick_plot(mask_array.astype(int),ax=ax[0]) # in row, column space
c1=ax[1].pcolormesh(cc[0],cc[1],mask_array.astype(int)) # in model coordinates
genu.plt.colorbar(c1,ax=ax[1],orientation='horizontal')
fig.gca().set_aspect('equal', adjustable='box')
ax[1].set_xlabel('X [m]')
ax[1].set_ylabel('Y [m]')

# HR Mask
fig,ax = genu.plt.subplots(1,2)
ax[0].set_xlabel('column #')
ax[0].set_ylabel('row #')
genu.quick_plot(np.ma.masked_array(dem_model,mask_array_HR!=1),vmin=0,vmax=20,ax=ax[0])
#genu.quick_plot(mask_array.astype(int),ax=ax[0]) # in row, column space
c1=ax[1].pcolormesh(cc[0],cc[1],mask_array_HR.astype(int)) # in model coordinates
genu.plt.colorbar(c1,ax=ax[1],orientation='horizontal')
fig.gca().set_aspect('equal', adjustable='box')
ax[1].set_xlabel('X [m]')
ax[1].set_ylabel('Y [m]')

# West Mask
fig,ax = genu.plt.subplots(1,2)
ax[0].set_xlabel('column #')
ax[0].set_ylabel('row #')
genu.quick_plot(np.ma.masked_array(dem_model,mask_array_CheqWest!=1),vmin=0,vmax=20,ax=ax[0])
#genu.quick_plot(mask_array.astype(int),ax=ax[0]) # in row, column space
c1=ax[1].pcolormesh(cc[0],cc[1],mask_array_CheqWest.astype(int)) # in model coordinates
genu.plt.colorbar(c1,ax=ax[1],orientation='horizontal')
fig.gca().set_aspect('equal', adjustable='box')
ax[1].set_xlabel('X [m]')
ax[1].set_ylabel('Y [m]')

# East Mask
fig,ax = genu.plt.subplots(1,2)
ax[0].set_xlabel('column #')
ax[0].set_ylabel('row #')
genu.quick_plot(np.ma.masked_array(dem_model,mask_array_CheqEast!=1),vmin=0,vmax=20,ax=ax[0])
#genu.quick_plot(mask_array.astype(int),ax=ax[0]) # in row, column space
c1=ax[1].pcolormesh(cc[0],cc[1],mask_array_CheqEast.astype(int)) # in model coordinates
genu.plt.colorbar(c1,ax=ax[1],orientation='horizontal')
fig.gca().set_aspect('equal', adjustable='box')
ax[1].set_xlabel('X [m]')
ax[1].set_ylabel('Y [m]')
                            
#%% DEM model inputs

Lx = np.amax(dem_X)-np.amin(dem_X) # should be combined DEM size in X (10 km)
Ly = np.amax(dem_Y)-np.amin(dem_Y) # should be combined DEM size in Y (30 km)
# ztop = dem_model_nofill # set the top to the unfilled dem elevations

dem_model_nofill_mask = np.ma.masked_invalid(dem_model_nofill).mask
ztop = dem_model # set the top to the dem elevations, including hydrofill
ztop[dem_model_nofill_mask==True] = np.nan # set the top to the filled dem elevations
# ztop = np.ma.masked_array(dem_model,mask_array!=1) # set the top to the dem elevations, masked
# zbot = -100 # if bottom of model is horizontal, approx. bedrock (Masterson_2004 - zone of transition (ZOT) on p.68)
# nlay = 1 # 1 layer model
nrow, ncol = cc[0].shape
delr = cell_spacing # delr = Lx/ncol in Tutorial 1
delc = cell_spacing # delc = Ly/now in Tutorial 1
# delv = (ztop - zbot) / nlay # fix for multilayer
# botm = zbot # from Tutorial 1, botm = np.linspace(ztop, zbot, nlay + 1), which gives array for various layer bottoms (Masterson_2004)

# by APK from Masterson_2004 (For multilayer model)
Lz = -500 # feet (from Masterson_2004)
botm = np.linspace(0, Lz, 26) # bottom of each layer
botm[0] = -5
botm = np.delete(botm,[23,24,25])
botm[11:]=np.linspace(-225, Lz, 12)
botm = botm*0.3048 # feet to meters
zbot = botm[-1]
nlay = botm.size
delv = -Lz/nlay*0.3048

botm_reduced = np.delete(botm,np.s_[2:22])
# botm_reduced = np.delete(botm_reduced,[0])
nlay_reduced = botm_reduced.size
# ztop[ztop<botm[0]] = botm[0] # Make all elevations higher than the bottom of the first layer?

# Make a filled height array at a higher elevation than the depth of the bottom of layer 2 (-6.1 meters)
filled_height = dem_model_hydrofill.copy()
# filled_height[(mask_array_HR==1) & (filled_height>abs(botm_reduced[1]))] = abs(np.around(botm_reduced[1]))

# Make botm an array
zbotm_red_array = botm_reduced[:,None,None]*np.ones([nlay_reduced,nrow,ncol])
# Set all first layer bottom elevations 
zbotm_red_array[0,~np.isnan(filled_height)] = ztop[~np.isnan(filled_height)]-filled_height[~np.isnan(filled_height)] # works as long as max(filled_height)<(ztop-zbtom[1])
# Anywhere it wasn't hydrofilled, set the bottom depth to botm[0]
zbotm_red_array[0][(dem_model_hydrofill_mask!=1) & (zbotm_red_array[0]>botm[0])] = botm[0]
# Set all nan vals (in the ocean) to 50 cm below the lowest elevation on the map
zbotm_red_array[0][np.isnan(ztop)] = np.nanmin(ztop)
# Set all first layer bottom elevations that are at or above the topography and unfilled to 50 cm below the topography
# zbotm_red_array[0][(ztop<=botm[0]) & (filled_height==0)] = zbotm_red_array[0][(ztop<=botm[0]) & (filled_height==0)]-0.5 
# Set all first layer bottom elevations within 1 m of the top elevation to 1 m below the top elevation
# zbotm_red_array[0][(ztop-zbotm_red_array[0])<=1.0] = zbotm_red_array[0][(ztop-zbotm_red_array[0])<=0.5]-0.5
zbotm_red_array[0][(ztop-zbotm_red_array[0])<=1.0] = zbotm_red_array[0][(ztop-zbotm_red_array[0])<=1.0]-1.0

# Round values to keep layer depths the same precision.
# zbotm_red_array = np.around(zbotm_red_array, decimals=2)

print("Smallest cell depth is",np.nanmin(ztop-zbotm_red_array[0]), "meters.") # this must be greater than 100*Thickfact, where Thickfact is default 1e-5 from NWT-Solver
print("Lowest first layer depth is",np.nanmin(zbotm_red_array[0]), "meters.") # this should not be too much smaller than botm[0], the bottom of Masterson's first layer
genu.quick_plot(zbotm_red_array[0],vmin=np.nanmin(zbotm_red_array[0]),vmax=-1) # plot showing where the cells are below -1 m 

#%% Time Stepping
"""
Time Stepping For tidal variations over ONE YEAR (accounts for seasonal variations)

Need to step through changes in land surface elevation as a result of spatially varying accretion?
    -> Only for long time steps (ignore for 1 year)
"""

# Time step parameters
total_length = 365 # days (add more for more spinup)
dt = 12 # stress period time step, months
perlen_years = dt/12. # stress period time step, years

nper = int(total_length/perlen_years) + 1 # the number of stress periods in the simulation (+ 1 for Steady State)
nstp_default = int(dt) # 1 time step per period
perlen = [perlen_years]*nper # length of a stress period; each item in the matrix is the amount 
                            # of elapsed time since the previous point (need to change the first)
perlen[0] = 1 # set first step as steady state
steady = [False]*nper
steady[0] = True # first step steady state
nstp = [nstp_default]*nper # number of time steps in a stress period
nstp[0] = 1

# Steady
nper = 1
nstp = [nstp[0]]
# perlen = [perlen[0]]
perlen = total_length # which one?
steady = [steady[0]]

"""
Changing sea-level: Only matters for < 6 hour time steps
https://tidesandcurrents.noaa.gov/noaatidepredictions.html?id=8446613
Using generalized sinusoidal equation for sea-level change with respect to time. Amplitude ~ 2m
Need to fix formula to better represent tidal predictions (can also use a more simplified curve to represent seasonal sea levels)
"""
# meas_times_hrs = []
# sea_levels = []
# next_time = 0
# for per in range(nper):
#     meas_times_hrs.append([next_time*24])
#     sea_levels.append([mean_sea_level_monthly_east + np.sin(86400*np.pi*per*perlen_days/22350)]) # 22350 represents the time, in seconds, 
#                                                                                    # to go from high tide to low tide (half of a period).
#     next_time = next_time + perlen[1]

# plt.plot(meas_times_hrs, sea_levels) # plots sea level changing over 40 time periods.
# plt.xlabel('time (hrs)')
# plt.ylabel('sea level (m)')
# plt.axis([24, 48, -0.5, 2.0])
# plt.grid(True)
# plt.show()

# sl_max = np.max(sea_levels) # maximum sea level from array generated by sea_levels formula
# sl_min = np.min(sea_levels) # minimum sea level from array generated by sea_levels formula
# max_amplitude = sl_max-sl_min

#%% Create the discretization (DIS) object

# laycbd indicates if a layer has a Quasi-3D confining bed below it. 0 indicates no confining bed. laycbd must be 0 on bottom layer - why? 
# Masterson, 2004 indicates bedrock between 450 ft in Eastham and 900 ft in Truro. # Masterson indicates 500 ft below NGVD 29 is no-flow.

dis = flopy.modflow.ModflowDis(mf, nlay_reduced, nrow, ncol, delr=delr, delc=delc,
                               top=ztop, botm=zbotm_red_array, nper=nper, perlen=perlen, nstp=nstp, steady=steady) # Tutorial 1 & 2, botm=botm[1:]
                                                              
#%% Variables for the BAS (basic) package

"""
Example (Added 5/28/19 from Bakker_2016):
Active cells and the like are defined with the Basic package (BAS), which is required for every MOD-FLOW model. 
It contains the ibound array, which is used to specify which cells are active (value is positive), inactive (value is 0), 
or fixed head (value is negative). The numpy package (aliased as np) can be used to quickly initialize the ibound array 
with values of 1, and then set the ibound value for the first and last columns to −1. The numpy package (and Python, in general) 
uses zero-based indexing and supports negative indexing so that row 1 and column 1, and row 1 and column 201, can be 
referenced as [0,0], and [0,−1], respectively. Although this simulation is for steady flow, starting heads still need 
to be specified. They are used as the head for fixed-head cells (where ibound is negative), and as a starting point to compute 
the saturated thickness for cases of unconfined flow.

ibound = np.ones((1, 201))
ibound[0, 0] = ibound[0, -1] = -1
"""

ibound = np.ones((nlay_reduced, nrow, ncol), dtype=np.int32) # make array of ones (active cells) with shape = nlay, nrow, ncol
ibound[:,mask_array!=1] = 0 # far offshore cells are inactive (set to zero)
# ibound[0,ztop<mean_sea_level_monthly_east] = -1 # fixed head for everything less than msl (0 for layer) (only leave in for constant head)
ibound[:,np.isnan(ztop)] = 0 # nan cells are inactive
genu.quick_plot(ibound[0]) # plots layer 1 boundary condition: 1 is above mean sea level (msl), 0 is msl, -1 is under msl.
# needs title and labels

"""
Starting heads without a diked Herring River
"""
strt = np.ones((nlay_reduced, nrow, ncol), dtype=np.float32) # starting heads at zero elevation
active_cells = mask_array.astype(bool)
active_dem_heights = ztop[active_cells & ~np.isnan(ztop)]

strt[0,active_cells & ~np.isnan(ztop)] = active_dem_heights # start with freshwater at surface elevation

# start at assumed steady-state levels
strt[0,~active_cells] = 0 # water (seawater) outside mask_array starts at 0 elevation
strt[0,(mask_array_CheqWest==1)&(ztop<=mean_sea_level_annual_west)] = mean_sea_level_annual_west # start with West surface water at mean Provincetown sea level
strt[0,(mask_array_CheqEast==1)&(ztop<=mean_sea_level_annual_east)] = mean_sea_level_annual_east # start with East surface water at mean Chatham sea level
strt[0,(mask_array_CheqWest==1)&(strt[0,:,:]==1)] = mean_sea_level_annual_west
strt[0,(mask_array_CheqEast==1)&(strt[0,:,:]==1)] = mean_sea_level_annual_east
strt[0,active_cells&(mask_array_CheqWest!=1)&(mask_array_CheqEast!=1)&(ztop<0)] = 0 # start with surface water at zero elevation

fig,ax = genu.plt.subplots(1,2)
ax[0].set_xlabel('column #')
ax[0].set_ylabel('row #')
genu.quick_plot(strt[0],vmin=-0.04,vmax=0.2,ax=ax[0])
genu.quick_plot(np.ma.masked_array(strt[0],mask_array!=1),vmin=-1,vmax=10,ax=ax[1])
fig.gca().set_aspect('equal', adjustable='box')
ax[1].set_xlabel('column #')
ax[1].set_ylabel('row #')
  
"""
Starting heads including the currently diked Herring River (comment out when not in use)
"""
# Need to look through Woods Hole Group Report to get better estimates for HR levels
CNRUS_UTM = (411883.34, 4642659.23) # meters (Easting, Northing)
DogLeg_UTM = (412671.25, 4643616.72)
HighToss_UTM = (412353.97, 4644108.12)
model_transform_inv = ~model_transform
CNRUS_colrow = model_transform_inv*CNRUS_UTM
DogLeg_colrow = model_transform_inv*DogLeg_UTM
HighToss_colrow = model_transform_inv*HighToss_UTM

x_coords = np.array([np.linspace(1,ncol,ncol),]*nrow)
y_coords = np.array([np.linspace(1,nrow,nrow),]*ncol).T
disp_CNRUS_sensor = cell_spacing*(((x_coords-CNRUS_colrow[0])**2 + (y_coords-CNRUS_colrow[1])**2)**0.5)/1000 # kilometers from CNR U/S CTD sensor
disp_CNRUStoHighToss = disp_CNRUS_sensor[int(HighToss_colrow[1]),int(HighToss_colrow[0])]

# updated displacements are linear distances instead of river-path distances
jan_HR_levels=(0.239641)*(disp_CNRUS_sensor**2)+(-0.201917)*(disp_CNRUS_sensor)+(-0.433711)
feb_HR_levels=(0.163921)*(disp_CNRUS_sensor**2)+(-0.147393)*(disp_CNRUS_sensor)+(-0.351285)
mar_HR_levels=(0.088200)*(disp_CNRUS_sensor**2)+(-0.092870)*(disp_CNRUS_sensor)+(-0.268859)
apr_HR_levels=(0.154211)*(disp_CNRUS_sensor**2)+(-0.158306)*(disp_CNRUS_sensor)+(-0.237086)
may_HR_levels=(-0.051253)*(disp_CNRUS_sensor**2)+(0.104700)*(disp_CNRUS_sensor)+(-0.237799)
jun_HR_levels=(0.006757)*(disp_CNRUS_sensor**2)+(0.029318)*(disp_CNRUS_sensor)+(-0.224866)
jul_HR_levels=(0.120142)*(disp_CNRUS_sensor**2)+(-0.126807)*(disp_CNRUS_sensor)+(-0.276057)
aug_HR_levels=(0.278647)*(disp_CNRUS_sensor**2)+(-0.361114)*(disp_CNRUS_sensor)+(-0.292101)
sep_HR_levels=(0.191879)*(disp_CNRUS_sensor**2)+(-0.223242)*(disp_CNRUS_sensor)+(-0.285530)
oct_HR_levels=(0.142866)*(disp_CNRUS_sensor**2)+(-0.123566)*(disp_CNRUS_sensor)+(-0.307802)
nov_HR_levels=(0.276687)*(disp_CNRUS_sensor**2)+(-0.345091)*(disp_CNRUS_sensor)+(-0.275408)
dec_HR_levels=(0.057434)*(disp_CNRUS_sensor**2)+(-0.002049)*(disp_CNRUS_sensor)+(-0.365406)

mean_HR_levels=(0.136838)*(disp_CNRUS_sensor**2)+(-0.136449)*(disp_CNRUS_sensor)+(-0.291329)

mean_HR_level_CNRUS = mean_HR_levels[int(np.around(CNRUS_colrow[1])),int(np.around(CNRUS_colrow[0]))]
mean_HR_level_DogLeg = mean_HR_levels[int(np.around(DogLeg_colrow[1])),int(np.around(DogLeg_colrow[0]))]
mean_HR_level_HighToss = mean_HR_levels[int(np.around(HighToss_colrow[1])),int(np.around(HighToss_colrow[0]))]

CNRUS_to_HighToss_km = (((HighToss_UTM[0]-CNRUS_UTM[0])**2+(HighToss_UTM[1]-CNRUS_UTM[1])**2)**0.5)/1000

"""
SLR Predictions from: https://tidesandcurrents.noaa.gov/publications/techrpt83_Global_and_Regional_SLR_Scenarios_for_the_US_final.pdf
"""
"""
CHANGE FOR SLR

Based on CTD data, tidal ranges are assumed to not be fluctuating substantially, so absolute rise added to mean levels
can also be added to means of lows and means of highs and get valid estimates.
"""

# 0.5 meters SLR results in 0.5 meters rise everywhere within HR, since there is still drainage assumed (Int-Low)
# mean_HR_levels[:,:] = mean_HR_level_CNRUS + 0.5
# 1.5 meters SLR will cause nearly double the rise inside the dike (now more like a dam), based on the ratio of tidal maxima 
# rise rates CNR/US:Oceanside from CTD/dike data. Not enough data, but definitely a potential scenario. This essentially
# matches the hydrofill (Int-High)
# mean_HR_levels[:,:] = mean_sea_level_annual_west + 1.0
# 2.5 meters SLR will cause the HR to almost always be overtopping the dike, essentially matching the top of the dike (Extreme)
# mean_HR_levels[:,:] = mean_sea_level_annual_west + 0.5


mean_HR_levels[disp_CNRUS_sensor>disp_CNRUStoHighToss] = mean_HR_levels[int(HighToss_colrow[1]),int(HighToss_colrow[0])]

HR_ss_monthly_levels = [mean_HR_levels, jan_HR_levels, feb_HR_levels, mar_HR_levels, apr_HR_levels, may_HR_levels, jun_HR_levels, jul_HR_levels, 
             aug_HR_levels, sep_HR_levels, oct_HR_levels, nov_HR_levels, dec_HR_levels]

strt[0,(mask_array_HR==1)&(dem_model_nofill<=mean_HR_levels)] = mean_HR_levels[(mask_array_HR==1)&(dem_model_nofill<=mean_HR_levels)]

# Round strt to see if that helps convergence
strt = np.around(strt,decimals=2)

fig,ax = genu.plt.subplots(1,2)
ax[0].set_xlabel('column #')
ax[0].set_ylabel('row #')
genu.quick_plot(strt[0],vmin=-0.04,vmax=0.2,ax=ax[0])
genu.quick_plot(np.ma.masked_array(strt[0],mask_array!=1),vmin=-1,vmax=10,ax=ax[1])
fig.gca().set_aspect('equal', adjustable='box')
ax[1].set_xlabel('column #')
ax[1].set_ylabel('row #')
ax[0].plot(CNRUS_colrow[0],CNRUS_colrow[1],'ro',markersize=4)
ax[0].annotate('CNR U/S',CNRUS_colrow,color='red',xytext=(240,700),arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=-0.6',color='red'))
ax[0].annotate('CNR U/S CTD',CNRUS_colrow,color='red',xytext=(240,700))
ax[0].plot(DogLeg_colrow[0],DogLeg_colrow[1],'ro',markersize=4)
ax[0].annotate('Dog Leg',DogLeg_colrow,color='red',xytext=(320,630),arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=-0.6',color='red'))
ax[0].annotate('Dog Leg CTD',DogLeg_colrow,color='red',xytext=(320,630))
ax[0].plot(HighToss_colrow[0],HighToss_colrow[1],'ro',markersize=4)
ax[0].annotate('High Toss',HighToss_colrow,color='red',xytext=(270,550),arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=-0.6',color='red'))
ax[0].annotate('High Toss CTD',HighToss_colrow,color='red',xytext=(270,550))

#%% Save starting heads as .tif

dx,dy,gt = rastu.get_rast_info(demfname)
projwkt = rastu.load_grid_prj(demfname)
fname = os.path.join(work_dir, 'Cheq_StrHead_Elevation_Topo.tif')
rastu.write_gdaltif(fname,cc_proj[0],cc_proj[1],strt[0],proj_wkt=projwkt)

#%% Use starting heads from previous run

strt = head # only works if the following has be run after running the model:

# hds = bf.HeadFile(os.path.join(work_dir,modelname+'.hds'))
# time = 0
# head = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
# head[head<-100] = np.nan

#%% Create the basic (BAS) object

bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

#%% Variables for the UPW (Upstream Weighting) Package - required for NWT solver, formerly LPF (Layer-Property Flow).

# added 3/8/19 - creates matrix where hydraulic conductivities (hk = horiz, vk = vert) can be implemented; units = m/d
# should I change the hk, vk, ss, and sy values for surface water bodies?

# hy values are initially in feet (from MF-96, HY is the hydraulic conductivity along rows. HY is multiplied by TRPY to 
                                    # obtain hydraulic conductivity along columns)

hk = np.zeros((nlay,nrow,ncol), dtype=np.float32)
for i in range(9):
    locals()["hy_lay0"+str(i+1)+"_fname"] = os.path.join(map_dir,'Masterson_2004 Grids','hy layers 1-16','hy_lay0'+str(i+1)+'.tif')
    with rasterio.open(locals()["hy_lay0"+str(i+1)+"_fname"]) as src:
        with WarpedVRT(src,**vrt_options) as vrt:
            locals()["hy_lay0"+str(i+1)] = vrt.read()[0] # [0] selects first band
            locals()["hy_lay0"+str(i+1)][locals()["hy_lay0"+str(i+1)]==vrt.nodata] = np.nan # Should be dem_model.shape = [model_height,model_width]
    # locals()["hk"+str(i+1)] = np.ones((nlay,nrow,ncol), np.float) # for making individual layer variables
    # locals()["hk"+str(i+1)][:,:,:] = locals()["hy_lay0"+str(i+1)]*0.3048 # converted to meters
    hk[i] = locals()["hy_lay0"+str(i+1)]*0.3048 # converted to meters
# hk1[:,:,:] = 10 # same hy everywhere    

for i in range(7):
    locals()["hy_lay"+str(i+10)+"_fname"] = os.path.join(map_dir,'Masterson_2004 Grids','hy layers 1-16','hy_lay'+str(i+10)+'.tif')
    with rasterio.open(locals()["hy_lay"+str(i+10)+"_fname"]) as src:
        with WarpedVRT(src,**vrt_options) as vrt:
            locals()["hy_lay"+str(i+10)] = vrt.read()[0] # [0] selects first band
            locals()["hy_lay"+str(i+10)][locals()["hy_lay"+str(i+10)]==vrt.nodata] = np.nan # Should be dem_model.shape = [model_height,model_width]
    hk[i+9] = locals()["hy_lay"+str(i+10)]*0.3048 # converted to meters
    
for i in range(6):
    locals()["hy_lay"+str(i+17)+"_fname"] = os.path.join(map_dir,'Masterson_2004 Grids','hy layers 17-23','hy_lay'+str(i+17)+'.tif')
    with rasterio.open(locals()["hy_lay"+str(i+17)+"_fname"]) as src:
        with WarpedVRT(src,**vrt_options) as vrt:
            locals()["hy_lay"+str(i+17)] = vrt.read()[0] # [0] selects first band
            locals()["hy_lay"+str(i+17)][locals()["hy_lay"+str(i+17)]==vrt.nodata] = np.nan # Should be dem_model.shape = [model_height,model_width]
    hk[i+16] = locals()["hy_lay"+str(i+17)]*0.3048 # converted to meters

# vka values are initially in feet (vertical hydraulic conductivity)
vka = np.zeros((nlay,nrow,ncol), dtype=np.float32)
for i in range(9):
    locals()["vka_lay0"+str(i+1)+"_fname"] = os.path.join(map_dir,'Masterson_2004 Grids','vka average between layer centers','vka_lay0'+str(i+1)+'.tif')
    with rasterio.open(locals()["vka_lay0"+str(i+1)+"_fname"]) as src:
        with WarpedVRT(src,**vrt_options) as vrt:
            locals()["vka_lay0"+str(i+1)] = vrt.read()[0] # [0] selects first band
            locals()["vka_lay0"+str(i+1)][locals()["vka_lay0"+str(i+1)]==vrt.nodata] = np.nan # Should be dem_model.shape = [model_height,model_width]    
    vka[i] = locals()["vka_lay0"+str(i+1)]*0.3048 # converted to meters
# vka1[:,:,:] = 0.1 # same vka everywhere

for i in range(13):
    locals()["vka_lay"+str(i+10)+"_fname"] = os.path.join(map_dir,'Masterson_2004 Grids','vka average between layer centers','vka_lay'+str(i+10)+'.tif')
    with rasterio.open(locals()["vka_lay"+str(i+10)+"_fname"]) as src:
        with WarpedVRT(src,**vrt_options) as vrt:
            locals()["vka_lay"+str(i+10)] = vrt.read()[0] # [0] selects first band
            locals()["vka_lay"+str(i+10)][locals()["vka_lay"+str(i+10)]==vrt.nodata] = np.nan # Should be dem_model.shape = [model_height,model_width]    
    vka[i+9] = locals()["vka_lay"+str(i+10)]*0.3048 # converted to meters

genu.quick_plot(np.ma.masked_array(hk[0],mask_array!=1),vmin=0,vmax=70)

uncletim_mask = vka[0].copy()
uncletim_mask[uncletim_mask>4] = 0
uncletim_mask[0:690] = 0
uncletim_mask[770::] = 0
uncletim_mask[:,0:440] = 0
uncletim_mask[:,530::] = 0
uncletim_mask[uncletim_mask!=0]=1

hk_reduced = np.delete(hk,np.s_[2:22],axis=0)
# hk_reduced = np.delete(hk_reduced,[0],axis=0)
# hk_reduced[2] = hk[21]
hk_reduced[2] = np.nanmean(np.array(hk[2:23]),axis=0) # bottom layer is average of Masterson's bottom layer
hk_reduced[2][np.isnan(hk_reduced[0])] = np.nan # conform to other layers by making the nan values in the same regions
hk_reduced[1] = np.nanmean(np.array(hk[0:2]),axis=0) # second layer is average of Masterson's first two layers
hk_reduced[1][np.isnan(hk_reduced[0])] = np.nan
# hk_reduced[0][vka_uncletim_mask==1] = hk_reduced[0][vka_uncletim_mask==1]*100
hk_reduced[0][dem_model_hydrofill_mask==1] = np.nanmax(hk[0]) # increase horiz hydraulic cond in hydrofill sections

# increase hk in top 2 layers of non-converging area 
hk_reduced[0][(mask_array_PoorConv==1)&(hk_reduced[0]<1000)] = hk_reduced[0][(mask_array_PoorConv==1)&(hk_reduced[0]<1000)]*10
hk_reduced[1][mask_array_PoorConv==1] = hk_reduced[1][mask_array_PoorConv==1]*10

hk_reduced = np.around(hk_reduced,decimals=2) # attempt to improve convergence through rounding
genu.quick_plot(np.ma.masked_array(hk_reduced[0],mask_array!=1),vmin=0,vmax=70) # plot 1st layer hk after adjusting hydrofill

# The vka values are strangely low in the southern part of the region.
# Can adjust those by taking the average of the nearby land cells (vka[land_cells][~4.5m/day] )
# and the average of the land cells intersecting the low vka areas (vka[land_cells][0.1->0.2]) 
# find the ratio between them, and multiply the low vka areas by that ratio to boost them
# This assumes that Masterson underestimated the vka's in the region which worked for coarser grid resolution.
# Not working - better convergence with Masterson's vkas.

"""
Various masks for adjusting hydraulic conductivities in different regions
"""
# vka_repl_mask_south = vka[0].copy() # mask for low (but not lowest) vkas in southern region
# vka_repl_mask_south[0:837]=0
# vka_lowvka_mask_south = vka_repl_mask_south.copy()
# vka_lowvka_mask_south[(vka_lowvka_mask_south<0.12)&(vka_lowvka_mask_south!=0)]=1
# vka_lowvka_mask_south[vka_lowvka_mask_south!=1]=0
# vka_lowvka_mask_south[:,0:600]=0
# vka_repl_mask_south[(vka_repl_mask_south>=0.12)&(vka_repl_mask_south<=3.3)]=1
# vka_repl_mask_south[vka_repl_mask_south!=1]=0

# vka_se_raisecon_mask = vka_repl_mask_south.copy()
# vka_se_raisecon_mask[:,0:782]=0

# vka_west_sea_mask = vka[0].copy() # combination of south mask and Cheq west mask to find sea cells to replace
# vka_west_sea_mask[mask_array_CheqWest!=1]=0
# vka_west_sea_mask[vka_repl_mask_south!=1]=0
# vka_west_sea_mask[dem_model>mean_sea_level_annual_west[0]]=0
# vka_west_sea_mask[vka_west_sea_mask!=0]=1

# # Make west increase factor (3.34 m/d is the vka on the coast)
# vka_incr_fact_west = 3.34/np.nanmedian(vka[0][vka_west_sea_mask==1])

# vka_east_sea_mask = vka[0].copy() # combination of south mask and Cheq east mask to find sea cells to replace
# vka_east_sea_mask[mask_array_CheqEast!=1]=0
# vka_east_sea_mask[vka_repl_mask_south!=1]=0
# vka_east_sea_mask[dem_model>mean_sea_level_annual_east[0]]=0
# vka_east_sea_mask[:,0:906]=0
# vka_east_sea_mask[(vka_east_sea_mask>0.121)&(vka_east_sea_mask<0.123)]=0
# vka_east_sea_mask[vka_east_sea_mask!=0]=1

# # Make east increase factor (3.34 m/d is the vka on the coast)
# vka_incr_fact_east = 3.34/np.nanmedian(vka[0][vka_east_sea_mask==1])

# vka_mainland_mask = vka[0].copy() # mask for values to use in replacement
# vka_mainland_mask[(vka_west_sea_mask==1)|(vka_east_sea_mask==1)|np.isnan(vka_mainland_mask)]=0
# vka_mainland_mask[(vka_mainland_mask<3.9)|(vka_mainland_mask>4.54)]=0
# vka_mainland_mask[vka_mainland_mask!=0]=1

# # Make land increase factor
# vka_incr_fact = np.nanmedian(vka[0][vka_mainland_mask==1])/np.nanmedian(vka[0][vka_repl_mask_south==1])

# # Increase southern vka's
# vka_top = vka[0].copy()
# vka_top[vka_west_sea_mask==1] = vka_top[vka_west_sea_mask==1]*vka_incr_fact_west
# vka_top[vka_east_sea_mask==1] = vka_top[vka_east_sea_mask==1]*vka_incr_fact_east
# vka_top[(vka_repl_mask_south==1)&(vka_west_sea_mask!=1)&(vka_east_sea_mask!=1)] = vka_top[(vka_repl_mask_south==1)&(vka_west_sea_mask!=1)&(vka_east_sea_mask!=1)]*vka_incr_fact
# vka_top[(vka_repl_mask_south==1)&(vka_top>4.5)]=np.nanmedian(vka[0][vka_mainland_mask==1])
# vka_top[(vka_west_sea_mask==1)] = np.nanmedian(vka_top[(vka_west_sea_mask==1)])
# vka_top[(vka_east_sea_mask==1)] = np.nanmedian(vka_top[(vka_east_sea_mask==1)])
# # vka_top[(dem_model_hydrofill_mask==0)&(vka_se_raisecon_mask==1)] = vka_top[(dem_model_hydrofill_mask==0)&(vka_se_raisecon_mask==1)]*2

# # Decrease vka's in headwater region of southern boundary river
# lowvka_repl_mask_south = vka_repl_mask_south.copy()
# lowvka_repl_mask_south[:,0:784]=0
# lowvka_repl_mask_south[vka_east_sea_mask==1]=0
# lowvka_south_incr_fact = np.nanmedian(vka_top[(vka_lowvka_mask_south==1)])/np.nanmedian(vka_top[(lowvka_repl_mask_south==1)])
# vka_top[(lowvka_repl_mask_south==1)] = vka_top[(lowvka_repl_mask_south==1)]*lowvka_south_incr_fact

genu.quick_plot(np.ma.masked_array(vka[0],mask_array!=1),vmin=0,vmax=10) # layer 1 plot before adjusting hydrofill

vka_reduced = np.delete(vka,np.s_[2:22],axis=0)
# vka_reduced[0] = vka_top
# vka_reduced = np.delete(vka_reduced,[0],axis=0)
# vka_reduced[2] = vka[21]
vka_reduced[2] = np.nanmean(np.array(vka[2:23]),axis=0)
vka_reduced[2][np.isnan(vka_reduced[0])] = np.nan
vka_reduced[1] = np.nanmean(np.array(vka[0:2]),axis=0) # second layer is average of Masterson's first two layers
vka_reduced[1][np.isnan(vka_reduced[0])] = np.nan
# vka_reduced[0][vka_uncletim_mask==1] = vka_reduced[0][vka_uncletim_mask==1]*100
vka_reduced[0][dem_model_hydrofill_mask==1] = np.nanmax(vka_reduced[0]) # increase vert hydraulic cond in hydrofill sections

# increase vka in non-converging area 
vka_reduced[0][(mask_array_PoorConv==1)&(vka_reduced[0]<1000)] = vka_reduced[0][(mask_array_PoorConv==1)&(vka_reduced[0]<1000)]*100
vka_reduced[1][mask_array_PoorConv==1] = vka_reduced[1][mask_array_PoorConv==1]*100

# vka_reduced[0][(dem_model_hydrofill_mask==1)&(vka_repl_mask_south!=1)] = np.nanmax(vka_reduced[0]) # increase vert hydraulic cond in hydrofill sections except in southern region
# vka_reduced[0][(lowvka_repl_mask_south==1)] = np.nanmax(vka_reduced[0]) # set headwater region to high vka
# vka_reduced[0][(vka_lowvka_mask_south==1)] = vka_top[(vka_lowvka_mask_south==1)]# set upstream area to low vka
vka_reduced = np.around(vka_reduced,decimals=2)
genu.quick_plot(np.ma.masked_array(vka_reduced[0],mask_array!=1),vmin=0,vmax=10) # plot vka after adjusting hydrofill

# if using PCG... Add LPF package to the MODFLOW model
# lpf = flopy.modflow.ModflowLpf(mf, laytyp=1, hk=hk1, vka=vka1, ss=1e-05, sy=0.25, storagecoefficient=True, ipakcb=53) # sy and ss from Masterson_2004 p.59
# laytyp = 1 means unconfined, ss is confined storage coefficient (not specific storage) if storagecoefficient=True, sy = specific yield

# if using NWT... Add UPW package to the MODFLOW model
upw = flopy.modflow.ModflowUpw(mf, laytyp=1, hk=hk_reduced, vka=vka_reduced, ss=1e-05, sy=0.25, ipakcb=53) # sy and ss from Masterson_2004 p.59

#%% Transient General-Head Boundary Package and Drain Condition

"""
Example (Added 5/28/19 from Bakker_2016): 
First, we will create the GHB object, which is of the following type: flopy.modflow.ModflowGhb.

The key to creating Flopy transient boundary packages is recognizing that the 
boundary data is stored in a dictionary with key values equal to the 
zero-based stress period number and values equal to the boundary conditions 
for that stress period. For a GHB the values can be a two-dimensional nested 
list of [layer, row, column, stage, conductance]:
"""
# Darcy's law states that
# Q = -KA(h1 - h0)/(X1 - X0)
# Where Q is the flow (L3/T)
# K is the hydraulic conductivity (L/T)
# A is the area perpendicular to flow (L2)
# h is head (L)
# X is the position at which head is measured (L)
# Conductance combines the K, A and X terms so that Darcy's law can be expressed as 
# Q = -C(h1 - h0)
# where C is the conductance (L2/T)
# https://water.usgs.gov/nrp/gwsoftware/modflow2000/MFDOC/index.html?drn.htm

# from Masterson, 2004
# C = KWL/M where
# C is hydraulic conductance of the seabed (ft2/d);
# K is vertical hydraulic conductivity of seabed deposits
# (ft/d);
# W is width of the model cell containing the seabed (ft);
# L is length of the model cell containing the seabed (ft);
# and
# M is thickness of seabed deposits (ft).

# The vertical hydraulic conductivity (K) of the seabed
# deposits in most of the study area was assumed to be 1 ft/d,
# which is consistent with model simulations of similar coastal
# discharge areas in other areas on Cape Cod (Masterson and
# others, 1998). In the area occupied by Salt Pond and Nauset
# Marsh, it was assumed that there were thick deposits of low permeability
# material (J.A. Colman, U.S. Geological Survey,
# oral commun., 2002) and the vertical hydraulic conductivity
# was set to 0.1 ft/d. The thickness of the seabed deposits was
# assumed to be half the thickness of the model cell containing the
# boundary.

# conductance_model = vka[0]*delc*delr/(0.5*(-botm[0])) # (modify 1000 to actual conductance, calculated on p. 55 Master_2004)
conductance_model = vka_reduced[0,:,:]*delc*delr/(0.5*((ztop-zbotm_red_array[0,:,:]))) # (modify 1000 to actual conductance, calculated on p. 55 Master_2004)
# adjust conductance in the problem area to be 1000x higher where it hasn't already been filled (not working, try vka's)
# conductance_model[(mask_array_PoorConv==1)&(conductance_model<1000)] = conductance_model[(mask_array_PoorConv==1)&(conductance_model<1000)]*1000

lrcec = {}    
for i in range(nper): # there are 12 stress periods (nper)
    sea_level_west = msl_ss_monthly_west[i]
    sea_level_east = msl_ss_monthly_east[i]
    river_levels = HR_ss_monthly_levels[i]
    
    stage_cells_west = (mask_array_CheqWest==1) & ~np.isnan(dem_model_nofill) & (dem_model_nofill <= sea_level_west)
    # stage_cells_west[uncletim_mask==1] = 1 # need to drop cell botms to -1.52 here
    stagerows_west, stagecols_west = stage_cells_west.nonzero()
    stage_west = np.zeros_like(dem_model_nofill[stage_cells_west])
    stage_west[:] = sea_level_west
    stage_west_fwh = stage_west*1.025 # freshwater equivalent head (1.025 is the density ratio, https://ngwa.onlinelibrary.wiley.com/doi/full/10.1111/j.1745-6584.2007.00339.x)
    cond_west = conductance_model[stage_cells_west]
    
    stage_cells_east = (mask_array_CheqEast==1) & ~np.isnan(dem_model_nofill) & (dem_model_nofill <= sea_level_east)
    stagerows_east, stagecols_east = stage_cells_east.nonzero()
    stage_east = np.zeros_like(dem_model_nofill[stage_cells_east])
    stage_east[:] = sea_level_east
    stage_east_fwh = stage_east*1.025 # freshwater equivalent head (1.025 is the density ratio, https://ngwa.onlinelibrary.wiley.com/doi/full/10.1111/j.1745-6584.2007.00339.x)
    cond_east = conductance_model[stage_cells_east]
    
    stage_cells_HR = (mask_array_HR==1) & ~np.isnan(dem_model_nofill) & (dem_model_nofill <= river_levels)
    # stage_cells[(mask_array_HR==1)&(ztop<=river_levels)] = river_levels[(mask_array_HR==1)&(ztop<=river_levels)]
    # stage = sea_level-ztop[stage_cells] # this is if stage is relative to sea floor
    stagerows_HR, stagecols_HR = stage_cells_HR.nonzero()
    stage_HR = np.zeros_like(dem_model_nofill[stage_cells_HR])
    stage_HR[:] = river_levels[(mask_array_HR==1)&(dem_model_nofill<=river_levels)]
    cond_HR = conductance_model[stage_cells_HR]
    
    stage_cells = stage_cells_west + stage_cells_east + stage_cells_HR
    stagerows = np.append(np.append(stagerows_west, stagerows_east),stagerows_HR)
    stagecols = np.append(np.append(stagecols_west, stagecols_east),stagecols_HR)
    stage = np.append(np.append(stage_west_fwh, stage_east_fwh),stage_HR)
    conductance_stage = np.append(np.append(cond_west, cond_east),cond_HR)
    
    land_cells = (mask_array==1)&~np.isnan(ztop)&(~stage_cells)
    land_cells_hydrofill = (mask_array==1)&~np.isnan(ztop)&(~stage_cells)&(dem_model_hydrofill_mask==1)
    landrows, landcols = land_cells.nonzero() # removes all 'false' values and assigns 'true' values to rows and columns
    conductance_increased = conductance_model.copy()
    conductance_increased[~land_cells_hydrofill] = conductance_increased[~land_cells_hydrofill]*100 # working okay
    conductance_land = conductance_increased[land_cells] # can increase to improve drainage at land surface
    # conductance_land[conductance_land<10]
    
    lrcec.update({i:np.column_stack([np.zeros_like(landrows),landrows,landcols,ztop[land_cells],conductance_land])}) # this drain will be applied to all stress periods                     
    locals()["bound_sp"+str(i+1)] = np.column_stack([np.zeros_like(stagerows), stagerows, stagecols, stage, conductance_stage]) # boundaries for stress period 1

print('Adding ', len(bound_sp1), 'GHBs for stress period 1 (steady state).')    
lrcsc = {}
for i in range(nper):
    lrcsc.update({i: locals()["bound_sp"+str(i+1)]})
    
ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=lrcsc, options = ['NOPRINT']) # for using general head
# chd = flopy.modflow.ModflowChd(mf, stress_period_data=lrcsc, options = ['NOPRINT']) # for using constant head
drn = flopy.modflow.ModflowDrn(mf, stress_period_data=lrcec, options = ['NOPRINT'])

# Plot Stage Cells and Land Cells
fig,ax = genu.plt.subplots(1,2)
ax[0].set_xlabel('X [m]')
ax[0].set_ylabel('Y [m]')
ax[0].title.set_text('Stage Cells, Final Time Step')
c0=ax[0].pcolormesh(cc[0],cc[1],stage_cells.astype(int)) # in model coordinates
genu.plt.colorbar(c0,ax=ax[0],orientation='horizontal')
ax[0].set_aspect(aspect='equal', adjustable='box')
c1=ax[1].pcolormesh(cc[0],cc[1],land_cells.astype(int)) # in model coordinates
genu.plt.colorbar(c1,ax=ax[1],orientation='horizontal')
fig.gca().set_aspect('equal', adjustable='box')
ax[1].set_xlabel('X [m]')
ax[1].set_ylabel('Y [m]')  
ax[1].title.set_text('Land Cells, Final Time Step') 
ax[0].plot(10*HighToss_colrow[0].astype(int),10*HighToss_colrow[1].astype(int),'ro',markersize=4)
ax[0].annotate('High Toss',(10*HighToss_colrow[0].astype(int),10*HighToss_colrow[1].astype(int)),color='red',xytext=(2700,6500),arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=-0.6',color='red'))
ax[0].annotate('High Toss CTD',(10*HighToss_colrow[0].astype(int),10*HighToss_colrow[1].astype(int)),color='red',xytext=(2700,6500))                                                                                                                  
ax[1].plot(10*HighToss_colrow[0].astype(int),10*HighToss_colrow[1].astype(int),'ro',markersize=4)
ax[1].annotate('High Toss',(10*HighToss_colrow[0].astype(int),10*HighToss_colrow[1].astype(int)),color='red',xytext=(2700,6500),arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=-0.6',color='red'))
ax[1].annotate('High Toss CTD',(10*HighToss_colrow[0].astype(int),10*HighToss_colrow[1].astype(int)),color='red',xytext=(2700,6500))                                                                                                                  

# Plot Conductance
genu.quick_plot(np.ma.masked_array(conductance_model,~stage_cells),vmin=0,vmax=1000) # for stage cells
genu.quick_plot(np.ma.masked_array(conductance_increased,~land_cells),vmin=0,vmax=10000) # for land cells

#%% Add recharge condition

# transient, from Masterson_2004 units in [m/day]?
recharge = np.array([18,18,18,18,6,6,6,6,6,18,18,18]) # inches/year, Jan-Dec
recharge = recharge/12*0.3048/365.25 # meters/day
mean_annual_rech = recharge.mean()
recharge_ss_monthly = np.append(mean_annual_rech, recharge)

# Make recharge a sine wave!

rech = {}
for i in range(nper):
    rech.update({i: recharge_ss_monthly[i]})

rch = flopy.modflow.ModflowRch(mf, nrchop=3, rech=rech) # from USGS Water-Supply Paper 2447, Masterson_1997
                                                            # https://pubs.usgs.gov/wsp/2447/report.pdf
                                                            # nrchop = 3 means recharge to highest active cell

#%% Add OC package to the MODFLOW model

stress_period_data = {}
for kper in range(nper):
    for kstp in range(nstp[kper]):
        stress_period_data[(kper, kstp)] = ['save head',
                                            'save budget',
                                            'print budget']
oc = flopy.modflow.ModflowOc(mf, stress_period_data=stress_period_data, compact=True)

#%% Add Solver package to the MODFLOW model

# PCG package
# pcg = flopy.modflow.ModflowPcg(mf)

# NWT package
nwt = flopy.modflow.ModflowNwt(mf,maxiterout=600,iprnwt=1)

#%% Write the MODFLOW model input files
mf.write_input()

# mf.write_input(SelPackList=['BAS6']) # to run a single package (in this case the bas package)
# mf.write_input(SelPackList=['UPW'])
# mf.write_input(SelPackList=['DIS'])
# mf.write_input(SelPackList=['DRN'])
# mf.write_input(SelPackList=['GHB'])
# mf.write_input(SelPackList=['NWT'])

#%% Run the MODFLOW model
success, mfoutput = mf.run_model(silent=False, pause=False, report=True)
if not success:
    raise Exception('MODFLOW did not terminate normally.')

#%% First Step: Getting heads
    
# plt.subplot(1,1,1,aspect='equal')
hds = bf.HeadFile(os.path.join(work_dir,modelname+'.hds'))
time = 0
head = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
head[head<-100] = np.nan

"""
CHANGE FOR SLR
"""
# head_ss = head
# head_lowslr = head
# head_intslr = head
# head_highslr = head

#%% Load previously saved head files

hds = bf.HeadFile(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Present',modelname+'.hds'))
time = 0
head_ss = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
head_ss[head_ss<-100] = np.nan

hds = bf.HeadFile(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-Low SLR',modelname+'.hds'))
time = 0
head_lowslr = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
head_lowslr[head_lowslr<-100] = np.nan

hds = bf.HeadFile(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-High SLR',modelname+'.hds'))
time = 0
head_intslr = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
head_intslr[head_intslr<-100] = np.nan

hds = bf.HeadFile(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Extreme SLR',modelname+'.hds'))
time = 0
head_highslr = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
head_highslr[head_highslr<-100] = np.nan

#%% Saving to .pdf

mpl.rcParams['pdf.fonttype'] = 42

#%% Save converged first layer heads as .tif

"""
Change File Name for SLR
"""

dx,dy,gt = rastu.get_rast_info(demfname)
projwkt = rastu.load_grid_prj(demfname)
fname = os.path.join(work_dir, 'Cheq_Head_Elevation_Topo_HighSLR.tif')
rastu.write_gdaltif(fname,cc_proj[0],cc_proj[1],head[0],proj_wkt=projwkt)
#%% Validating Model: Compare SS with Well Measurements

# From CTDetc_SW_rawdata.py code

# Well LatLong UTM in meters
Phrag_Wetland_UTM = (412505.44, 4643367.90)
Wet_Shrub_UTM = (412881.77, 4643996.02)
Dry_Forest_UTM = (412407.84, 4644033.50)
Phrag_Wetland_colrow = model_transform_inv*Phrag_Wetland_UTM
Wet_Shrub_colrow = model_transform_inv*Wet_Shrub_UTM
Dry_Forest_colrow = model_transform_inv*Dry_Forest_UTM

# Elevs in NAVD88
head_Phrag_Wetland_ss = head_ss[0,np.around(Phrag_Wetland_colrow[1]).astype(int),np.around(Phrag_Wetland_colrow[0]).astype(int)]
elev_Phrag_Wetland_DEM = dem_model_nofill[np.around(Phrag_Wetland_colrow[1]).astype(int),np.around(Phrag_Wetland_colrow[0]).astype(int)]
elev_Phrag_Wetland_RTK = 0.06026 # meters
head_Wet_Shrub_ss = head_ss[0,np.around(Wet_Shrub_colrow[1]).astype(int),np.around(Wet_Shrub_colrow[0]).astype(int)]
elev_Wet_Shrub_DEM = dem_model_nofill[np.around(Wet_Shrub_colrow[1]).astype(int),np.around(Wet_Shrub_colrow[0]).astype(int)]
elev_Wet_Shrub_RTK = 0.64153 # meters
head_Dry_Forest_ss = head_ss[0,np.around(Dry_Forest_colrow[1]).astype(int),np.around(Dry_Forest_colrow[0]).astype(int)]
elev_Dry_Forest_DEM = dem_model_nofill[np.around(Dry_Forest_colrow[1]).astype(int),np.around(Dry_Forest_colrow[0]).astype(int)]
elev_Dry_Forest_RTK = 0.8332 # meters
head_DogLeg_ss = head_ss[0,np.around(DogLeg_colrow[1]).astype(int),np.around(DogLeg_colrow[0]).astype(int)]
elev_DogLeg_DEM = dem_model_nofill[np.around(DogLeg_colrow[1]).astype(int),np.around(DogLeg_colrow[0]).astype(int)]
head_HighToss_ss = head_ss[0,np.around(HighToss_colrow[1]).astype(int),np.around(HighToss_colrow[0]).astype(int)]
elev_HighToss_DEM = dem_model_nofill[np.around(HighToss_colrow[1]).astype(int),np.around(HighToss_colrow[0]).astype(int)]

head_CNRUS_ss = head_ss[0,np.around(CNRUS_colrow[1]).astype(int),np.around(CNRUS_colrow[0]).astype(int)]
elev_CNRUS_DEM = dem_model_nofill[np.around(CNRUS_colrow[1]).astype(int),np.around(CNRUS_colrow[0]).astype(int)]

locations = np.array(["Phrag Wetland Well", "Wet Shrub Well", "Dry Forest Well", "Dog Leg CTD", "High Toss CTD"])
head_model_ss = np.array([head_Phrag_Wetland_ss, head_Wet_Shrub_ss, head_Dry_Forest_ss, head_DogLeg_ss, head_HighToss_ss])
elev_model = np.array([elev_Phrag_Wetland_DEM, elev_Wet_Shrub_DEM, elev_Dry_Forest_DEM, elev_DogLeg_DEM, elev_HighToss_DEM])
elev_RTK = np.array([elev_Phrag_Wetland_RTK, elev_Wet_Shrub_RTK, elev_Dry_Forest_RTK, np.nan, np.nan])

# WL_NAVD88
# Phrag Wetland June 2017 - December 2018
mean_wl_monthly_Phrag_Wetland = np.array([np.nan,np.nan,np.nan,np.nan, 0.06477857, 0.07941793, 0.06316704, 0.07381617, 0.08366671, 0.09134606, 0.09771469, 0.09073098])
mean_wl_annual_Phrag_Wetland = np.nanmean(mean_wl_monthly_Phrag_Wetland)
# Wet Shrub June 2017 - December 2018
mean_wl_monthly_Wet_Shrub = np.array([np.nan,np.nan,np.nan,np.nan, 0.41522857, 0.40902163, 0.32545714, 0.37099477, 0.42702704, 0.46485835, 0.5305826 , 0.48387548])
mean_wl_annual_Wet_Shrub = np.nanmean(mean_wl_monthly_Wet_Shrub)
# Dry Forest April 2017 - December 2018
mean_wl_monthly_Dry_Forest = np.array([0.21157923, 0.25532835, 0.27953367, 0.20585484, 0.19271088, 0.16619653, 0.07513688, 0.08050746, 0.10113121, 0.14856348, 0.22802049, 0.22048253])            
mean_wl_annual_Dry_Forest = np.nanmean(mean_wl_monthly_Dry_Forest)
# Dog Leg July 2017 - July 2019
mean_wl_monthly_DogLeg = np.array([-0.31951468, np.nan, -0.24964992, -0.19849661, -0.18655801, -0.17858246, -0.25026634, -0.31470557, -0.26982269, -0.24364567, -0.2812699 , -0.28103455])     
mean_wl_annual_DogLeg = np.nanmean(mean_wl_monthly_DogLeg)
# High Toss July 2017 - July 2019
mean_wl_monthly_HighToss = np.array([-0.18695657, np.nan, -0.20624233, -0.12142227, -0.19706912, -0.16469158, -0.19122697, -0.19720732, -0.181539  , -0.16554384, -0.16068823, -0.23582508])            
mean_wl_annual_HighToss = np.nanmean(mean_wl_monthly_HighToss)      

head_wells_ss = np.array([mean_wl_annual_Phrag_Wetland, mean_wl_annual_Wet_Shrub, mean_wl_annual_Dry_Forest, mean_wl_annual_DogLeg, mean_wl_annual_HighToss])

# Plot Heads and Elevations vs. Location for Model and Real Data
plt.figure()
pylab.plot(locations, elev_model, '_', mew=2, ms=40, label='Model Elevation (10m x 10m DEM)')
pylab.plot(locations, elev_RTK, '_', mew=2, ms=40, label='Well Location RTK Measured Elevation')
pylab.plot(locations, head_model_ss, '_', mew=2, ms=40, label='Mean Annual Modeled Heads')
pylab.plot(locations, head_wells_ss, '_', mew=2, ms=40, label='Mean Annual Well/CTD Heads')
pylab.ylabel("Elevation, NAVD88 (m)")
plt.legend()

plt.figure()
pylab.plot(locations, elev_model, label='Model Elevation (10m x 10m DEM)')
pylab.plot(locations, elev_RTK, label='Well Location RTK Measured Elevation')
pylab.plot(locations, head_model_ss, label='Mean Annual Modeled Heads')
pylab.plot(locations, head_wells_ss, label='Mean Annual Well/CTD Heads')
pylab.ylabel("Elevation, NAVD88 (m)")
plt.legend()

# Plot Well Locations Next to Graph of Model Results vs. Well Measurements
fig,ax = genu.plt.subplots(1,2)
ax[0].set_xlabel('column #')
ax[0].set_ylabel('row #')
ax[0].title.set_text('Head, NAVD88 (m)')
genu.quick_plot(head_ss[0],vmin=-0.04,vmax=2.0,ax=ax[0])
genu.quick_plot(np.ma.masked_array(dem_model_nofill,mask_array!=1),vmin=-1,vmax=10,ax=ax[1])
# ax[2].plot(locations, elev_model, '_', mew=2, ms=20, label='Model Elevation (10m x 10m DEM)', rotation='vertical')
# ax[2].plot(locations, elev_RTK, '_', mew=2, ms=20, label='Well Location RTK Measured Elevation')
# ax[2].plot(locations, head_model_ss, '_', mew=2, ms=20, label='Mean Annual Modeled Heads')
# ax[2].plot(locations, head_wells_ss, '_', mew=2, ms=20, label='Mean Annual Well/CTD Heads')
# ax[2].set_ylabel('Elevation, NAVD88 (m)')
# fig.gca().set_aspect('equal', adjustable='box')
ax[1].set_xlabel('column #')
ax[1].set_ylabel('row #')
ax[1].title.set_text('Land Elevations, NAVD88 (m)')
ax[0].plot(Phrag_Wetland_colrow[0],Phrag_Wetland_colrow[1],'ro',markersize=4)
ax[0].annotate('Phrag Wetland',Phrag_Wetland_colrow,color='red',xytext=(240,700),arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=-0.6',color='red'))
ax[0].annotate('Phrag Wetland Well',Phrag_Wetland_colrow,color='red',xytext=(240,700))
ax[0].plot(Wet_Shrub_colrow[0],Wet_Shrub_colrow[1],'ro',markersize=4)
ax[0].annotate('Wet Shrub',Wet_Shrub_colrow,color='red',xytext=(320,630),arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=-0.6',color='red'))
ax[0].annotate('Wet Shrub Well',Wet_Shrub_colrow,color='red',xytext=(320,630))
ax[0].plot(Dry_Forest_colrow[0],Dry_Forest_colrow[1],'ro',markersize=4)
ax[0].annotate('Dry Forest',Dry_Forest_colrow,color='red',xytext=(310,580),arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=-0.6',color='red'))
ax[0].annotate('Dry Forest Well',Dry_Forest_colrow,color='red',xytext=(310,580))
ax[1].plot(Phrag_Wetland_colrow[0],Phrag_Wetland_colrow[1],'ro',markersize=4)
ax[1].annotate('Phrag Wetland',Phrag_Wetland_colrow,color='red',xytext=(240,700),arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=-0.6',color='red'))
ax[1].annotate('Phrag Wetland Well',Phrag_Wetland_colrow,color='red',xytext=(240,700))
ax[1].plot(Wet_Shrub_colrow[0],Wet_Shrub_colrow[1],'ro',markersize=4)
ax[1].annotate('Wet Shrub',Wet_Shrub_colrow,color='red',xytext=(320,630),arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=-0.6',color='red'))
ax[1].annotate('Wet Shrub Well',Wet_Shrub_colrow,color='red',xytext=(320,630))
ax[1].plot(Dry_Forest_colrow[0],Dry_Forest_colrow[1],'ro',markersize=4)
ax[1].annotate('Dry Forest',Dry_Forest_colrow,color='red',xytext=(270,550),arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=-0.6',color='red'))
ax[1].annotate('Dry Forest Well',Dry_Forest_colrow,color='red',xytext=(270,550))
ax[0].plot(DogLeg_colrow[0],DogLeg_colrow[1],'ro',markersize=4)
ax[0].annotate('Dog Leg',DogLeg_colrow,color='red',xytext=(320,650),arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=-0.6',color='red'))
ax[0].annotate('Dog Leg CTD',DogLeg_colrow,color='red',xytext=(320,650))
ax[0].plot(HighToss_colrow[0],HighToss_colrow[1],'ro',markersize=4)
ax[0].annotate('High Toss',HighToss_colrow,color='red',xytext=(270,550),arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=-0.6',color='red'))
ax[0].annotate('High Toss CTD',HighToss_colrow,color='red',xytext=(270,550))


#%% Graphs
"""
Post-Processing the Results
Now that we have successfully built and run our MODFLOW model, we can look at the results. 
MODFLOW writes the simulated heads to a binary data output file. 
We cannot look at these heads with a text editor, but flopy has a binary utility that can be used to read the heads. 
The following statements will read the binary head file and create a plot of simulated heads for layer 1:
"""

"""
CHANGE FOR SLR
"""
# SS
# stage_cells_ss = stage_cells
# stage_cells_west_ss = stage_cells_west
# stage_west_ss = stage_west
# stage_cells_east_ss = stage_cells_east
# stage_east_ss = stage_east

# Low SLR
# stage_cells_lowslr = stage_cells
# stage_cells_west_lowslr = stage_cells_west
# stage_west_lowslr = stage_west
# stage_cells_east_lowslr = stage_cells_east
# stage_east_lowslr = stage_east

# Int SLR
# stage_cells_intslr = stage_cells
# stage_cells_west_intslr = stage_cells_west
# stage_west_intslr = stage_west
# stage_cells_east_intslr = stage_cells_east
# stage_east_intslr = stage_east

# High SLR
# stage_cells_highslr = stage_cells
# stage_cells_west_highslr = stage_cells_west
# stage_west_highslr = stage_west
# stage_cells_east_highslr = stage_cells_east
# stage_east_highslr = stage_east

#%% Save Stage Arrays

np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Present','stage_cells_ss'), stage_cells_ss)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Present','stage_cells_west_ss'), stage_cells_west_ss)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Present','stage_west_ss'), stage_west_ss)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Present','stage_cells_east_ss'), stage_cells_east_ss)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Present','stage_east_ss'), stage_east_ss)

np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-Low SLR','stage_cells_lowslr'), stage_cells_lowslr)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-Low SLR','stage_cells_west_lowslr'), stage_cells_west_lowslr)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-Low SLR','stage_west_lowslr'), stage_west_lowslr)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-Low SLR','stage_cells_east_lowslr'), stage_cells_east_lowslr)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-Low SLR','stage_east_lowslr'), stage_east_lowslr)

np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-High SLR','stage_cells_intslr'), stage_cells_intslr)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-High SLR','stage_cells_west_intslr'), stage_cells_west_intslr)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-High SLR','stage_west_intslr'), stage_west_intslr)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-High SLR','stage_cells_east_intslr'), stage_cells_east_intslr)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-High SLR','stage_east_intslr'), stage_east_intslr)

np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Extreme SLR','stage_cells_highslr'), stage_cells_highslr)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Extreme SLR','stage_cells_west_highslr'), stage_cells_west_highslr)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Extreme SLR','stage_west_highslr'), stage_west_highslr)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Extreme SLR','stage_cells_east_highslr'), stage_cells_east_highslr)
np.save(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Extreme SLR','stage_east_highslr'), stage_east_highslr)

#%% Load Stage Arrays

stage_cells_ss = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Present','stage_cells_ss.npy'))
stage_cells_west_ss = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Present','stage_cells_west_ss.npy')) 
stage_west_ss = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Present','stage_west_ss.npy')) 
stage_cells_east_ss = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Present','stage_cells_east_ss.npy')) 
stage_east_ss = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Present','stage_east_ss.npy')) 

stage_cells_lowslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-Low SLR','stage_cells_lowslr.npy')) 
stage_cells_west_lowslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-Low SLR','stage_cells_west_lowslr.npy')) 
stage_west_lowslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-Low SLR','stage_west_lowslr.npy')) 
stage_cells_east_lowslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-Low SLR','stage_cells_east_lowslr.npy')) 
stage_east_lowslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-Low SLR','stage_east_lowslr.npy')) 

stage_cells_intslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-High SLR','stage_cells_intslr.npy')) 
stage_cells_west_intslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-High SLR','stage_cells_west_intslr.npy')) 
stage_west_intslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-High SLR','stage_west_intslr.npy')) 
stage_cells_east_intslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-High SLR','stage_cells_east_intslr.npy')) 
stage_east_intslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Int-High SLR','stage_east_intslr.npy')) 

stage_cells_highslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Extreme SLR','stage_cells_highslr.npy')) 
stage_cells_west_highslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Extreme SLR','stage_cells_west_highslr.npy')) 
stage_west_highslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Extreme SLR','stage_west_highslr.npy')) 
stage_cells_east_highslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Extreme SLR','stage_cells_east_highslr.npy')) 
stage_east_highslr = np.load(os.path.join('E:\Herring Models\Seasonal\Final_SS_v5\Diked\Extreme SLR','stage_east_highslr.npy')) 

#%% SS Adjustments

# Replace Stage Cells (Remove FW Equivalent head_ss)
head_ss_toplayer = head_ss[0]
head_ss_toplayer[stage_cells_west_ss==True] = stage_west_ss
head_ss_toplayer[stage_cells_east_ss==True] = stage_east_ss

# Depth to W.T.
land_cells_ss = (mask_array==1)&~np.isnan(ztop)&(~stage_cells_ss)
depth_to_wt_ss = (dem_model_nofill-head_ss_toplayer) #*land_cells_ss.astype(int)
# depth_to_wt_ss[depth_to_wt_ss<=0] = np.nan # masks surface water

# what is the point of this hydrofill?
# hydrofill = dem_model_nofill-depth_to_wt_ss
# hydrofill_mask = np.ma.masked_array(hydrofill,hydrofill>0).mask
# depth_to_wt_ss = depth_to_wt_ss*hydrofill_mask
# depth_to_wt_ss[depth_to_wt_ss==0] = np.nan # anywhere depths are exactly zero, assume stage.

genu.quick_plot(depth_to_wt_ss)

dx,dy,gt = rastu.get_rast_info(demfname)
projwkt = rastu.load_grid_prj(demfname)
fname = os.path.join(work_dir, 'Cheq_Head_DBS_Present.tif')
rastu.write_gdaltif(fname,cc_proj[0],cc_proj[1],depth_to_wt_ss,proj_wkt=projwkt)

# Surface Water vs. WT above Surface


# Outline of Chequesset
cheq_border = dem_model_nofill*stage_cells_ss
cheq_border[stage_cells_west_ss] = 0.02 - cheq_border[stage_cells_west_ss]
cheq_border[stage_cells_east_ss] = 0.13 - cheq_border[stage_cells_east_ss]
cheq_border[cheq_border<0.02] = np.nan
cheq_border[cheq_border>1.5] = np.nan
cheq_border[~np.isnan(cheq_border)] = 0
# plot all at once
plt.matshow(cheq_border)
plt.xlabel('Lx')
plt.ylabel('Ly')
head_ssplot = plt.contour(head_ss[0])
plt.colorbar(head_ssplot, label='head_ss (m)') # plots head_ss_sss as contours 
# plt.colorbar.set_label('head_ss_sss')
plt.savefig(os.path.join(work_dir,'CheqModel2a_ss.png'))

# mytimes = np.arange(0, nper, 1) # for plotting nper-1 graphs
# for time in mytimes: 
#     head_ss_ss = hds.get_data(totim=hds.get_times()[time]) # steady-state head_ss_ss (0th time step)
#     head_ss_ss[head_ss_ss<-100] = np.nan
#     Cheqhead_ss_ssplot = genu.quick_plot(head_ss_ss, vmin=0.0, vmax = 2.5) # plots head_ss_sss with color gradient
#     Cheqhead_ss_sss = Cheqhead_ss_ssplot.get_figure()
#     Cheqhead_ss_sss.suptitle(str(time)+" hrs") # how does one label the figure?
#     Cheqhead_ss_sss.savefig(os.path.join(work_dir,"Cheqhead_ss_sssCheqhead_ss_sss0"+str(f"{time:02d}")+".png"))

#%% Low SLR Adjustments

# Replace Stage Cells (Remove FW Equivalent Head)
head_lowslr_toplayer = head_lowslr[0]
head_lowslr_toplayer[stage_cells_west_lowslr==True] = stage_west_lowslr
head_lowslr_toplayer[stage_cells_east_lowslr==True] = stage_east_lowslr

# Depth to W.T.
land_cells_lowslr = (mask_array==1)&~np.isnan(ztop)&(~stage_cells_lowslr)
depth_to_wt_lowslr = (dem_model_nofill-head_lowslr_toplayer)# *land_cells_lowslr.astype(int)
# depth_to_wt_lowslr[depth_to_wt_lowslr<=0] = np.nan
# hydrofill = dem_model_nofill-depth_to_wt_lowslr
# hydrofill_mask = np.ma.masked_array(hydrofill,hydrofill>0).mask
# depth_to_wt_lowslr = depth_to_wt_lowslr*hydrofill_mask
# depth_to_wt_lowslr[depth_to_wt_lowslr==0] = np.nan

# Surface Water vs. WT above Surface


# Outline of Chequesset
cheq_border = dem_model_nofill*stage_cells_lowslr
cheq_border[stage_cells_west_lowslr] = 0.02 - cheq_border[stage_cells_west_lowslr]
cheq_border[stage_cells_east_lowslr] = 0.13 - cheq_border[stage_cells_east_lowslr]
cheq_border[cheq_border<0.02] = np.nan
cheq_border[cheq_border>1.5] = np.nan
cheq_border[~np.isnan(cheq_border)] = 0
# plot all at once
plt.matshow(cheq_border)
plt.xlabel('Lx')
plt.ylabel('Ly')
head_lowslrplot = plt.contour(head_lowslr[0])
plt.colorbar(head_lowslrplot, label='head_lowslr (m)') # plots head_lowslrs as contours 
# plt.colorbar.set_label('head_lowslrs')
plt.savefig(os.path.join(work_dir,'CheqModel2a_lowslr.png'))

# mytimes = np.arange(0, nper, 1) # for plotting nper-1 graphs
# for time in mytimes: 
#     head = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
#     head[head<-100] = np.nan
#     Cheqheadplot = genu.quick_plot(head, vmin=0.0, vmax = 2.5) # plots heads with color gradient
#     Cheqheads = Cheqheadplot.get_figure()
#     Cheqheads.suptitle(str(time)+" hrs") # how does one label the figure?
#     Cheqheads.savefig(os.path.join(work_dir,"CheqHeadsCheqHeads0"+str(f"{time:02d}")+".png"))
    
#%% Intermediate SLR Adjustments

# Replace Stage Cells (Remove FW Equivalent Head)
head_intslr_toplayer = head_intslr[0]
head_intslr_toplayer[stage_cells_west_intslr==True] = stage_west_intslr
head_intslr_toplayer[stage_cells_east_intslr==True] = stage_east_intslr

# Depth to W.T.
land_cells_intslr = (mask_array==1)&~np.isnan(ztop)&(~stage_cells_intslr)
depth_to_wt_intslr = (dem_model_nofill-head_intslr_toplayer) #*land_cells_intslr.astype(int)
# depth_to_wt_intslr[depth_to_wt_intslr<=0] = np.nan
# hydrofill = dem_model_nofill-depth_to_wt_intslr
# hydrofill_mask = np.ma.masked_array(hydrofill,hydrofill>0).mask
# depth_to_wt_intslr = depth_to_wt_intslr*hydrofill_mask
# depth_to_wt_intslr[depth_to_wt_intslr==0] = np.nan

# Surface Water vs. WT above Surface


# Outline of Chequesset
cheq_border = dem_model_nofill*stage_cells_intslr
cheq_border[stage_cells_west_intslr] = 0.02 - cheq_border[stage_cells_west_intslr]
cheq_border[stage_cells_east_intslr] = 0.13 - cheq_border[stage_cells_east_intslr]
cheq_border[cheq_border<0.02] = np.nan
cheq_border[cheq_border>1.5] = np.nan
cheq_border[~np.isnan(cheq_border)] = 0
# plot all at once
plt.matshow(cheq_border)
plt.xlabel('Lx')
plt.ylabel('Ly')
head_intslrplot = plt.contour(head_intslr[0])
plt.colorbar(head_intslrplot, label='head_intslr (m)') # plots head_intslrs as contours 
# plt.colorbar.set_label('head_intslrs')
plt.savefig(os.path.join(work_dir,'CheqModel2a_intslr.png'))

# mytimes = np.arange(0, nper, 1) # for plotting nper-1 graphs
# for time in mytimes: 
#     head = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
#     head[head<-100] = np.nan
#     Cheqheadplot = genu.quick_plot(head, vmin=0.0, vmax = 2.5) # plots heads with color gradient
#     Cheqheads = Cheqheadplot.get_figure()
#     Cheqheads.suptitle(str(time)+" hrs") # how does one label the figure?
#     Cheqheads.savefig(os.path.join(work_dir,"CheqHeadsCheqHeads0"+str(f"{time:02d}")+".png"))

#%% High SLR Adjustments

# Replace Stage Cells (Remove FW Equivalent Head)
head_highslr_toplayer = head_highslr[0]
head_highslr_toplayer[stage_cells_west_highslr==True] = stage_west_highslr
head_highslr_toplayer[stage_cells_east_highslr==True] = stage_east_highslr

# Depth to W.T.
land_cells_highslr = (mask_array==1)&~np.isnan(ztop)&(~stage_cells_highslr)
depth_to_wt_highslr = (dem_model_nofill-head_highslr_toplayer) #*land_cells_highslr.astype(int)
# depth_to_wt_highslr[depth_to_wt_highslr<=0] = np.nan
# hydrofill = dem_model_nofill-depth_to_wt_highslr
# hydrofill_mask = np.ma.masked_array(hydrofill,hydrofill>0).mask
# depth_to_wt_highslr = depth_to_wt_highslr*hydrofill_mask
# depth_to_wt_highslr[depth_to_wt_highslr==0] = np.nan

# Surface Water vs. WT above Surface


# Outline of Chequesset
cheq_border = dem_model_nofill*stage_cells_highslr
cheq_border[stage_cells_west_highslr] = 0.02 - cheq_border[stage_cells_west_highslr]
cheq_border[stage_cells_east_highslr] = 0.13 - cheq_border[stage_cells_east_highslr]
cheq_border[cheq_border<0.02] = np.nan
cheq_border[cheq_border>1.5] = np.nan
cheq_border[~np.isnan(cheq_border)] = 0
# plot all at once
plt.matshow(cheq_border)
plt.xlabel('Lx')
plt.ylabel('Ly')
head_highslrplot = plt.contour(head_highslr[0])
plt.colorbar(head_highslrplot, label='head_highslr (m)') # plots head_highslrs as contours 
# plt.colorbar.set_label('head_highslrs')
plt.savefig(os.path.join(work_dir,'CheqModel2a_highslr.png'))

# mytimes = np.arange(0, nper, 1) # for plotting nper-1 graphs
# for time in mytimes: 
#     head = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
#     head[head<-100] = np.nan
#     Cheqheadplot = genu.quick_plot(head, vmin=0.0, vmax = 2.5) # plots heads with color gradient
#     Cheqheads = Cheqheadplot.get_figure()
#     Cheqheads.suptitle(str(time)+" hrs") # how does one label the figure?
#     Cheqheads.savefig(os.path.join(work_dir,"CheqHeadsCheqHeads0"+str(f"{time:02d}")+".png"))    
    
#%% Graphs of boundary conditions (GHB) on HR changing for SLR

mean_HR_levels_present=(0.136838)*(disp_CNRUS_sensor**2)+(-0.136449)*(disp_CNRUS_sensor)+(-0.291329)

mean_HR_levels_lowslr = mean_HR_levels_present.copy()
mean_HR_levels_lowslr[:,:] = mean_HR_level_CNRUS + 0.5

mean_HR_levels_intslr = mean_HR_levels_present.copy()
mean_HR_levels_intslr[:,:] = np.array([mean_sea_level_monthly_west.mean()]) + 2.5

mean_HR_levels_highslr = mean_HR_levels_present.copy()
mean_HR_levels_highslr[:,:] = np.array([mean_sea_level_monthly_west.mean()]) + 3.0

mean_HR_level_CNRUS_present = mean_HR_levels_present[int(np.around(CNRUS_colrow[1])),int(np.around(CNRUS_colrow[0]))]
mean_HR_level_DogLeg_present = mean_HR_levels_present[int(np.around(DogLeg_colrow[1])),int(np.around(DogLeg_colrow[0]))]
mean_HR_level_HighToss_present = mean_HR_levels_present[int(np.around(HighToss_colrow[1])),int(np.around(HighToss_colrow[0]))]

mean_HR_level_CNRUS_lowslr = mean_HR_levels_lowslr[int(np.around(CNRUS_colrow[1])),int(np.around(CNRUS_colrow[0]))]
mean_HR_level_DogLeg_lowslr = mean_HR_levels_lowslr[int(np.around(DogLeg_colrow[1])),int(np.around(DogLeg_colrow[0]))]
mean_HR_level_HighToss_lowslr = mean_HR_levels_lowslr[int(np.around(HighToss_colrow[1])),int(np.around(HighToss_colrow[0]))]

mean_HR_level_CNRUS_intslr = mean_HR_levels_intslr[int(np.around(CNRUS_colrow[1])),int(np.around(CNRUS_colrow[0]))]
mean_HR_level_DogLeg_intslr = mean_HR_levels_intslr[int(np.around(DogLeg_colrow[1])),int(np.around(DogLeg_colrow[0]))]
mean_HR_level_HighToss_intslr = mean_HR_levels_intslr[int(np.around(HighToss_colrow[1])),int(np.around(HighToss_colrow[0]))]

mean_HR_level_CNRUS_highslr = mean_HR_levels_highslr[int(np.around(CNRUS_colrow[1])),int(np.around(CNRUS_colrow[0]))]
mean_HR_level_DogLeg_highslr = mean_HR_levels_highslr[int(np.around(DogLeg_colrow[1])),int(np.around(DogLeg_colrow[0]))]
mean_HR_level_HighToss_highslr = mean_HR_levels_highslr[int(np.around(HighToss_colrow[1])),int(np.around(HighToss_colrow[0]))]

disp_HighToss = disp_CNRUS_sensor[int(np.around(HighToss_colrow[1])),int(np.around(HighToss_colrow[0]))]
disp_DogLeg = disp_CNRUS_sensor[int(np.around(DogLeg_colrow[1])),int(np.around(DogLeg_colrow[0]))]
disp_CNRUS = 0

disp_HRghb = np.array([disp_CNRUS, disp_DogLeg, disp_HighToss])
levels_HRghb = np.array([(mean_HR_level_CNRUS_present, mean_HR_level_DogLeg_present, mean_HR_level_HighToss_present),
                 (mean_HR_level_CNRUS_lowslr, mean_HR_level_DogLeg_lowslr, mean_HR_level_HighToss_lowslr),
                 (mean_HR_level_CNRUS_intslr, mean_HR_level_DogLeg_intslr, mean_HR_level_HighToss_intslr),
                 (mean_HR_level_CNRUS_highslr, mean_HR_level_DogLeg_highslr, mean_HR_level_HighToss_highslr)])

number = 4
cmap = plt.get_cmap('Reds')
color = [cmap(i) for i in np.linspace(0.5,0.8,number)]
    
for i in range(len(levels_HRghb)):
    plt.scatter(disp_HRghb, levels_HRghb[i], color=color[i])
    idx_levels_HRghb = np.isfinite(disp_HRghb) & np.isfinite(levels_HRghb[i])
    z_levels_HRghb = np.polyfit(disp_HRghb[idx_levels_HRghb], levels_HRghb[i][idx_levels_HRghb], 2)
    p_levels_HRghb = np.poly1d(z_levels_HRghb)
    polyX_levels_HRghb = np.linspace(disp_HRghb.min(), disp_HRghb.max(), 100)
    pylab.plot(polyX_levels_HRghb,p_levels_HRghb(polyX_levels_HRghb), color=color[i])
    
pylab.plot(-0.1, np.array([mean_sea_level_monthly_west.mean()]), color=color[0], marker='_', mew=2, ms=40, label='Present Mean Sea Level')
pylab.plot(-0.1, np.array([mean_sea_level_monthly_west.mean()])+0.5, color=color[1], marker='_', mew=2, ms=40, label='Mean Sea Level, 0.5 m SLR')
pylab.plot(-0.1, np.array([mean_sea_level_monthly_west.mean()])+1.5, color=color[2], marker='_', mew=2, ms=40, label='Mean Sea Level, 1.5 m SLR')
pylab.plot(-0.1, np.array([mean_sea_level_monthly_west.mean()])+2.5, color=color[3], marker='_', mew=2, ms=40, label='Mean Sea Level, 2.5 m SLR')
pylab.ylabel("Elevation, NAVD88 (m)", fontsize=26)
plt.ylim(ymin=-0.5,ymax=3.5)
plt.legend(fontsize=22)    

# for xe, ye in zip(disp_HRghb, levels_HRghb):
#     plt.scatter([xe] * len(ye), ye)
    
plt.xticks(disp_HRghb)
plt.axes().set_xticklabels(['CNR U/S', 'Dog Leg', 'High Toss'])

#%% Histograms of WT Depth in Chequesset region

depth_to_wt_ss_all = depth_to_wt_ss[~np.isnan(depth_to_wt_ss)]
depth_to_wt_lowslr_all = depth_to_wt_lowslr[~np.isnan(depth_to_wt_lowslr)]
depth_to_wt_intslr_all = depth_to_wt_intslr[~np.isnan(depth_to_wt_intslr)]
depth_to_wt_highslr_all = depth_to_wt_highslr[~np.isnan(depth_to_wt_highslr)]

nonzero_depth_to_wt_ss_count = np.count_nonzero(depth_to_wt_ss_all)
nonzero_depth_to_wt_lowslr_count = np.count_nonzero(depth_to_wt_lowslr_all)
nonzero_depth_to_wt_intslr_count = np.count_nonzero(depth_to_wt_intslr_all)
nonzero_depth_to_wt_highslr_count = np.count_nonzero(depth_to_wt_highslr_all)

depth_to_wt_ss_all_hist, depth_to_wt_ss_all_bins = np.histogram(depth_to_wt_ss_all, bins=np.arange(-2,5.5,0.5))
nonzero_depth_to_wt_ss_count_subfive = np.sum(depth_to_wt_ss_all_hist)
depth_to_wt_lowslr_all_hist, depth_to_wt_lowslr_all_bins = np.histogram(depth_to_wt_lowslr_all, bins=np.arange(-2,5.5,0.5))
nonzero_depth_to_wt_lowslr_count_subfive = np.sum(depth_to_wt_ss_all_hist)
depth_to_wt_intslr_all_hist, depth_to_wt_intslr_all_bins = np.histogram(depth_to_wt_intslr_all, bins=np.arange(-2,5.5,0.5))
nonzero_depth_to_wt_intslr_count_subfive = np.sum(depth_to_wt_ss_all_hist)
depth_to_wt_highslr_all_hist, depth_to_wt_highslr_all_bins = np.histogram(depth_to_wt_highslr_all, bins=np.arange(-2,5.5,0.5))
nonzero_depth_to_wt_highslr_count_subfive = np.sum(depth_to_wt_ss_all_hist)

depth_to_wt_ss_all_relfreq = np.zeros_like(depth_to_wt_ss_all_hist,dtype=float)
depth_to_wt_lowslr_all_relfreq = np.zeros_like(depth_to_wt_lowslr_all_hist,dtype=float)
depth_to_wt_intslr_all_relfreq = np.zeros_like(depth_to_wt_intslr_all_hist,dtype=float)
depth_to_wt_highslr_all_relfreq = np.zeros_like(depth_to_wt_highslr_all_hist,dtype=float)
# Probability of occurence
for i in range(len(depth_to_wt_ss_all_hist)):
    depth_to_wt_ss_all_relfreq[i] = depth_to_wt_ss_all_hist[i]/nonzero_depth_to_wt_ss_count_subfive
    depth_to_wt_lowslr_all_relfreq[i] = depth_to_wt_lowslr_all_hist[i]/nonzero_depth_to_wt_lowslr_count_subfive
    depth_to_wt_intslr_all_relfreq[i] = (depth_to_wt_intslr_all_hist[i]/nonzero_depth_to_wt_intslr_count_subfive)
    depth_to_wt_highslr_all_relfreq[i] = (depth_to_wt_highslr_all_hist[i]/nonzero_depth_to_wt_highslr_count_subfive)

# width = 0.7 * (depth_to_wt_ss_all_bins[1] - depth_to_wt_ss_all_bins[0])
# center = (depth_to_wt_ss_all_bins[:-1] + depth_to_wt_ss_all_bins[1:]) / 2
# plt.bar(center, depth_to_wt_ss_all_relfreq, align='center', width=width)
# plt.show()
# fig, ax = plt.subplots()
# ax.bar(center, depth_to_wt_ss_all_relfreq, align='center', width=width)

# fig.savefig("1.png")

# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)
# bins = [0, 40, 60, 75, 90, 110, 125, 140, 160, 200]
# hist, bins = np.histogram(x, bins=bins)
# width = np.diff(bins)
# center = (bins[:-1] + bins[1:]) / 2

# fig, ax = plt.subplots(figsize=(8,3))
# ax.bar(center, hist, align='center', width=width)
# ax.set_xticks(bins)
# fig.savefig("/tmp/out.png")

# plt.show()

# plt.figure()
# plt.hist(depth_to_wt_ss_SLAMM,range=(np.nanmin(depth_to_wt_highslr[mask_array_SLAMM!=0]),np.nanmax(depth_to_wt_ss[mask_array_SLAMM!=0])),label='Present Day')
# plt.hist(depth_to_wt_lowslr_SLAMM,range=(np.nanmin(depth_to_wt_highslr[mask_array_SLAMM!=0]),np.nanmax(depth_to_wt_ss[mask_array_SLAMM!=0])),label='0.5 Meter SLR (m)')
# plt.hist(depth_to_wt_intslr_SLAMM,range=(np.nanmin(depth_to_wt_highslr[mask_array_SLAMM!=0]),np.nanmax(depth_to_wt_ss[mask_array_SLAMM!=0])),label='1.5 Meter SLR (m)')
# plt.hist(depth_to_wt_highslr_SLAMM,range=(np.nanmin(depth_to_wt_highslr[mask_array_SLAMM!=0]),np.nanmax(depth_to_wt_ss[mask_array_SLAMM!=0])),label='2.5 Meter SLR (m)')
# plt.legend()
# plt.xlabel('depth to wt')
# plt.ylabel('# occurences')

# attempting to plot histogram bars side by side.    
number = 4
cmap = plt.get_cmap('Reds')
color = [cmap(i) for i in np.linspace(0.5,0.8,number)] 
           
ax = plt.subplot(111)
barWidth = 0.1
r1 = np.arange(-1.9,5.1,0.5)
r2 = [x+barWidth for x in r1]
r3 = [x+barWidth for x in r2]
r4 = [x+barWidth for x in r3]
ax.bar(r1,depth_to_wt_ss_all_hist, color=color[0], width=0.1, label='Present Day')
ax.bar(r2,depth_to_wt_lowslr_all_hist, color=color[1], width=0.1, label='0.5 Meter SLR')
ax.bar(r3,depth_to_wt_intslr_all_hist, color=color[2], width=0.1, label='1.5 Meter SLR')
ax.bar(r4,depth_to_wt_highslr_all_hist, color=color[3], width=0.1, label='2.5 Meter SLR')
plt.legend(fontsize=22)
plt.xlabel('Depth to WT Bin End (m)',fontsize=26)
plt.xticks([r+0.4 for r in r1])
plt.ylabel('Counts',fontsize=26)

#%% Histograms of WT Depth in HR region

depth_to_wt_ss_SLAMM_wNaN = depth_to_wt_ss[mask_array_SLAMM!=0]
depth_to_wt_ss_SLAMM = depth_to_wt_ss_SLAMM_wNaN[~np.isnan(depth_to_wt_ss_SLAMM_wNaN)]
depth_to_wt_lowslr_SLAMM_wNaN = depth_to_wt_lowslr[mask_array_SLAMM!=0]
depth_to_wt_lowslr_SLAMM = depth_to_wt_lowslr_SLAMM_wNaN[~np.isnan(depth_to_wt_lowslr_SLAMM_wNaN)]
depth_to_wt_intslr_SLAMM_wNaN = depth_to_wt_intslr[mask_array_SLAMM!=0]
depth_to_wt_intslr_SLAMM = depth_to_wt_intslr_SLAMM_wNaN[~np.isnan(depth_to_wt_intslr_SLAMM_wNaN)]
depth_to_wt_highslr_SLAMM_wNaN = depth_to_wt_highslr[mask_array_SLAMM!=0]
depth_to_wt_highslr_SLAMM = depth_to_wt_highslr_SLAMM_wNaN[~np.isnan(depth_to_wt_highslr_SLAMM_wNaN)]

nonzero_depth_to_wt_ss_HRcount = np.count_nonzero(depth_to_wt_ss_SLAMM)
nonzero_depth_to_wt_lowslr_HRcount = np.count_nonzero(depth_to_wt_lowslr_SLAMM)
nonzero_depth_to_wt_intslr_HRcount = np.count_nonzero(depth_to_wt_intslr_SLAMM)
nonzero_depth_to_wt_highslr_HRcount = np.count_nonzero(depth_to_wt_highslr_SLAMM)

depth_to_wt_ss_SLAMM_hist = np.histogram(depth_to_wt_ss_SLAMM, bins=np.arange(0,2.2,0.2))
nonzero_depth_to_wt_ss_HRcount_subtwo = np.sum(depth_to_wt_ss_SLAMM_hist[0])
depth_to_wt_lowslr_SLAMM_hist = np.histogram(depth_to_wt_lowslr_SLAMM, bins=np.arange(0,2.2,0.2))
nonzero_depth_to_wt_lowslr_HRcount_subtwo = np.sum(depth_to_wt_lowslr_SLAMM_hist[0])
depth_to_wt_intslr_SLAMM_hist = np.histogram(depth_to_wt_intslr_SLAMM, bins=np.arange(0,2.2,0.2))
nonzero_depth_to_wt_intslr_HRcount_subtwo = np.sum(depth_to_wt_intslr_SLAMM_hist[0])
depth_to_wt_highslr_SLAMM_hist = np.histogram(depth_to_wt_highslr_SLAMM, bins=np.arange(0,2.2,0.2))
nonzero_depth_to_wt_highslr_HRcount_subtwo = np.sum(depth_to_wt_highslr_SLAMM_hist[0])

# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)
# hist, bins = np.histogram(x, bins=50)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.show()
# fig, ax = plt.subplots()
# ax.bar(center, hist, align='center', width=width)
# fig.savefig("1.png")
# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)
# bins = [0, 40, 60, 75, 90, 110, 125, 140, 160, 200]
# hist, bins = np.histogram(x, bins=bins)
# width = np.diff(bins)
# center = (bins[:-1] + bins[1:]) / 2

# fig, ax = plt.subplots(figsize=(8,3))
# ax.bar(center, hist, align='center', width=width)
# ax.set_xticks(bins)
# fig.savefig("/tmp/out.png")

# plt.show()


plt.figure()
plt.hist(depth_to_wt_ss_SLAMM,range=(np.nanmin(depth_to_wt_highslr[mask_array_SLAMM!=0]),np.nanmax(depth_to_wt_ss[mask_array_SLAMM!=0])),label='Present Day')
plt.hist(depth_to_wt_lowslr_SLAMM,range=(np.nanmin(depth_to_wt_highslr[mask_array_SLAMM!=0]),np.nanmax(depth_to_wt_ss[mask_array_SLAMM!=0])),label='0.5 Meter SLR (m)')
plt.hist(depth_to_wt_intslr_SLAMM,range=(np.nanmin(depth_to_wt_highslr[mask_array_SLAMM!=0]),np.nanmax(depth_to_wt_ss[mask_array_SLAMM!=0])),label='1.5 Meter SLR (m)')
plt.hist(depth_to_wt_highslr_SLAMM,range=(np.nanmin(depth_to_wt_highslr[mask_array_SLAMM!=0]),np.nanmax(depth_to_wt_ss[mask_array_SLAMM!=0])),label='2.5 Meter SLR (m)')
plt.legend()
plt.xlabel('depth to wt')
plt.ylabel('# occurences')

# attempting to plot histogram bars side by side.          
ax = plt.subplot(111)
barWidth = 0.25
r1 = np.arange(10)
r2 = [x+barWidth for x in r1]
r3 = [x+barWidth for x in r2]
r4 = [x+barWidth for x in r3]
nonzero_depth_to_wt_ss_HRcount = np.count_nonzero(depth_to_wt_ss[mask_array_SLAMM!=0])
ax.hist(r1,depth_to_wt_ss[mask_array_SLAMM!=0],range=(np.nanmin(depth_to_wt_highslr[mask_array_SLAMM!=0]),np.nanmax(depth_to_wt_ss[mask_array_SLAMM!=0])),label='Present Day')
nonzero_depth_to_wt_lowslr_HRcount = np.count_nonzero(depth_to_wt_lowslr[mask_array_SLAMM!=0])
ax.hist(r2,depth_to_wt_lowslr[mask_array_SLAMM!=0],range=(np.nanmin(depth_to_wt_highslr[mask_array_SLAMM!=0]),np.nanmax(depth_to_wt_ss[mask_array_SLAMM!=0])),label='0.5 Meter SLR (m)')
nonzero_depth_to_wt_intslr_HRcount = np.count_nonzero(depth_to_wt_intslr[mask_array_SLAMM!=0])
ax.hist(r3,depth_to_wt_intslr[mask_array_SLAMM!=0],range=(np.nanmin(depth_to_wt_highslr[mask_array_SLAMM!=0]),np.nanmax(depth_to_wt_ss[mask_array_SLAMM!=0])),label='1.5 Meter SLR (m)')
nonzero_depth_to_wt_highslr_HRcount = np.count_nonzero(depth_to_wt_highslr[mask_array_SLAMM!=0])
ax.hist(r4,depth_to_wt_highslr[mask_array_SLAMM!=0],range=(np.nanmin(depth_to_wt_highslr[mask_array_SLAMM!=0]),np.nanmax(depth_to_wt_ss[mask_array_SLAMM!=0])),label='2.5 Meter SLR (m)')
plt.legend()
plt.xlabel('depth to wt')
plt.xticks([r+barWidth for r in range(len(10))])
plt.ylabel('# occurences')

#%% Steps along 1 day sea-level plot

# mytimes = np.arange(24, nper, 1) # for plotting nper-1 graphs
# for time in mytimes:
#     plt.figure()
#     plt.plot(meas_times_hrs, sea_levels) # plots sea level changing hourly over 40 time periods.
#     plt.xlabel('time (hrs)')
#     plt.ylabel('sea level (m)')
#     plt.axis([24, nper, -0.5, 2.0])
#     plt.grid(True)
#     plt.plot(int(time),sea_levels[int(time)],'ro') # add red dot to plot
#     # plt.title(str(time)+" hrs") # how does one label the figure? not using title for these since the stepping is on the x-axis.
#     plt.savefig(os.path.join(work_dir,"CheqSeaLevelCheqSeaLevel0"+str(f"{time:02d}")+".png"))
#     plt.show()    
    
#%% Model results post-processing from Hatari Labs
    # https://www.hatarilabs.com/ih-en/regional-groundwater-modeling-with-modflow-and-flopy-tutorial

# Model grid representation
# First step is to set up the plot
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, aspect='equal')

# Next we create an instance of the ModelMap class
modelmap = flopy.plot.ModelMap(sr=mf.dis.sr)

# Then we can use the plot_grid() method to draw the grid
# The return value for this function is a matplotlib LineCollection object,
# which could be manipulated (or used) later if necessary.
linecollection = modelmap.plot_grid(linewidth=0.4)

# Cross section of model grid representation
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(1, 1, 1)
# Next we create an instance of the ModelCrossSection class
#modelxsect = flopy.plot.ModelCrossSection(model=ml, line={'Column': 5})
modelxsect = flopy.plot.ModelCrossSection(model=mf, line={'Row': 99}) # ValueError: Axis limits cannot be NaN or Inf

# Then we can use the plot_grid() method to draw the grid
# The return value for this function is a matplotlib LineCollection object,
# which could be manipulated (or used) later if necessary.
linecollection = modelxsect.plot_grid(linewidth=0.4)
t = ax.set_title('Column 6 Cross-Section - Model Grid')

# Active/inactive cells on model extension
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(model=mf, rotation=0)
quadmesh = modelmap.plot_ibound(color_noflow='cyan')
linecollection = modelmap.plot_grid(linewidth=0.4)

# Cross sections of active/inactive cells
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(1, 1, 1)
modelxsect = flopy.plot.ModelCrossSection(model=mf, line={'Column': 5}) # ValueError: Axis limits cannot be NaN or Inf
patches = modelxsect.plot_ibound(color_noflow='cyan')
linecollection = modelxsect.plot_grid(linewidth=0.4)
t = ax.set_title('Column 6 Cross-Section with IBOUND Boundary Conditions')

# Channel network as drain (DRN) package
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(model=mf, rotation=-14)
quadmesh = modelmap.plot_ibound(color_noflow='cyan')
quadmesh = modelmap.plot_bc('DRN', color='blue')
linecollection = modelmap.plot_grid(linewidth=0.4)

# Model grid and heads representation
fname = os.path.join(work_dir, 'CheqModel2.hds')
hdobj = flopy.utils.HeadFile(fname)
head = hdobj.get_data()

fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(1, 2, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(model=mf, rotation=-14)
#quadmesh = modelmap.plot_ibound()
quadmesh = modelmap.plot_array(head, masked_values=[-2.e+20], alpha=0.8)
linecollection = modelmap.plot_grid(linewidth=0.2)

#%% Animating Images

# img = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png',
#        '7.png', '8.png', '9.png', '10.png', '11.png']

# clips = [ImageClip(m).set_duration(0.5)
#       for m in img]

# concat_clip = concatenate_videoclips(clips, method="compose")
# concat_clip.write_videofile("test.mp4", fps=24)


# # Extract elevation data at pond
# pond_elev=dem_xr.data[0]
# pond_elev = pond_elev[pond_mask]
# median_pond_elev = np.median(pond_elev)

# # Export mask as shapefile

# out_shp_fname = os.path.join(cape_dir,'data','Coast-0.shp')

# nrow,ncol,gt = pgwl.get_rast_info(demfname)

# to_poly_dict = {'Z':all_ponds_mask,'in_proj':prj,'out_shp':out_shp_fname,'gt':gt}
# pgwl.raster_to_polygon_gdal(**to_poly_dict)

# # Filename example

# file_fmt = 'land_sl{0:4.2f}mbpsl_time{1:3.2f}kBP.shp'

# sl = -13.23
# time= -6.5

# out_filename = file_fmt.format(abs(sl),abs(time))
# print(out_filename)

# Make a movie of head map (.png files must be saved with Tools>Preferences>IPython Console>Graphics>Backend set to Automatic),
    # otherwise the files will not be the same size
os.chdir(r'E:HerringCheqHeads')   
img = os.listdir(r'E:HerringCheqHeads')

clip = mpy.ImageSequenceClip(img, fps=2)
os.chdir(r'E:HerringCheqHeads') 
clip.write_gif('CheqHeads_oneday_unconf_48.gif', fps=12)

# Make a movie of sea levels
os.chdir(r'E:HerringCheqSeaLevel')   
img = os.listdir(r'E:HerringCheqSeaLevel')

clip = mpy.ImageSequenceClip(img, fps=2)
os.chdir(r'E:HerringCheqSeaLevel') 
clip.write_gif('CheqSeaLevel_oneday_unconf_48.gif', fps=12)

#%% Model SW, Present

sw_levels_present = head_ss[0].copy()
sw_levels_present[sw_levels_present-dem_model_nofill<0] = np.nan
sw_levels_present[~np.isnan(sw_levels_present)] = sw_levels_present[~np.isnan(sw_levels_present)]-dem_model_nofill[~np.isnan(sw_levels_present)]
plt.matshow(sw_levels_present)
cbar = plt.colorbar(location='bottom')
cbar.set_label("Modeled Surface Water Depth (Diked) [m]")

dx,dy,gt = rastu.get_rast_info(demfname_nofill)
projwkt = rastu.load_grid_prj(demfname_nofill)
fname_sw_present = os.path.join(work_dir,'Cheq_modelSW_diked.tif')
rastu.write_gdaltif(fname_sw_present,cc_proj[0],cc_proj[1],sw_levels_present,proj_wkt=projwkt)

sw_levels_present_mask = sw_levels_present.copy()
sw_levels_present_mask[~np.isnan(sw_levels_present)] = 1
genu.quick_plot(sw_levels_present_mask)

# Remove the ocean stage cells
sw_levels_present_mask_fw = sw_levels_present_mask.copy()
sw_levels_present_mask_fw[stage_cells_west_ss] = np.nan
sw_levels_present_mask_fw[stage_cells_east_ss] = np.nan

np.count_nonzero(~np.isnan(sw_levels_present_mask_fw))

#%% Emergent GW

# Heads with Surface Water Masked, Present
head_ss_sw_masked = head_ss[0].copy()
head_ss_sw_masked[head_ss_sw_masked>dem_model_nofill] = np.nan

emergent_gw_lowslr = head_lowslr[0].copy()
emergent_gw_lowslr[emergent_gw_lowslr<=dem_model_nofill] = np.nan
emergent_gw_lowslr[np.isnan(head_ss_sw_masked)] = np.nan
emergent_gw_lowslr[~np.isnan(emergent_gw_lowslr)] = emergent_gw_lowslr[~np.isnan(emergent_gw_lowslr)] - dem_model_nofill[~np.isnan(emergent_gw_lowslr)]
np.count_nonzero(~np.isnan(emergent_gw_lowslr))

emergent_gw_intslr = head_intslr[0].copy()
emergent_gw_intslr[emergent_gw_intslr<=dem_model_nofill] = np.nan
emergent_gw_intslr[np.isnan(head_ss_sw_masked)] = np.nan
emergent_gw_intslr[~np.isnan(emergent_gw_intslr)] = emergent_gw_intslr[~np.isnan(emergent_gw_intslr)] - dem_model_nofill[~np.isnan(emergent_gw_intslr)]
np.count_nonzero(~np.isnan(emergent_gw_intslr))

emergent_gw_highslr = head_highslr[0].copy()
emergent_gw_highslr[emergent_gw_highslr<=dem_model_nofill] = np.nan
emergent_gw_highslr[np.isnan(head_ss_sw_masked)] = np.nan
emergent_gw_highslr[~np.isnan(emergent_gw_highslr)] = emergent_gw_highslr[~np.isnan(emergent_gw_highslr)] - dem_model_nofill[~np.isnan(emergent_gw_highslr)]
np.count_nonzero(~np.isnan(emergent_gw_highslr))

# Plots of emergent gw
fig,ax = genu.plt.subplots(1,3)
ax[0].set_xlabel('column #')
ax[0].set_ylabel('row #')
ax[0].title.set_text('0.5 Meter SLR (m)')
ax[1].set_xlabel('column #')
ax[1].set_ylabel('row #')
ax[1].title.set_text('1.5 Meter SLR (m)')
ax[2].set_xlabel('column #')
ax[2].set_ylabel('row #')
ax[2].title.set_text('2.5 Meter SLR (m)')
genu.quick_plot(emergent_gw_lowslr, ax=ax[0])
genu.quick_plot(emergent_gw_intslr, ax=ax[1])
genu.quick_plot(emergent_gw_highslr, ax=ax[2])
all_ax = ax[0].figure.get_axes()
# Every other axis is a colorbar, starting with second one
for iax in all_ax[1::2]:
    iax.set_xlabel('Height Above Land Surface (m)') # for horizontal bars, ylabel for vertical
# cbar = plt.colorbar(location='bottom')
# cbar.set_label("$\Delta$WT [m]")

dx,dy,gt = rastu.get_rast_info(demfname_nofill)
projwkt = rastu.load_grid_prj(demfname_nofill)
fname_lowslr = os.path.join(work_dir,'CheqEmergentGW_lowslr.tif')
rastu.write_gdaltif(fname_lowslr,cc_proj[0],cc_proj[1],emergent_gw_lowslr,proj_wkt=projwkt)
fname_intslr = os.path.join(work_dir,'CheqEmergentGW_intslr.tif')
rastu.write_gdaltif(fname_intslr,cc_proj[0],cc_proj[1],emergent_gw_intslr,proj_wkt=projwkt)
fname_highslr = os.path.join(work_dir,'CheqEmergentGW_highslr.tif')
rastu.write_gdaltif(fname_highslr,cc_proj[0],cc_proj[1],emergent_gw_highslr,proj_wkt=projwkt)

#%% Non-negligible effects of SLR

head_all = hds.get_alldata().squeeze()
head_all[head_all<-100] = np.nan

# where water rises above hydrofill
depth_to_wt_present_hydrofill = dem_model-head_ss[0]
depth_to_wt_present_hydrofill[depth_to_wt_present_hydrofill<=0.5] = np.nan

head_range_lowSLR = head_lowslr[0] - head_ss[0]
head_range_intSLR = head_intslr[0] - head_ss[0]
head_range_highSLR = head_highslr[0] - head_ss[0]
# head_range[stage_cells] = sl_max-sl_min
head_range_lowSLR_stagemasked = head_range_lowSLR.copy()
head_range_intSLR_stagemasked = head_range_intSLR.copy()
head_range_highSLR_stagemasked = head_range_highSLR.copy()
head_range_lowSLR_stagemasked[stage_cells_lowslr]=np.nan
head_range_intSLR_stagemasked[stage_cells_intslr]=np.nan
head_range_highSLR_stagemasked[stage_cells_highslr]=np.nan

# Plots of head ranges, vmax at respective SLRs
fig,ax = plt.subplots(1,3, sharey=True, constrained_layout=True)
ax[0].set_xlabel('Column', fontsize=26)
ax[0].set_ylabel('Row', fontsize=26)
# ax[0].title.set_text('0.5 Meter SLR (m)')
# ax[0].title.set_fontsize(26)
ax[1].set_xlabel('Column', fontsize=26)
# ax[1].set_ylabel('row #')
# ax[1].title.set_text('1.5 Meter SLR (m)')
# ax[1].title.set_fontsize(26)
ax[2].set_xlabel('Column', fontsize=26)
# ax[2].set_ylabel('row #')
# ax[2].title.set_text('2.5 Meter SLR (m)')
# ax[2].title.set_fontsize(26)
ax1 = ax[0].matshow(head_range_lowSLR_stagemasked, vmin=0, vmax=0.5)
ax2 = ax[1].matshow(head_range_intSLR_stagemasked, vmin=0, vmax=1.5)
ax3 = ax[2].matshow(head_range_highSLR_stagemasked, vmin=0, vmax=2.5)
cbar = fig.colorbar(ax1, ax=ax[0], location='bottom')
# cbar.set_label("$\Delta$WT [m]", fontsize=26)
cbar = fig.colorbar(ax2, ax=ax[1], location='bottom')
cbar.set_label("$\Delta$WT [m]", fontsize=26)
cbar = fig.colorbar(ax3, ax=ax[2], location='bottom')
# cbar.set_label("$\Delta$WT [m]", fontsize=26)
# fig.suptitle('Diked Conditions')

# mask where heads are above dem_model, using emergent (slr-induced sw) gw as surface water
head_range_lowSLR_SWmasked = head_range_lowSLR.copy()
head_range_intSLR_SWmasked = head_range_intSLR.copy()
head_range_highSLR_SWmasked = head_range_highSLR.copy()
head_range_lowSLR_SWmasked[(head_lowslr[0]-dem_model_nofill)>0] = np.nan
head_range_intSLR_SWmasked[(head_intslr[0]-dem_model_nofill)>0] = np.nan
head_range_highSLR_SWmasked[(head_highslr[0]-dem_model_nofill)>0] = np.nan

genu.quick_plot(head_range_lowSLR_SWmasked, vmin=0, vmax=0.5)
genu.quick_plot(head_range_intSLR_SWmasked, vmin=0, vmax=1.5)
genu.quick_plot(head_range_highSLR_SWmasked, vmin=0, vmax=2.5)

SLRrange_low = head_range_lowSLR.copy()
SLRrange_int = head_range_intSLR.copy()
SLRrange_high = head_range_highSLR.copy()
# SLRrange_low[SLRrange_low<0.05] = np.nan
# SLRrange_int[SLRrange_int<0.05] = np.nan
# SLRrange_high[SLRrange_high<0.05] = np.nan

# Remove Stage Cells
SLRrange_low[stage_cells_lowslr] = np.nan
SLRrange_int[stage_cells_intslr] = np.nan
SLRrange_high[stage_cells_highslr] = np.nan

# Plots of head ranges, vmax at SLR
fig,ax = genu.plt.subplots(1,3)
ax[0].set_xlabel('column #')
ax[0].set_ylabel('row #')
ax[0].title.set_text('0.5 Meter SLR (m)')
ax[1].set_xlabel('column #')
ax[1].set_ylabel('row #')
ax[1].title.set_text('1.5 Meter SLR (m)')
ax[2].set_xlabel('column #')
ax[2].set_ylabel('row #')
ax[2].title.set_text('2.5 Meter SLR (m)')
genu.quick_plot(SLRrange_low, vmin=0, vmax=0.5, ax=ax[0])
genu.quick_plot(SLRrange_int, vmin=0, vmax=1.5, ax=ax[1])
genu.quick_plot(SLRrange_high, vmin=0, vmax=2.5, ax=ax[2])

# Plots of head ranges, vmax at max SLR
fig,ax = genu.plt.subplots(1,3)
ax[0].set_xlabel('column #')
ax[0].set_ylabel('row #')
ax[0].title.set_text('0.5 Meter SLR (m)')
ax[1].set_xlabel('column #')
ax[1].set_ylabel('row #')
ax[1].title.set_text('1.5 Meter SLR (m)')
ax[2].set_xlabel('column #')
ax[2].set_ylabel('row #')
ax[2].title.set_text('2.5 Meter SLR (m)')
genu.quick_plot(SLRrange_low, vmin=0, vmax=2.5, ax=ax[0])
genu.quick_plot(SLRrange_int, vmin=0, vmax=2.5, ax=ax[1])
genu.quick_plot(SLRrange_high, vmin=0, vmax=2.5, ax=ax[2])

# Plots of depth to water table with slr next to present heads 
genu.quick_plot(dem_model_nofill[None,:]-np.vstack([head_lowslr[0][None,:],head_ss[0][None,:]]),vmin=0,vmax=1)
genu.quick_plot(dem_model_nofill[None,:]-np.vstack([head_intslr[0][None,:],head_ss[0][None,:]]),vmin=0,vmax=1)
genu.quick_plot(dem_model_nofill[None,:]-np.vstack([head_highslr[0][None,:],head_ss[0][None,:]]),vmin=0,vmax=1)
fig,ax = genu.plt.subplots(1,3)
ax[0].set_xlabel('column #')
ax[0].set_ylabel('row #')
ax[0].title.set_text('0.5 Meter SLR (m)')
ax[1].set_xlabel('column #')
ax[1].set_ylabel('row #')
ax[1].title.set_text('1.5 Meter SLR (m)')
ax[2].set_xlabel('column #')
ax[2].set_ylabel('row #')
ax[2].title.set_text('2.5 Meter SLR (m)')
# figure how to make above plots into subplots

dx,dy,gt = rastu.get_rast_info(demfname_nofill)
projwkt = rastu.load_grid_prj(demfname_nofill)
fname_lowslr = os.path.join(work_dir,'CheqSLRInfluence_lowslr.tif')
rastu.write_gdaltif(fname_lowslr,cc_proj[0],cc_proj[1],head_range_lowSLR_stagemasked,proj_wkt=projwkt)
fname_intslr = os.path.join(work_dir,'CheqSLRInfluence_intslr.tif')
rastu.write_gdaltif(fname_intslr,cc_proj[0],cc_proj[1],head_range_intSLR_stagemasked,proj_wkt=projwkt)
fname_highslr = os.path.join(work_dir,'CheqSLRInfluence_highslr.tif')
rastu.write_gdaltif(fname_highslr,cc_proj[0],cc_proj[1],head_range_highSLR_stagemasked,proj_wkt=projwkt)

#%%  3d surface plot with an attached colorbar 

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm,colors
import matplotlib.pyplot as plt
import numpy as np
#%%
# main code after https://stackoverflow.com/a/6543777
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-5, 5, .25)
Y = np.arange(-5, 5, .25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
Gx, Gy = np.gradient(Z) # gradients with respect to x and y
G = (Gx**2+Gy**2)**.5  # gradient magnitude
#N = G/G.max()  # normalize 0..1
cmap = cm.viridis
N = colors.Normalize(vmin=G.min(),vmax=G.max()) # going off https://stackoverflow.com/a/43449310
sm = cm.ScalarMappable(cmap=cmap,norm=N)
sm.set_array([])

surf = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1,
    facecolors=cmap(N(G)),
    linewidth=0, antialiased=False, shade=False)

fig.colorbar(sm,ticks=np.linspace(0,G.max(),5),orientation='vertical',ax=ax,label='Gradient')

plt.show()

#%% Non-negligible effects of tides

# stage_cells = mask_array & ~np.isnan(ztop) & (ztop <= sl_max)

# head_all = hds.get_alldata().squeeze()
# head_all[head_all<-100] = np.nan
# headmax = np.nanmax(head_all_lay1, axis=0) 
# headmin = np.nanmin(head_all_lay1, axis=0)

# head_range = headmax-headmin
# # head_range[stage_cells] = sl_max-sl_min

# genu.quick_plot(headmax, vmin=0, vmax=2.5)
# genu.quick_plot(headmin, vmin=0, vmax=2.5)
# genu.quick_plot(head_range, vmin=0, vmax=0.1)

# tidalheads = head_range.copy()
# tidalheads[tidalheads<0.05] = np.nan
# tidalheads[stage_cells] = np.nan
# genu.quick_plot(tidalheads, vmin=0, vmax=2)

# dx,dy,gt = rastu.get_rast_info(demfname)
# projwkt = rastu.load_grid_prj(demfname)
# fname = work_dir + 'CheqTidalInfluence_48hr.tif'
# rastu.write_gdaltif(fname,cc_proj[0],cc_proj[1],tidalheads,proj_wkt=projwkt)

# loop later when files get too large

# head = hds.get_data(totim=hds.get_times()[0]) # steady-state head (0th time step)
# head[head<-100] = np.nan
# headmax = head
# headmin = head

# mytimes = np.arange(0, 24, 1) # for plotting 23 graphs
# for time in mytimes: 
#     head = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
#     for i in range(nrow):
#       for j in range(ncol):
#         if head(0,i,j)>headmax: headmax(0,i,j)=head(0,i,j)
#         if head(0,i,j)<headmin: headmin(0,i,j)=head(0,i,j)
    
#     Cheqheadplot_cons = genu.quick_plot(head, vmin=0.0, vmax = 2.5) # plots heads under consideration with color gradient
#     Cheqheads_cons = Cheqheadplot_cons.get_figure()
#     Cheqheads_cons.suptitle(str(time)+" hrs") # how does one label the figure?
#     Cheqheads_cons.savefig(work_dir+"CheqHeadsCheqHeads0"+str(f"{time:02d}")+".png")

#%% Herring River Transects

dike_transect_y = np.arange(690,730,4)
dike_transect_x = np.arange(170,180,1)
dike_transect_cells = np.vstack((dike_transect_x, dike_transect_y)).T

# upstrm_transect_y = np.arange(300,223,-7)
# upstrm_transect_x = np.arange(540,551,1)
# upstrm_transect_cells = np.vstack((upstrm_transect_x, upstrm_transect_y)).T

dike_transect_heads_alltimes = []
upstrm_transect_heads_alltimes = []
for time in mytimes:
    head = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
    head[head<-100] = np.nan
    dike_transect_heads = []
    dike_transect_elev = []
    upstrm_transect_heads = []
    upstrm_transect_elev = []
    for irow, icol in dike_transect_cells:
        temp_head = head[:,irow,icol]
        dike_transect_heads.append(temp_head[0])
        temp_elev = np.array([ztop[irow,icol]])
        dike_transect_elev.append(temp_elev[0])
        plt.plot(dike_transect_heads) # smoothing is unrealistic compared to the binning that the model does
        plt.plot(dike_transect_elev) # smoothing is unrealistic compared to the binning that the model does
        plt.ylabel('elevation (m)')
        plt.xlabel('transect length (m)')
    dike_transect_heads_alltimes.append(dike_transect_heads)
    # for irow, icol in upstrm_transect_cells:
    #     temp_head = head[:,irow,icol]
    #     upstrm_transect_heads.append(temp_head[0])
    #     temp_elev = np.array([ztop[irow,icol]])
    #     upstrm_transect_elev.append(temp_elev[0])
    #     plt.plot(upstrm_transect_heads) # smoothing is unrealistic compared to the binning that the model does
    #     plt.plot(upstrm_transect_elev) # smoothing is unrealistic compared to the binning that the model does
    #     plt.ylabel('elevation (m)')
    #     plt.xlabel('transect length (m)')
    # upstrm_transect_heads_alltimes.append(upstrm_transect_heads)

# switched x and y to account for grid rotation
dike_transect_pt1_xcoord = cc_proj[0][210,410] # for x-coordinate in UTM
dike_transect_pt1_ycoord = cc_proj[1][210,410] # for y-coordinate in UTM

dike_transect_pt2_xcoord = cc_proj[0][190,460] # for x-coordinate in UTM
dike_transect_pt2_ycoord = cc_proj[1][190,460] # for y-coordinate in UTM

dike_transect_length = np.sqrt((dike_transect_pt2_xcoord-dike_transect_pt1_xcoord)**2+(dike_transect_pt2_ycoord-dike_transect_pt1_ycoord)**2)

upstrm_transect_pt1_xcoord = cc_proj[0][300,540] # for x-coordinate in UTM
upstrm_transect_pt1_ycoord = cc_proj[1][300,540] # for y-coordinate in UTM

upstrm_transect_pt2_xcoord = cc_proj[0][230,550] # for x-coordinate in UTM
upstrm_transect_pt2_ycoord = cc_proj[1][230,550] # for y-coordinate in UTM

upstrm_transect_length = np.sqrt((upstrm_transect_pt2_xcoord-upstrm_transect_pt1_xcoord)**2+(upstrm_transect_pt2_ycoord-upstrm_transect_pt1_ycoord)**2)

dike_transect_midpts = []
for i in range(len(dike_transect_elev)):
    dike_transect_midpts.append(i*dike_transect_length/len(dike_transect_elev))
 
for i in range(len(mytimes)):
    plt.figure()
    plt.step(dike_transect_midpts,dike_transect_elev,where='mid',color='saddlebrown')
    plt.step(dike_transect_midpts,dike_transect_heads_alltimes[i],where='mid',color='aquamarine')    
    plt.ylabel('elevation (m)')
    plt.xlabel('transect length (m)')
    # plt.axis([24, nper, -0.5, 2.0])
    plt.grid(True)
    plt.title("Near Dike Water Level at "+str(mytimes[i])+" hrs") # how does one label the figure? not using title for these since the stepping is on the x-axis.
    plt.savefig(work_dir+"DikeSeaLevelDikeSeaLevel0"+str(f"{i:02d}")+".png")
    plt.show()    

upstrm_transect_midpts = []
for i in range(len(dike_transect_elev)):
    upstrm_transect_midpts.append(i*upstrm_transect_length/len(upstrm_transect_elev))
 
for i in range(len(mytimes)):
    plt.figure()
    plt.step(upstrm_transect_midpts,upstrm_transect_elev,where='mid',color='saddlebrown')
    plt.step(upstrm_transect_midpts,upstrm_transect_heads_alltimes[i],where='mid',color='aquamarine')    
    plt.ylabel('elevation (m)')
    plt.xlabel('transect length (m)')
    # plt.axis([24, nper, -0.5, 2.0])
    plt.grid(True)
    plt.title("Upstream of Dike Water Level at "+str(mytimes[i])+" hrs") # how does one label the figure? not using title for these since the stepping is on the x-axis.
    plt.savefig(work_dir+"DikeUpstrmSeaLevelDikeUpstrmSeaLevel0"+str(f"{i:02d}")+".png")
    plt.show()
    
"""
Animation
"""
# Make a movie of near dike transect (.png files must be saved with Tools>Preferences>IPython Console>Graphics>Backend set to Automatic),
    # otherwise the files will not be the same size
os.chdir(r'E:HerringDikeSeaLevel')   
img = os.listdir(r'E:HerringDikeSeaLevel')

clip = mpy.ImageSequenceClip(img, fps=2)
os.chdir(r'E:HerringDikeSeaLevel') 
clip.write_gif('DikeTransect_24-47hr_unconf.gif', fps=12)

# Make a movie of upstream of dike transect
os.chdir(r'E:HerringDikeUpstrmSeaLevel')   
img = os.listdir(r'E:HerringDikeUpstrmSeaLevel')

clip = mpy.ImageSequenceClip(img, fps=2)
os.chdir(r'E:HerringDikeUpstrmSeaLevel') 
clip.write_gif('UpstrmTransect_24-47hr_unconf.gif', fps=12)

#%% Time-series on x-axis w/in transects

plt.figure()
plt.plot(mytimes, dike_transect_heads_alltimes)
plt.ylabel('head (m)')
plt.xlabel('time (hrs)')
plt.show()

#%%
"""
Flopy also has some pre-canned plotting capabilities can can be accessed using the ModelMap class. 
The following code shows how to use the modelmap class to plot boundary conditions (IBOUND), 
plot the grid, plot head contours, and plot vectors:
"""

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')

# hds = bf.HeadFile(modelname+'.hds')
# times = hds.get_times()
# head = hds.get_data(totim=times[-1])
# levels = np.linspace(0, 10, 11)

# cbb = bf.CellBudgetFile(modelname+'.cbc')
# kstpkper_list = cbb.get_kstpkper()
# frf = cbb.get_data(text='FLOW RIGHT FACE', totim=times[-1])[0]
# fff = cbb.get_data(text='FLOW FRONT FACE', totim=times[-1])[0]
#%%
"""
The pre-canned plotting doesn't seem to be able to allow averaging to reduce nrow and ncol
on the plot, making it difficult to plot a large grid. The commented section below uses the
modelmap class from Tutorial 1, followed by use of the plotting from the Henry Problem.
"""

#modelmap = flopy.plot.ModelMap(model=mf, layer=0)
#qm = modelmap.plot_ibound()
#lc = modelmap.plot_grid() # Need to fix grid to have fewer rows and columns
#cs = modelmap.contour_array(head, levels=levels)
#quiver = modelmap.plot_discharge(frf, fff, head=head)
#plt.savefig('CheqModel2b.png')

"""
# Load data (when implementing SEAWAT)
ucnobj = bf.UcnFile('MT3D001.UCN', model=swt)
times = ucnobj.get_times()
concentration = ucnobj.get_data(totim=times[-1])
"""

# Average flows to cell centers
qx_avg = np.empty(frf.shape, dtype=frf.dtype)
qx_avg[:, :, 1:] = 0.5 * (frf[:, :, 0:ncol-1] + frf[:, :, 1:ncol])
qx_avg[:, :, 0] = 0.5 * frf[:, :, 0]
qy_avg = np.empty(fff.shape, dtype=fff.dtype)
qy_avg[1:, :, :] = 0.5 * (fff[0:nlay-1, :, :] + fff[1:nlay, :, :])
qy_avg[0, :, :] = 0.5 * fff[0, :, :]

# Make the plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
#ax.imshow(concentration[:, 0, :], interpolation='nearest',
#           extent=(0, Lx, 0, Ly))

y, x, z = dis.get_node_coordinates()
X, Y = np.meshgrid(x, y)
iskip = 3
ax.quiver(X[::iskip, ::iskip], Y[::iskip, ::iskip],
           qx_avg[::iskip, 0, ::iskip], -qy_avg[::iskip, 0, ::iskip],
           color='k', scale=5, headwidth=3, headlength=2,
           headaxislength=2, width=0.0025)
plt.savefig('CheqModel2b.png')
plt.show()
#%%
"""
Post-Processing the Results
Once again, we can read heads from the MODFLOW binary output file, using the flopy.utils.binaryfile module. 
Included with the HeadFile object are several methods that we will use here: * get_times() will return a 
list of times contained in the binary head file * get_data() will return a three-dimensional head array for 
the specified time * get_ts() will return a time series array [ntimes, headval] for the specified cell

Using these methods, we can create head plots and hydrographs from the model results.:
"""

# Create the headfile and budget file objects
times = hds.get_times()
cbb = bf.CellBudgetFile(modelname+'.cbc')

# Setup contour parameters 
levels = np.linspace(0, 10, 11)
extent = (delr/2., Lx - delr/2., delc/2., Ly - delc/2.)
print('Levels: ', levels)
print('Extent: ', extent)

# Make the plots

#Print statistics
print('Head statistics')
print('  min: ', head.min())
print('  max: ', head.max())
print('  std: ', head.std())

"""
Again, commented out section using modelmap
"""

## Flow right face and flow front face already extracted
##%%
##Create the plot
#f = plt.figure()
#plt.subplot(1, 1, 1, aspect='equal')
#
#
#modelmap = flopy.plot.ModelMap(model=mf, layer=0)
#qm = modelmap.plot_ibound()
## 
## lc = modelmap.plot_grid()
#qm = modelmap.plot_bc('GHB', alpha=0.5)
#cs = modelmap.contour_array(head, levels=levels)
#plt.clabel(cs, inline=1, fontsize=10, fmt='%1.1f', zorder=11)
#quiver = modelmap.plot_discharge(frf, fff, head=head)
#
#mfc='black'
#plt.plot(lw=0, marker='o', markersize=8,
#         markeredgewidth=0.5,
#         markeredgecolor='black', markerfacecolor=mfc, zorder=9)
#plt.savefig('CheqModel2-{}.png')
    
"""
From Henry Problem
"""

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
im = ax.imshow(head[:, 500, 500:600], interpolation='nearest',
               extent=(0, Lx, 0, Ly))
ax.set_title('Simulated Heads')    
    
#%% Heads from Masterson's Model

plt.subplot(1,1,1,aspect='equal')
hds_Mast04 = bf.HeadFile(os.path.join(work_dir,'Lcape_Masterson.hds'))
head_Mast04 = hds_Mast04.get_data(totim=hds_Mast04.get_times()[0])
head_Mast04[head_Mast04<-100] = np.nan
head_Mast04[head_Mast04>200] = np.nan
extent = (delr/2., Lx - delr/2., Ly - delc/2., delc/2.)
plt.xlabel('Lx')
plt.ylabel('Ly')
headplot_Mast04 = plt.contour(head_Mast04[0, :, :], extent=extent)
plt.colorbar(headplot_Mast04)

# First Layer of heads from Masterson_2004 Model
dx,dy,gt = rastu.get_rast_info(demfname)
projwkt = rastu.load_grid_prj(demfname)
fname = work_dir + 'MastersonFirstLayerHeads.tif'
rastu.write_gdaltif(fname,cc_proj[0],cc_proj[1],head_Mast04[0],proj_wkt=projwkt)

"""
For writing Masterson head data to shapefile
"""
# ncol = 110
# nrow = 320
# delc = 400
# delr = 400
# rot_xy = np.deg2rad(33) # rotation angle in degrees to radians
# lowleft_X = 1006459.2 # NAD27, feet
# lowleft_Y = 285071.4 # NAD27, feet
# dem_X_Masterson = np.array([np.arange(lowleft_X, lowleft_X+400*(ncol+1), delc),]*321)
# dem_Y_Masterson = np.array([np.arange(lowleft_Y+400*(nrow), lowleft_Y-400, -delr),]*111).transpose()

# dem_X_Mast_zeroed = dem_X_Masterson - lowleft_X
# dem_Y_Mast_zeroed = dem_Y_Masterson - lowleft_Y

# # rotation based on https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
# dem_X_Mast_rot = dem_X_Mast_zeroed*np.cos(rot_xy) - dem_Y_Mast_zeroed*np.sin(rot_xy) + lowleft_X
# dem_Y_Mast_rot = dem_X_Mast_zeroed*np.sin(rot_xy) + dem_Y_Mast_zeroed*np.cos(rot_xy) + lowleft_Y

# height_orig = nrow*delr
# width_orig = ncol*delc

# proj_wkt = r'PROJCS["NAD27 / Massachusetts Mainland",GEOGCS["GCS_North_American_1927",DATUM["D_North_American_1927",SPHEROID["Clarke_1866",6378206.4,294.9786982138982]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Conformal_Conic"],PARAMETER["standard_parallel_1",41.71666666666667],PARAMETER["standard_parallel_2",42.68333333333333],PARAMETER["latitude_of_origin",41],PARAMETER["central_meridian",-71.5],PARAMETER["false_easting",600000],PARAMETER["false_northing",0],UNIT["Foot_US",0.30480060960121924]]'

# # from raster_utils

# dx = delc
# dy = delr

# geodata = [dem_X_Mast_rot[0,0],
#            np.cos(rot_xy)*dx,
#            np.sin(rot_xy)*dx,
#            dem_Y_Mast_rot[0,0],
#            np.sin(rot_xy)*dy,
#            -np.cos(rot_xy)*dy]
# rastu.write_gdaltif(os.path.join(work_dir,"MastersonFirstLayerHeads.tif"),None,None,head_Mast04[0],rot_xy=None,proj_wkt=proj_wkt,set_proj=True,nan_val = -9999.,dxdy=[None,None],geodata=geodata)
    
#%% Drains

ss_drains = 1*(abs(head_ss[0]-dem_model_nofill)<0.01)
lowslr_drains = 1*(abs(head_lowslr[0]-dem_model_nofill)<0.01)
intslr_drains = 1*(abs(head_intslr[0]-dem_model_nofill)<0.01)
highslr_drains = 1*(abs(head_highslr[0]-dem_model_nofill)<0.01)
    
fname_ss_drains = os.path.join(work_dir, 'Cheq_Drain_SS.tif')
fname_lowslr_drains = os.path.join(work_dir, 'Cheq_Drain_LowSLR.tif')
fname_intslr_drains = os.path.join(work_dir, 'Cheq_Drain_IntSLR.tif')
fname_highslr_drains = os.path.join(work_dir, 'Cheq_Drain_HighSLR.tif')

rastu.write_gdaltif(fname_ss_drains,cc_proj[0],cc_proj[1],ss_drains,proj_wkt=projwkt)
rastu.write_gdaltif(fname_lowslr_drains,cc_proj[0],cc_proj[1],lowslr_drains,proj_wkt=projwkt)
rastu.write_gdaltif(fname_intslr_drains,cc_proj[0],cc_proj[1],intslr_drains,proj_wkt=projwkt)
rastu.write_gdaltif(fname_highslr_drains,cc_proj[0],cc_proj[1],highslr_drains,proj_wkt=projwkt)




