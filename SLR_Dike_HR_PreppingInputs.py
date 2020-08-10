# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 08:55:10 2020

@author: akurnizk
"""

import utm
import csv
import math
import flopy
import scipy
import fiona
import pyproj
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
from functools import partial
from scipy.optimize import fsolve
from shapely.ops import transform
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
from matplotlib.colors import ListedColormap
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
mattblspc = loadmat(os.path.join(map_dir, 'GUI_version2','tblspc.mat'))
xl,yl,belv = matcont_hr['xl'], matcont_hr['yl'], matcont_hr['belv']
whg_points = np.concatenate((xl,yl),axis=1)
geom_whg = [Point(xy) for xy in zip(whg_points[:,0],whg_points[:,1])]
crs_ma_mainland = {'init': 'EPSG:26986'}
whg_df = gpd.GeoDataFrame(crs=crs_ma_mainland,geometry=geom_whg)
# whg_df_utm = whg_df.to_crs({'init':'epsg:26919'})
# project = partial(pyproj.transform,pyproj.Proj(whg_df.crs),pyproj.Proj(init='epsg:26919'))
whg_df_proj = whg_df.copy()
whg_df_proj = whg_df_proj.to_crs({'init': 'EPSG:26919'})

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

#%% Elevations at intersections and at 1m increments along cross-section line

"""
Load and sort Channel Geometry
"""
centerline_df_HR = gpd.read_file(os.path.join(map_dir, 'HR_XsecLines\CenterLine_HR.shp'))
xsecs_df_HR = gpd.read_file(os.path.join(map_dir, 'HR_XsecLines\ClineXsecs_HR_Sort.shp'))
c_line_xsecs_intersect_unsorted = gpd.GeoDataFrame(geometry=list(centerline_df_HR.unary_union.intersection(xsecs_df_HR.unary_union)))
c_line_length = centerline_df_HR['geometry'].length # only the linear distance from start to end?

list_arrays = [np.array((c_line_xsecs_intersect_unsorted.geometry[geom].xy[0][0], c_line_xsecs_intersect_unsorted.geometry[geom].xy[1][0])) for geom in range(c_line_xsecs_intersect_unsorted.shape[0])]
c_line_xsecs_intersect_xy_unsorted = np.array(list_arrays)
intersects_x = c_line_xsecs_intersect_xy_unsorted[:,0]
intersects_y = c_line_xsecs_intersect_xy_unsorted[:,1]

points = c_line_xsecs_intersect_xy_unsorted.copy()
x = intersects_x.copy()
y = intersects_y.copy()
from sklearn.neighbors import NearestNeighbors

clf = NearestNeighbors(2).fit(points)
G = clf.kneighbors_graph()

import networkx as nx

T = nx.from_scipy_sparse_matrix(G)

order = list(nx.dfs_preorder_nodes(T, 0))

xx = x[order]
yy = y[order]

plt.plot(xx, yy)
plt.show()

paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]

mindist = np.inf
minidx = 0

for i in range(len(points)):
    p = paths[i]           # order of nodes
    ordered = points[p]    # ordered nodes
    # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
    cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
    if cost < mindist:
        mindist = cost
        minidx = i
        
opt_order = paths[minidx]

xx = x[opt_order]
yy = y[opt_order]

plt.plot(xx, yy)
plt.show()

c_line_xsecs_intersect_xy_sorted = np.vstack((xx,yy)).T
geom_cline_xsecs = [Point(xy) for xy in zip(c_line_xsecs_intersect_xy_sorted[:,0],c_line_xsecs_intersect_xy_sorted[:,1])]
c_line_xsecs_intersect = gpd.GeoDataFrame(geometry=geom_cline_xsecs)

#%% Turn centerline into GeoSeries of Points from Linestring
centerline_xy_HR = centerline_df_HR.geometry[0].xy

cline_x = centerline_xy_HR[0]
cline_y = centerline_xy_HR[1]

cline_xy_arr = np.vstack((cline_x,cline_y)).T
cline_xy_df = pd.DataFrame(cline_xy_arr)
geom_cline = [Point(xy) for xy in zip(cline_xy_arr[:,0],cline_xy_arr[:,1])]
cline_xy_gdf = gpd.GeoDataFrame(geometry=geom_cline)

#%% Add start and end points to intersection point gdf

cline_start = cline_xy_arr[0]
geom_cline_start = [Point(cline_start[0],cline_start[1])]
cline_start_gdf = gpd.GeoDataFrame(geometry=geom_cline_start)
cline_end = cline_xy_arr[-1]
geom_cline_end = [Point(cline_end[0],cline_end[1])]
cline_end_gdf = gpd.GeoDataFrame(geometry=geom_cline_end)

c_line_xsecs_intersect_ends = pd.concat((cline_start_gdf,c_line_xsecs_intersect,cline_end_gdf),ignore_index=True)
c_line_xsecs_intersect_ends.plot(cmap=ListedColormap(['r','g','b']))

def ckdnearest(gdA, gdB): # WHYYYYYYYYY
    nA = np.array(list(zip(gdA.geometry.x, gdA.geometry.y)) )
    nB = np.array(list(zip(gdB.geometry.x, gdB.geometry.y)) )
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdf = pd.concat(
        [gdA.reset_index(drop=True), gdB.loc[idx, gdB.columns != 'geometry'].reset_index(drop=True),
          pd.Series(dist, name='dist')], axis=1)
    return gdf

dist_strt_to_xsecint = ckdnearest(c_line_xsecs_intersect_ends, cline_start_gdf) # not every sequential point is farther from the dike

# Getting Grid X & Y for xsecs
# xsecs_temp_crs = xsecs_df_HR.crs
# xsecs_minx,xsecs_miny,xsecs_maxx,xsecs_maxy = xsecs_df_HR.bounds.values.T
# leftbound,topbound = minx.min(),maxy.max()

#%% Discretize cross-sections and interpolate elevations
xsecs_df_HR.plot(cmap=ListedColormap(['r','g','b'])) # to make sure ordering is correct

out_x = []
out_y = []
n1=100 # number of subdivisions of cross section
for j in np.arange(xsecs_df_HR.shape[0]):
    temp_xsec = xsecs_df_HR.iloc[j] # first cross section
    xsec_geom = temp_xsec.geometry
    out_xy = []
    for i in np.arange(n1):
        out_xy.append(xsec_geom.interpolate(i/n1,normalized=True).xy)
    out_xy = np.array(out_xy).squeeze()
    out_x.append(out_xy[:,0])
    out_y.append(out_xy[:,1])

out_x_stacked = np.stack(out_x) # write out
out_y_stacked = np.stack(out_y) # write out

np.savetxt(os.path.join(map_dir, 'HR_XsecLines\HR_xsec_all_xcoords.csv'), out_x_stacked, delimiter=',')
np.savetxt(os.path.join(map_dir, 'HR_XsecLines\HR_xsec_all_ycoords.csv'), out_y_stacked, delimiter=',')

newxy = np.array([igeom.xy for igeom in whg_df_proj.geometry.values])
newxy = newxy.squeeze()
elevs_interp = scipy.interpolate.griddata(newxy,belv,(out_x_stacked,out_y_stacked),method='linear')
elevs_interp = elevs_interp.squeeze() # write out

np.savetxt(os.path.join(map_dir, 'HR_XsecLines\HR_xsec_all_elevs.csv'), elevs_interp, delimiter=',')

plt.figure()
plt.plot(elevs_interp[0,:], label = 'Dike')
plt.plot(elevs_interp[18,:], label = 'Halfway up')
plt.plot(elevs_interp[36,:], label = 'High Toss')
plt.legend()

intersect_newxy = np.array([igeom.xy for igeom in c_line_xsecs_intersect_ends.geometry.values])
intersect_newxy = intersect_newxy.squeeze() # write out

np.savetxt(os.path.join(map_dir, 'HR_XsecLines\HR_xsec_all_inscts.csv'), intersect_newxy, delimiter=',')

"""
Finding dx values between transects
"""

min_dist = np.empty(c_line_xsecs_intersect.shape[0]-1) # write out
for i in range(c_line_xsecs_intersect.shape[0]-1):
    min_dist[i] = np.sqrt((c_line_xsecs_intersect.geometry[i].xy[0][0]-c_line_xsecs_intersect.geometry[i+1].xy[0][0])**2 + 
            (c_line_xsecs_intersect.geometry[i].xy[1][0]-c_line_xsecs_intersect.geometry[i+1].xy[1][0])**2)

np.savetxt(os.path.join(map_dir, 'HR_XsecLines\HR_xsec_all_dx.csv'), min_dist, delimiter=',')

