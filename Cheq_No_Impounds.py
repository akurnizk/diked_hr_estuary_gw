# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:25:03 2019
​
@author: kbefus & akurnizk
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio import mask
from rasterio.io import MemoryFile
from scipy.interpolate import griddata

#%%
def xy_from_affine(tform=None,nx=None,ny=None):
    X,Y = np.meshgrid(np.arange(nx)+0.5,np.arange(ny)+0.5)*tform
    return X,Y

def get_meta(in_fname=None):
    with rasterio.open(in_fname) as src:
        meta=src.profile
    return meta

def read_geotiff(in_fname,band=0):
    with rasterio.open(in_fname) as src:
        data = src.read()[band]
        data[data==src.nodata]=np.nan
        ny,nx = data.shape
        X,Y = xy_from_affine(src.transform,nx,ny)
    return X,Y,data
#%%
work_dir = r'E:\Maps'
dem_fname = os.path.join(work_dir,'USGS_NED_Chequesset_one_meter_Combined.tif') # dem to edit
shp_fname = os.path.join(work_dir,'Chequesset_Impoundments_NoDike.shp') # shapefile with features outlining areas to remove & interpolate over

shp_df = gpd.read_file(shp_fname)

# Options
nodata = -9999
npt_domainbuffer = 10

# if shapefile and dem are in the same coordinate system
X,Y,dem_data = read_geotiff(dem_fname)
dem_profile = get_meta(dem_fname)
temp_profile = dem_profile.copy()
temp_profile['driver'] = 'GTiff'
rmv_list = ['tiled']
for rmv in rmv_list:
    adum=temp_profile.pop(rmv)
#%%
rast_filt = None
# Loop through features in shapefile and interpolate over the gaps
for igeom,geom_temp in enumerate(shp_df.geometry.values):
    print('===== Geometry {}/{} ======='.format(igeom+1,shp_df.shape[0]))
    if rast_filt is None:
        print('Loading original DEM')
        with rasterio.open(dem_fname) as src:
            d1 = src.read()[0]
            ocean_mask = d1<=nodata
            cut_array,vtransform = mask.mask(src,[geom_temp],all_touched=True,invert=True,nodata=nodata) # masks inside the geom feature
            cut_array[0,ocean_mask] = np.nan
    else:
        # Use updated dem from previous loop
        with MemoryFile() as memfile:
            with memfile.open(**temp_profile) as dataset:
                dataset.write(rast_filt[None,:,:]) # need to make 3d array to write
                cut_array,vtransform = mask.mask(dataset,[geom_temp],all_touched=True,invert=True,nodata=nodata) # masks inside the geom feature
                
    cut_array2 = cut_array.squeeze()
    rast_nans = cut_array2==nodata
    nanX,nanY = X[rast_nans],Y[rast_nans]
    
    # Use only area around active feature
    rowinds,colinds = rast_nans.nonzero()
    minrow = np.min(rowinds)-npt_domainbuffer
    maxrow = np.max(rowinds)+npt_domainbuffer
    mincol = np.min(colinds)-npt_domainbuffer
    maxcol = np.max(colinds)+npt_domainbuffer
    z1 = cut_array2[minrow:maxrow+1,mincol:maxcol+1]
    znans = z1==nodata
    x1 = X[minrow:maxrow+1,mincol:maxcol+1]
    y1 = Y[minrow:maxrow+1,mincol:maxcol+1]

    # Linearly interpolate over masked area
    new_vals = griddata(np.c_[x1[~znans],y1[~znans]],z1[~znans],(nanX,nanY),method='linear')
    cut_array2[rast_nans] = new_vals # insert original values where possible
    rast_filt = cut_array2.copy()

#%%

dem_data[dem_data<=nodata] = np.nan
rast_filt[rast_filt==nodata] = np.nan
plt.matshow(rast_filt-dem_data, vmin=0, vmax=1)
plt.matshow(dem_data)
plt.matshow(rast_filt)

fname = os.path.join(work_dir, 'USGS_NED_Chequesset_one_meter_Combined_Dike_NoImpounds.tif')
with rasterio.open(fname,'w',**temp_profile) as src:
    src.write(rast_filt[None,:,:])