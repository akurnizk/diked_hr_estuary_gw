# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:46:20 2018

@author: kbefus
"""

import sys,os

# Location of BitBucket folder containing cgw folder
cgw_code_dir = 'E:\python'

sys.path.insert(0,cgw_code_dir)

from cgw.utils import general_utils as genu
from cgw.utils import feature_utils as shpu
from cgw.utils import raster_utils as rastu
#%%

# Example of making a MODFLOW-like grid from a shapefile

data_dir = r'E:\ArcGIS'
shp_fname = os.path.join(data_dir,'Chequesset_Model_Area_UTM.shp')

cell_spacing = 10. # model grid cell spacing in meters
# Define inputs for shp_to_grid function
shp_to_grid_dict = {'shp':shp_fname,'cell_spacing':cell_spacing}
grid_outputs = shpu.shp_to_grid(**shp_to_grid_dict)
# Pop out all of the outputs into individual variables
[X_nodes,Y_nodes],model_polygon,[out_proj,[xshift,yshift],min_angle] = grid_outputs

grid_transform = [out_proj,[xshift,yshift],min_angle] # make transform list

# Can calculate cell centers (where heads are calculated), in different coordinates
cc,cc_proj,cc_ll = shpu.nodes_to_cc([X_nodes,Y_nodes],grid_transform)

# Use model_polygon to define active cells in the model
ir,ic,_ = shpu.gridpts_in_shp(model_polygon,cc)
active_cells = genu.define_mask(cc,[ir,ic])

# Plot active cells
fig,ax = genu.plt.subplots(1,2)
genu.quick_plot(active_cells.astype(int),ax=ax[0]) # in row, column space
ax[0].set_xlabel('column #')
ax[0].set_ylabel('row #')
c1=ax[1].pcolormesh(cc[0],cc[1],active_cells.astype(int)) # in model coordinates
genu.plt.colorbar(c1,ax=ax[1],orientation='horizontal')
ax[1].set_xlabel('X [m]')
ax[1].set_ylabel('Y [m]')

#%% Example of loading DEM data for that area
dem_fname = os.path.join(data_dir,'Cheq10mx10m_UTM.tif')
# Experimental part \/
dem_X,dem_Y,dem_da = rastu.load_geotif(dem_fname) # da is an xarray data array
dem_vals = dem_da.values.squeeze()

#dem_X, dem_Y, dem_vals = rastu.read_griddata(dem_fname)

# Know that dem is way higher resolution...can decimate it to save time
decimate_by_ncells = 1 # by every n cells
#dem_X = dem_X[::decimate_by_ncells,::decimate_by_ncells]
#dem_Y = dem_Y[::decimate_by_ncells,::decimate_by_ncells]
#dem_vals = dem_vals[::decimate_by_ncells,::decimate_by_ncells]

# Set no-data value to nan
dem_vals[dem_vals==dem_da.nodatavals[0]] = genu.np.nan

# Transform dem to model coordinates with linear interpolation
trans_dict = {'orig_xy':[dem_X,dem_Y],'orig_val':dem_vals,'active_method':'linear',
              'new_xy':cc_proj} # if dem in same projection as model boundary shp
dem_trans = rastu.subsection_griddata(**trans_dict)
dem_trans[dem_trans<-1000] = genu.np.nan

genu.quick_plot(dem_trans)
