# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:47:36 2018

@author: akurnizk
"""

import flopy
import numpy as np
import sys,os
import matplotlib.pyplot as plt
# Location of BitBucket folder containing cgw folder
cgw_code_dir = 'E:\python'

sys.path.insert(0,cgw_code_dir)

from cgw.utils import general_utils as genu
from cgw.utils import feature_utils as shpu
from cgw.utils import raster_utils as rastu

# Assign name and create modflow model object
modelname = 'CheqModel1'
work_dir = r'E:\Herring'
mf = flopy.modflow.Modflow(modelname, exe_name='mf2005',model_ws=work_dir)
swt = flopy.seawat.Seawat(modelname, exe_name='swtv4')
print(swt.namefile)
mean_sea_level = 0.843 # in meters at closest NOAA station
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

"""
Plot active cells
"""
#fig,ax = genu.plt.subplots(1,2)
#genu.quick_plot(active_cells.astype(int),ax=ax[0]) # in row, column space
#ax[0].set_xlabel('column #')
#ax[0].set_ylabel('row #')
#c1=ax[1].pcolormesh(cc[0],cc[1],active_cells.astype(int)) # in model coordinates
#genu.plt.colorbar(c1,ax=ax[1],orientation='horizontal')
#ax[1].set_xlabel('X [m]')
#ax[1].set_ylabel('Y [m]')

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
#%% DEM model inputs

Lx = np.amax(dem_X)-np.amin(dem_X)
Ly = np.amax(dem_Y)-np.amin(dem_Y)
zbot = -100 # if bottom of model is horizontal, approx. bedrock (check Masterson)
nlay = 1 # 1 layer model
nrow, ncol = cc[0].shape # to use when cheq_griddev is implemented
delr = cell_spacing
delc = cell_spacing
delv = (dem_trans - zbot) / nlay
botm = zbot

# Tutorial 1 model domain and grid definition
#Lx = 1000.
#Ly = 1000.
#ztop = 0.
#zbot = -50.
#nlay = 1
#nrow = 10
#ncol = 10
#delr = Lx/ncol
#delc = Ly/nrow
#delv = (ztop - zbot) / nlay
#botm = np.linspace(ztop, zbot, nlay + 1)

#%%
"""
Time Stepping
"""

# Time step parameters
total_length = 10 # days
dt = 6 # stress period time step, hrs
perlen_days = dt/24. # stress period time step, days

nper = int(total_length/perlen_days) # the number of stress periods in the simulation
nstp_default = dt/0.5 # stress period time step divided by step time length (to better interpolate tidal changes, set to 0.5 hrs)
perlen = [perlen_days]*nper # length of a stress period; each item in the matrix is the amount 
                            # of elapsed time since the previous point (need to change the first)
perlen[0] = 1 # set first step as steady state
steady = [False]*nper
steady[0] = True # first step steady state
nstp = [nstp_default]*nper # number of time steps in a stress period
nstp[0] = 1

#Tutorial 2 default time step parameters
#nper = 3
#perlen = [1, 100, 100]
#nstp = [1, 100, 100]
#steady = [True, False, False]

#%% # Create the discretization (DIS) object
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                               top=dem_trans, botm=botm)
# Tutorial 1 DIS object
#dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                               #top=dem_vals, botm=botm[1:])
                               
# Tutorial 2 DIS object when transient conditions are implemented
# dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
#                               top=ztop, botm=botm[1:],
#                               nper=nper, perlen=perlen, nstp=nstp, steady=steady)
                               

#%% # Variables for the BAS (basic) package
    # Added 5/28/19
"""
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

ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
ibound[:,~active_cells] = 0 # far offshore cells are inactive
ibound[0,dem_trans<mean_sea_level] = -1 # fixed head for everything less than msl
ibound[:,np.isnan(dem_trans)] = 0 # nan cells are inactive

genu.quick_plot(ibound) # plots boundary condition: 1 is above mean sea level (msl), 0 is msl, -1 is under msl.

strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
active_dem_heights = dem_trans[active_cells & ~np.isnan(dem_trans)]
strt[0, active_cells & ~np.isnan(dem_trans)] = active_dem_heights # start with freshwater at surface elevation
strt[0, dem_trans<mean_sea_level] = mean_sea_level # start with water at sea level

genu.quick_plot(strt) # plots starting condition

bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

#%% # added 3/8/19 - creates matrix where hydraulic conductivities (hk = horiz, vk = vert) can be implemented
hk1 = np.ones((nlay,nrow,ncol), np.float)
hk1[:,:,:]=10. # everything set to 10 - use data? calculate?
vka1 = np.ones((nlay,nrow,ncol), np.float)
vka1[:,:,:]=10. # everything set to 10.

# Add LPF package to the MODFLOW model
lpf = flopy.modflow.ModflowLpf(mf, hk=hk1, vka=vka1, ipakcb=53)
#%%
"""
Transient General-Head Boundary Package 
First, we will create the GHB object, which is of the following type: 
    flopy.modflow.ModflowGhb.
The key to creating Flopy transient boundary packages is recognizing that the 
boundary data is stored in a dictionary with key values equal to the 
zero-based stress period number and values equal to the boundary conditions 
for that stress period. For a GHB the values can be a two-dimensional nested 
list of [layer, row, column, stage, conductance]:
    
    Datums for 8447435, Chatham, Lydia Cove MA
https://tidesandcurrents.noaa.gov/datums.html?units=1&epoch=0&id=8447435&name=Chatham%2C+Lydia+Cove&state=MA
"""
# Make list for stress period 1
# Using Mean Sea Level (MSL) in meters at closest NOAA station for stages
#stageleft = mean_sea_level
#stageright = mean_sea_level
#bound_sp1 = []
#for il in range(nlay):
#    # Figure out looping through hk1 array to get hk values at each cell for changing conductance.
#    condleft = hk1[0,0,0] * (stageleft - zbot) * delc
#    condright = hk1[0,0,0] * (stageright - zbot) * delc
#    for ir in range(nrow):
#        bound_sp1.append([il, ir, 0, stageleft, condleft])
#        bound_sp1.append([il, ir, ncol - 1, stageright, condright])
## Only 1 stress period for steady-state model
#print('Adding ', len(bound_sp1), 'GHBs for stress period 1.')
#
#stress_period_data = {0: bound_sp1}
#ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=stress_period_data)

# using single conductance value (see drain for modification based on Masterson, 2004)
conductance = 1000. # (modify 1000 to actual conductance)
bound_sp1 = []

stress_period_data = {0: bound_sp1}
ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=stress_period_data)

#%% # Add drain condition

#Darcy's law states that
#Q = -KA(h1 - h0)/(X1 - X0)
#Where Q is the flow (L3/T)
#K is the hydraulic conductivity (L/T)
#A is the area perpendicular to flow (L2)
#h is head (L)
#X is the position at which head is measured (L)
#Conductance combines the K, A and X terms so that Darcy's law can be expressed as 
#Q = -C(h1 - h0)
#where C is the conductance (L2/T)
# https://water.usgs.gov/nrp/gwsoftware/modflow2000/MFDOC/index.html?drn.htm

# from Masterson, 2004
# C = KWL/M where
#C is hydraulic conductance of the seabed (ft2/d);
#K is vertical hydraulic conductivity of seabed deposits
#(ft/d);
#W is width of the model cell containing the seabed (ft);
#L is length of the model cell containing the seabed (ft);
#and
#M is thickness of seabed deposits (ft).

#The vertical hydraulic conductivity (K) of the seabed
#deposits in most of the study area was assumed to be 1 ft/d,
#which is consistent with model simulations of similar coastal
#discharge areas in other areas on Cape Cod (Masterson and
#others, 1998). In the area occupied by Salt Pond and Nauset
#Marsh, it was assumed that there were thick deposits of lowpermeability
#material (J.A. Colman, U.S. Geological Survey,
#oral commun., 2002) and the vertical hydraulic conductivity
#was set to 0.1 ft/d. The thickness of the seabed deposits was
#assumed to be half the thickness of the model cell containing the
#boundary.

# still using simple conductance
land_cells = active_cells & ~np.isnan(dem_trans) & (dem_trans>mean_sea_level)
landrows, landcols = land_cells.nonzero()
lrcec = {0:np.column_stack([np.zeros_like(landrows),landrows,landcols,dem_trans[land_cells],conductance*np.ones_like(landrows)])} # this drain will be applied to all stress periods
drn = flopy.modflow.ModflowDrn(mf, stress_period_data=lrcec)
                                                                                                                                 

#%% # Add recharge condition

# steady state, units in [m/day]?

rch = flopy.modflow.ModflowRch(mf, nrchop=3, rech=1.4e-3) # from https://pubs.usgs.gov/wsp/2447/report.pdf

#%% # Add OC package to the MODFLOW model
spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}
oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)

#%% # Add PCG package to the MODFLOW model
pcg = flopy.modflow.ModflowPcg(mf)

#%% # Write the MODFLOW model input files
mf.write_input()

#%% # Run the MODFLOW model
success, buff = mf.run_model()
#%%
"""
Post-Processing the Results
Now that we have successfully built and run our MODFLOW model, we can look at the results. 
MODFLOW writes the simulated heads to a binary data output file. 
We cannot look at these heads with a text editor, but flopy has a binary utility that can be used to read the heads. 
The following statements will read the binary head file and create a plot of simulated heads for layer 1:
"""

import flopy.utils.binaryfile as bf
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.subplot(1,1,1,aspect='equal')
hds = bf.HeadFile(os.path.join(work_dir,modelname+'.hds'))
head = hds.get_data(totim=hds.get_times()[-1])
head[head<-100] = np.nan
#levels = np.arange(1,10,1)
extent = (delr/2., Lx - delr/2., Ly - delc/2., delc/2.)
 # headplot = plt.contour(head[0, :, :], levels=levels, extent=extent) #
headplot = plt.contour(head[0, :, :], extent=extent)
plt.xlabel('Lx')
plt.ylabel('Ly')
plt.colorbar(headplot) # plots heads as contours
#plt.colorbar.set_label('heads')
plt.savefig('CheqModel1a.png')
genu.quick_plot(head) # plots heads with color gradient
genu.quick_plot(dem_trans) # plots elevations
#%%
"""
Flopy also has some pre-canned plotting capabilities can can be accessed using the ModelMap class. 
The following code shows how to use the modelmap class to plot boundary conditions (IBOUND), 
plot the grid, plot head contours, and plot vectors:
"""

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, aspect='equal')

hds = bf.HeadFile(modelname+'.hds')
times = hds.get_times()
head = hds.get_data(totim=times[-1])
levels = np.linspace(0, 10, 11)

cbb = bf.CellBudgetFile(modelname+'.cbc')
kstpkper_list = cbb.get_kstpkper()
frf = cbb.get_data(text='FLOW RIGHT FACE', totim=times[-1])[0]
fff = cbb.get_data(text='FLOW FRONT FACE', totim=times[-1])[0]
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
#plt.savefig('CheqModel1b.png')

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
plt.savefig('CheqModel1b.png')
plt.show()
#%%
"""
Post-Processing the Results
Once again, we can read heads from the MODFLOW binary output file, using the flopy.utils.binaryfile module. Included with the HeadFile object are several methods that we will use here: * get_times() will return a list of times contained in the binary head file * get_data() will return a three-dimensional head array for the specified time * get_ts() will return a time series array [ntimes, headval] for the specified cell
Using these methods, we can create head plots and hydrographs from the model results.:
"""

# headfile and budget file objects already created

# Setup contour parameters (levels already set)
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
im = ax.imshow(head[:, 0, :], interpolation='nearest',
               extent=(0, Lx, 0, Ly))
ax.set_title('Simulated Heads')    
    
    
    
    
    
    