# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:16:25 2019

@author: akurnizk
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:47:36 2018

@author: akurnizk
"""

import utm
import math
import flopy
import sys,os
import numpy as np
import pandas as pd
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf

cgw_code_dir = 'E:\python' # Location of BitBucket folder containing cgw folder
sys.path.insert(0,cgw_code_dir)

from moviepy.editor import *
from shapely.geometry import Point
from cgw.utils import general_utils as genu
from cgw.utils import feature_utils as shpu
from cgw.utils import raster_utils as rastu
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Assign name and create modflow model object

modelname = 'CheqModel1_NWT_test'
work_dir = r'E:\Herring_NWT_test'
mfexe = "MODFLOW-NWT.exe"
# mf = flopy.modflow.Modflow(modelname, exe_name='mf2005',model_ws=work_dir)
mf = flopy.modflow.Modflow(modelname, exe_name=mfexe,model_ws=work_dir, version='mfnwt') # fix me - playing with obtaining NWT solver
swt = flopy.seawat.Seawat(modelname, exe_name='swtv4')
print(swt.namefile)

mean_sea_level = 0.843 # Datum in meters at closest NOAA station (8447435), Chatham, Lydia Cove MA
# https://tidesandcurrents.noaa.gov/datums.html?units=1&epoch=0&id=8447435&name=Chatham%2C+Lydia+Cove&state=MA

#%% Example of making a MODFLOW-like grid from a shapefile

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
Plot active cells (need to figure out how to plot active cells in model coordinates in subplot 2)
"""
# fig,ax = genu.plt.subplots(1,2)
# ax[0].set_xlabel('column #')
# ax[0].set_ylabel('row #')
# genu.quick_plot(active_cells.astype(int),ax=ax[0]) # in row, column space
# c1=ax[1].pcolormesh(cc[0],cc[1],active_cells.astype(int)) # in model coordinates
# genu.plt.colorbar(c1,ax=ax[1],orientation='horizontal')
# ax[1].set_xlabel('X [m]')
# ax[1].set_ylabel('Y [m]')

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

genu.quick_plot(dem_trans, vmin=0, vmax=60) # plots elevations (in meters), with only active cells
                            # plot needs x, y labels and colorbar label (elevation)
                            
#%% DEM model inputs

Lx = np.amax(dem_X)-np.amin(dem_X) # should be DEM size (10 km)
Ly = np.amax(dem_Y)-np.amin(dem_Y) # should be DEM size (10 km)
# ztop = 0. # replaced ztop everywhere with dem_trans
zbot = -100 # if bottom of model is horizontal, approx. bedrock (Masterson_2004 - zone of transition (ZOT) on p.68)
nlay = 1 # 1 layer model
nrow, ncol = cc[0].shape
delr = cell_spacing # delr = Lx/ncol in Tutorial 1
delc = cell_spacing # delc = Ly/now in Tutorial 1
delv = (dem_trans - zbot) / nlay
botm = zbot # from Tutorial 1, botm = np.linspace(ztop, zbot, nlay + 1)

#%% Time Stepping
"""
Time Stepping For tidal variations over ONE DAY
Need to lengthen for seasonal variations
"""

# Time step parameters
total_length = 2 # days
dt = 1 # stress period time step, hrs
perlen_days = dt/24. # stress period time step, days

nper = int(total_length/perlen_days) # the number of stress periods in the simulation
# nstp_default = int(dt/0.5) # stress period time step divided by step time length (to better interpolate tidal changes, set to 0.5 hrs)
nstp_default = int(dt) # try 1 time step per period
perlen = [perlen_days]*nper # length of a stress period; each item in the matrix is the amount 
                            # of elapsed time since the previous point (need to change the first)
perlen[0] = 1 # set first step as steady state
steady = [False]*nper
steady[0] = True # first step steady state
nstp = [nstp_default]*nper # number of time steps in a stress period
nstp[0] = 1 # why?

"""
Changing sea-level
https://tidesandcurrents.noaa.gov/noaatidepredictions.html?id=8446613
Using generalized sinusoidal equation for sea-level change with respect to time. Amplitude ~ 2m
Need to fix formula to better represent tidal predictions (can also use a more simplified curve to represent seasonal sea levels)
"""
meas_times_hrs = []
sea_levels = []
next_time = 0
for per in range(nper):
    meas_times_hrs.append([next_time*24])
    sea_levels.append([mean_sea_level + np.sin(86400*np.pi*per*perlen_days/22350)]) # 22350 represents the time, in seconds, 
                                                                                   # to go from high tide to low tide (half of a period).
    next_time = next_time + perlen[1]

plt.plot(meas_times_hrs, sea_levels) # plots sea level changing over 40 time periods.
plt.xlabel('time (hrs)')
plt.ylabel('sea level (m)')
plt.axis([24, 48, -0.5, 2.0])
plt.grid(True)
plt.show()

sl_max = np.max(sea_levels) # maximum sea level from array generated by sea_levels formula
sl_min = np.min(sea_levels) # minimum sea level from array generated by sea_levels formula
max_amplitude = sl_max-sl_min

#%% Create the discretization (DIS) object

dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                               top=dem_trans, botm=botm, nper=nper, perlen=perlen, nstp=nstp, steady=steady) # Tutorial 1 & 2, botm=botm[1:]
                                                              
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

ibound = np.ones((nlay, nrow, ncol), dtype=np.int32) # make array of ones (active cells) with shape = nlay, nrow, ncol
ibound[:,~active_cells] = 0 # far offshore cells are inactive (set to zero)
# ibound[0,dem_trans<mean_sea_level] = -1 # fixed head for everything less than msl (0 for layer) (only leave in for constant head)
ibound[:,np.isnan(dem_trans)] = 0 # nan cells are inactive

genu.quick_plot(ibound) # plots boundary condition: 1 is above mean sea level (msl), 0 is msl, -1 is under msl.
# needs title and labels

strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
active_dem_heights = dem_trans[active_cells & ~np.isnan(dem_trans)]
strt[0, active_cells & ~np.isnan(dem_trans)] = active_dem_heights # start with freshwater at surface elevation
strt[0, dem_trans<mean_sea_level] = mean_sea_level # start with surface water at mean sea level

genu.quick_plot(strt) # plots starting condition (which will look like dem_trans or sea_level for entire grid)
# needs title and labels

bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

#%% Variables for the LPF (Layer-Property Flow) Package

# added 3/8/19 - creates matrix where hydraulic conductivities (hk = horiz, vk = vert) can be implemented
# should I change the hk, vk, ss, and sy values for surface water bodies?

hk1 = np.ones((nlay,nrow,ncol), np.float)
hk1[:,:,:]=10. # everything set to 10 - use data? calculate?
vka1 = np.ones((nlay,nrow,ncol), np.float)
vka1[:,:,:]=10. # everything set to 10.

# Add LPF package to the MODFLOW model

lpf = flopy.modflow.ModflowLpf(mf, laytyp=1, hk=hk1, vka=vka1, ss=1e-05, sy=0.25, storagecoefficient=True, ipakcb=53) # sy and ss from Masterson_2004 p.59
# laytyp = 1 means unconfined, ss is confined storage coefficient (not specific storage) if storagecoefficient=True, sy = specific yield

#%% Transient General-Head Boundary Package

"""
Example (Added 5/28/19 from Bakker_2016): 
First, we will create the GHB object, which is of the following type: flopy.modflow.ModflowGhb.

The key to creating Flopy transient boundary packages is recognizing that the 
boundary data is stored in a dictionary with key values equal to the 
zero-based stress period number and values equal to the boundary conditions 
for that stress period. For a GHB the values can be a two-dimensional nested 
list of [layer, row, column, stage, conductance]:
"""

# using single conductance value (see drain for modification based on Masterson, 2004)
conductance = 1000. # (modify 1000 to actual conductance, calculated on p. 55 Master_2004)
    
for i in range(nper): # there are 40 stress periods (nper)
    sea_level = sea_levels[i]
    stage_cells = active_cells & ~np.isnan(dem_trans) & (dem_trans <= sea_level)
    stagerows, stagecols = stage_cells.nonzero()
    # stage = sea_level-dem_trans[stage_cells] # this is if stage is relative to sea floor
    stage = np.zeros_like(dem_trans[stage_cells])
    stage[:] = sea_level[0]   # add freshwater equivalent head eventually                             
    locals()["bound_sp"+str(i+1)] = np.column_stack([np.zeros_like(stagerows), stagerows, stagecols, stage, conductance*np.ones_like(stagerows)]) # boundaries for stress period 1

print('Adding ', len(bound_sp1), 'GHBs for stress period 1 (steady state).')    
lrcsc = {}
for i in range(nper):
    lrcsc.update({i: locals()["bound_sp"+str(i+1)]})
    
ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=lrcsc, options = ['NOPRINT']) # for using general head
# chd = flopy.modflow.ModflowChd(mf, stress_period_data=lrcsc, options = ['NOPRINT']) # for using constant head
    
#%% Add drain condition

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

lrcec = {}    
for i in range(nper): # there are 40 stress periods (nper)
    sea_level = sea_levels[i]
    
    land_cells = active_cells & ~np.isnan(dem_trans) & (dem_trans>sea_level)
    landrows, landcols = land_cells.nonzero() # removes all 'false' values and assigns 'true' values to rows and columns
    lrcec.update({i:np.column_stack([np.zeros_like(landrows),landrows,landcols,dem_trans[land_cells],conductance*np.ones_like(landrows)])}) # this drain will be applied to all stress periods

drn = flopy.modflow.ModflowDrn(mf, stress_period_data=lrcec, options = ['NOPRINT'])                                                                                                                          

#%% Add recharge condition

# steady state, units in [m/day]?

rch = flopy.modflow.ModflowRch(mf, nrchop=3, rech=1.4e-3) # from USGS Water-Supply Paper 2447, Masterson_1997
                                                            # https://pubs.usgs.gov/wsp/2447/report.pdf

#%% Add OC package to the MODFLOW model

stress_period_data = {}
for kper in range(nper):
    for kstp in range(nstp[kper]):
        stress_period_data[(kper, kstp)] = ['save head',
                                            'save budget',
                                            'print budget']
oc = flopy.modflow.ModflowOc(mf, stress_period_data=stress_period_data,
                             compact=True)

#%% Add PCG package to the MODFLOW model
nwt = flopy.modflow.ModflowNwt(mf)

#%% Write the MODFLOW model input files
mf.write_input()

# mf.write_input(SelPackList=['BAS6']) # to run a single package (in this case the bas package)

#%% Run the MODFLOW model
success, mfoutput = mf.run_model(silent=False, pause=False, report=True)
if not success:
    raise Exception('MODFLOW did not terminate normally.')

# #%% Graphs
# """
# Post-Processing the Results
# Now that we have successfully built and run our MODFLOW model, we can look at the results. 
# MODFLOW writes the simulated heads to a binary data output file. 
# We cannot look at these heads with a text editor, but flopy has a binary utility that can be used to read the heads. 
# The following statements will read the binary head file and create a plot of simulated heads for layer 1:
# """

# plt.subplot(1,1,1,aspect='equal')
# hds = bf.HeadFile(os.path.join(work_dir,modelname+'.hds'))
# time = 0
# head = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
# head[head<-100] = np.nan
# # levels = np.arange(1,10,1)
# extent = (delr/2., Lx - delr/2., Ly - delc/2., delc/2.)
# # headplot = plt.contour(head[0, :, :], levels=levels, extent=extent) # no need for levels?
# plt.xlabel('Lx')
# plt.ylabel('Ly')
# headplot = plt.contour(head[0, :, :], extent=extent)
# plt.colorbar(headplot) # plots heads as contours 
# # plt.colorbar.set_label('heads')
# plt.savefig(work_dir + '\CheqModel1a.png')

# mytimes = np.arange(0, nper, 1) # for plotting nper-1 graphs
# for time in mytimes: 
#     head = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
#     head[head<-100] = np.nan
#     Cheqheadplot = genu.quick_plot(head, vmin=0.0, vmax = 2.5) # plots heads with color gradient
#     Cheqheads = Cheqheadplot.get_figure()
#     Cheqheads.suptitle(str(time)+" hrs") # how does one label the figure?
#     Cheqheads.savefig(work_dir+"\CheqHeads\CheqHeads0"+str(f"{time:02d}")+".png")

# #%% Steps along 1 day sea-level plot

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
#     plt.savefig(work_dir+"\CheqSeaLevel\CheqSeaLevel0"+str(f"{time:02d}")+".png")
#     plt.show()    
    

# #%% Animating Images

# # img = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png',
# #        '7.png', '8.png', '9.png', '10.png', '11.png']

# # clips = [ImageClip(m).set_duration(0.5)
# #       for m in img]

# # concat_clip = concatenate_videoclips(clips, method="compose")
# # concat_clip.write_videofile("test.mp4", fps=24)


# # # Extract elevation data at pond
# # pond_elev=dem_xr.data[0]
# # pond_elev = pond_elev[pond_mask]
# # median_pond_elev = np.median(pond_elev)

# # # Export mask as shapefile

# # out_shp_fname = os.path.join(cape_dir,'data','Coast-0.shp')

# # nrow,ncol,gt = pgwl.get_rast_info(dem_fname)

# # to_poly_dict = {'Z':all_ponds_mask,'in_proj':prj,'out_shp':out_shp_fname,'gt':gt}
# # pgwl.raster_to_polygon_gdal(**to_poly_dict)

# # # Filename example

# # file_fmt = 'land_sl{0:4.2f}mbpsl_time{1:3.2f}kBP.shp'

# # sl = -13.23
# # time= -6.5

# # out_filename = file_fmt.format(abs(sl),abs(time))
# # print(out_filename)

# # Make a movie of head map (.png files must be saved with Tools>Preferences>IPython Console>Graphics>Backend set to Automatic),
#     # otherwise the files will not be the same size
# os.chdir(r'E:\Herring\CheqHeads')   
# img = os.listdir(r'E:\Herring\CheqHeads')

# clip = mpy.ImageSequenceClip(img, fps=2)
# os.chdir(r'E:\Herring\CheqHeads') 
# clip.write_gif('CheqHeads_oneday_unconf_48.gif', fps=12)

# # Make a movie of sea levels
# os.chdir(r'E:\Herring\CheqSeaLevel')   
# img = os.listdir(r'E:\Herring\CheqSeaLevel')

# clip = mpy.ImageSequenceClip(img, fps=2)
# os.chdir(r'E:\Herring\CheqSeaLevel') 
# clip.write_gif('CheqSeaLevel_oneday_unconf_48.gif', fps=12)

# #%% Non-negligible effects of tides

# stage_cells = active_cells & ~np.isnan(dem_trans) & (dem_trans <= sl_max)

# head_all = hds.get_alldata().squeeze()
# head_all[head_all<-100] = np.nan
# headmax = np.nanmax(head_all, axis=0) 
# headmin = np.nanmin(head_all, axis=0)

# head_range = headmax-headmin
# # head_range[stage_cells] = sl_max-sl_min

# genu.quick_plot(headmax, vmin=0, vmax=2.5)
# genu.quick_plot(headmin, vmin=0, vmax=2.5)
# genu.quick_plot(head_range, vmin=0, vmax=0.1)

# tidalheads = head_range.copy()
# tidalheads[tidalheads<0.05] = np.nan
# tidalheads[stage_cells] = np.nan
# genu.quick_plot(tidalheads, vmin=0, vmax=2)

# dx,dy,gt = rastu.get_rast_info(dem_fname)
# projwkt = rastu.load_grid_prj(dem_fname)
# fname = work_dir + '\CheqTidalInfluence_48hr.tif'
# rastu.write_gdaltif(fname,cc_proj[0],cc_proj[1],tidalheads,proj_wkt=projwkt)

# # loop later when files get too large

# # head = hds.get_data(totim=hds.get_times()[0]) # steady-state head (0th time step)
# # head[head<-100] = np.nan
# # headmax = head
# # headmin = head

# # mytimes = np.arange(0, 24, 1) # for plotting 23 graphs
# # for time in mytimes: 
# #     head = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
# #     for i in range(nrow):
# #       for j in range(ncol):
# #         if head(0,i,j)>headmax: headmax(0,i,j)=head(0,i,j)
# #         if head(0,i,j)<headmin: headmin(0,i,j)=head(0,i,j)
    
# #     Cheqheadplot_cons = genu.quick_plot(head, vmin=0.0, vmax = 2.5) # plots heads under consideration with color gradient
# #     Cheqheads_cons = Cheqheadplot_cons.get_figure()
# #     Cheqheads_cons.suptitle(str(time)+" hrs") # how does one label the figure?
# #     Cheqheads_cons.savefig(work_dir+"\CheqHeads\CheqHeads0"+str(f"{time:02d}")+".png")

# #%% Herring River Transects

# dike_transect_y = np.arange(410,465,5)
# dike_transect_x = np.arange(210,188,-2)
# dike_transect_cells = np.vstack((dike_transect_x, dike_transect_y)).T

# upstrm_transect_y = np.arange(540,551,1)
# upstrm_transect_x = np.arange(300,223,-7)
# upstrm_transect_cells = np.vstack((upstrm_transect_x, upstrm_transect_y)).T

# for time in mytimes:
#     head = hds.get_data(totim=hds.get_times()[time]) # steady-state head (0th time step)
#     head[head<-100] = np.nan
#     dike_transect_heads = []
#     dike_transect_elev = []
#     for irow, icol in dike_transect_cells:
#         temp_head = head[:,irow,icol]
#         dike_transect_heads.append(temp_head[0])
#         temp_elev = dem_trans[irow,icol]
#         dike_transect_elev.append(temp_elev)
#         plt.plot(dike_transect_heads)
#         plt.plot(dike_transect_elev)
#         plt.ylabel('elevation (m)')
#         plt.xlabel('transect length (m)')
 
# plt.figure()
# plt.step(dike_transect_x,dike_transect_heads,where='mid')    
# plt.ylabel('elevation (m)')
# plt.xlabel('transect length (m)')

# cc_proj[0][410,210] # for x-coordinate in UTM
# cc_proj[1][410,210] # for y-coordinate in UTM

# #%%
# """
# Flopy also has some pre-canned plotting capabilities can can be accessed using the ModelMap class. 
# The following code shows how to use the modelmap class to plot boundary conditions (IBOUND), 
# plot the grid, plot head contours, and plot vectors:
# """

# # fig = plt.figure(figsize=(10,10))
# # ax = fig.add_subplot(1, 1, 1, aspect='equal')

# # hds = bf.HeadFile(modelname+'.hds')
# # times = hds.get_times()
# # head = hds.get_data(totim=times[-1])
# # levels = np.linspace(0, 10, 11)

# # cbb = bf.CellBudgetFile(modelname+'.cbc')
# # kstpkper_list = cbb.get_kstpkper()
# # frf = cbb.get_data(text='FLOW RIGHT FACE', totim=times[-1])[0]
# # fff = cbb.get_data(text='FLOW FRONT FACE', totim=times[-1])[0]
# #%%
# """
# The pre-canned plotting doesn't seem to be able to allow averaging to reduce nrow and ncol
# on the plot, making it difficult to plot a large grid. The commented section below uses the
# modelmap class from Tutorial 1, followed by use of the plotting from the Henry Problem.
# """

# #modelmap = flopy.plot.ModelMap(model=mf, layer=0)
# #qm = modelmap.plot_ibound()
# #lc = modelmap.plot_grid() # Need to fix grid to have fewer rows and columns
# #cs = modelmap.contour_array(head, levels=levels)
# #quiver = modelmap.plot_discharge(frf, fff, head=head)
# #plt.savefig('CheqModel1b.png')

# """
# # Load data (when implementing SEAWAT)
# ucnobj = bf.UcnFile('MT3D001.UCN', model=swt)
# times = ucnobj.get_times()
# concentration = ucnobj.get_data(totim=times[-1])
# """

# # Average flows to cell centers
# qx_avg = np.empty(frf.shape, dtype=frf.dtype)
# qx_avg[:, :, 1:] = 0.5 * (frf[:, :, 0:ncol-1] + frf[:, :, 1:ncol])
# qx_avg[:, :, 0] = 0.5 * frf[:, :, 0]
# qy_avg = np.empty(fff.shape, dtype=fff.dtype)
# qy_avg[1:, :, :] = 0.5 * (fff[0:nlay-1, :, :] + fff[1:nlay, :, :])
# qy_avg[0, :, :] = 0.5 * fff[0, :, :]

# # Make the plot
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# #ax.imshow(concentration[:, 0, :], interpolation='nearest',
# #           extent=(0, Lx, 0, Ly))

# y, x, z = dis.get_node_coordinates()
# X, Y = np.meshgrid(x, y)
# iskip = 3
# ax.quiver(X[::iskip, ::iskip], Y[::iskip, ::iskip],
#            qx_avg[::iskip, 0, ::iskip], -qy_avg[::iskip, 0, ::iskip],
#            color='k', scale=5, headwidth=3, headlength=2,
#            headaxislength=2, width=0.0025)
# plt.savefig('CheqModel1b.png')
# plt.show()
# #%%
# """
# Post-Processing the Results
# Once again, we can read heads from the MODFLOW binary output file, using the flopy.utils.binaryfile module. 
# Included with the HeadFile object are several methods that we will use here: * get_times() will return a 
# list of times contained in the binary head file * get_data() will return a three-dimensional head array for 
# the specified time * get_ts() will return a time series array [ntimes, headval] for the specified cell

# Using these methods, we can create head plots and hydrographs from the model results.:
# """

# # Create the headfile and budget file objects
# times = hds.get_times()
# cbb = bf.CellBudgetFile(modelname+'.cbc')

# # Setup contour parameters 
# levels = np.linspace(0, 10, 11)
# extent = (delr/2., Lx - delr/2., delc/2., Ly - delc/2.)
# print('Levels: ', levels)
# print('Extent: ', extent)

# # Make the plots

# #Print statistics
# print('Head statistics')
# print('  min: ', head.min())
# print('  max: ', head.max())
# print('  std: ', head.std())

# """
# Again, commented out section using modelmap
# """

# ## Flow right face and flow front face already extracted
# ##%%
# ##Create the plot
# #f = plt.figure()
# #plt.subplot(1, 1, 1, aspect='equal')
# #
# #
# #modelmap = flopy.plot.ModelMap(model=mf, layer=0)
# #qm = modelmap.plot_ibound()
# ## 
# ## lc = modelmap.plot_grid()
# #qm = modelmap.plot_bc('GHB', alpha=0.5)
# #cs = modelmap.contour_array(head, levels=levels)
# #plt.clabel(cs, inline=1, fontsize=10, fmt='%1.1f', zorder=11)
# #quiver = modelmap.plot_discharge(frf, fff, head=head)
# #
# #mfc='black'
# #plt.plot(lw=0, marker='o', markersize=8,
# #         markeredgewidth=0.5,
# #         markeredgecolor='black', markerfacecolor=mfc, zorder=9)
# #plt.savefig('CheqModel2-{}.png')
    
# """
# From Henry Problem
# """

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1, aspect='equal')
# im = ax.imshow(head[:, 500, 500:600], interpolation='nearest',
#                extent=(0, Lx, 0, Ly))
# ax.set_title('Simulated Heads')    
    
    
    
    
    