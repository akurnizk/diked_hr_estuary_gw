# -*- coding: utf-8 -*-
"""
FORTRAN code, translated
"""

import numpy as np
import matplotlib.pyplot as plt
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
#%% Necessary functions, defined at end of Fortran code
def AR(D):
    return (b0+D*s)*D

def HR(D):
    return (b0+D*s)*D/(b0+2*D*np.sqrt(1+s*s))

def TOP(D):
    return b0+2*D*s

def CENTR(D):
    return D*D*(b0/2+D*s/3)

def DCENDY(D):
    return D*(b0+D*s)

def BW(D):
    return b0 + 2*s*D

def make_sf(Y,V,cmn_squared):
    return abs(V)*V*cmn_squared/HR(Y)**1.333

def make_C2(Y,V,ARi,ARiP1,cmn_squared,s0,grav):
    sf1 = make_sf(Y[:-1],V[:-1],cmn_squared)
    sf2 = make_sf(Y[1:],V[1:],cmn_squared)
    term1 = -dt*(1-alpha)*(grav*ARiP1*(s0-sf2)+grav*ARi*(s0-sf1))
    term2 = -(V[:-1]*ARi+V[1:]*ARiP1)
    term3 = dtx2*(1-alpha)*((V[1:]**2)*ARiP1 + \
                  grav*CENTR(Y[1:]) - \
                  (V[:-1]**2)*ARi-grav*CENTR(Y[:-1]))
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
nsec = 20 # this should be the channel-line distance from the dike to a point with no tidal influence, if grid cells are 1mx1m
np11 = 2*nsec + 2
tlast = 350 # time for transient flow computation (measurements are at 5 min (300s) intervals - use these?)
iprint = 1 # counter for printing results
# READ(20,*)
# READ(20,*) CHL,B0,S,CMN,S0,Q0,Y0,YD,ALPHA,TOL,MAXITER
chl = 2000 # channel length (should be equal to nsec is grid is 1m)
b0 = 10 # channel bottom width - ignore. use dem to find xsec area
s = 2 # channel lateral slope - igore. use dem to find xsec area
cmn = 0.025 # manning's coefficient (use spaulding & grilli?)
s0 = 0.001 # channel bottom slope - ignore. use dem to find xsec area
q0 = 30 # initial steady state discharge (use monthly-averaged fw discharge values)
y0 = 1.58 # uniform flow depth (starting condition?) - may have to make high toss a transient boundary
yd = 1.58 # flow depth at lower end (initial condition) - need to solve so that adjust so that Q_calc_channel_out = Q_dike
alpha = 1 # weighting coefficient
tol = 0.0001 # tolerance for iterations
maxiter = 50 # maximum number of iterations

C1, C2 = ( np.array([np.nan]*(nsec)) for i in range(2))

T = 0 # steady state, initial time
cmn_squared = cmn**2 # gravity units are metric, so no changes are necessary.

# Steady state conditions
c = np.sqrt(grav*AR(y0)/TOP(y0)) # celerity
v0 = q0/AR(y0) # flow velocity
dx = chl/nsec
# dx = cell_spacing # assuming channel length and number of sections are the same
dt = dx/(v0+c) # time step length
dtx2 = 2*dt/dx
yres = y0
i = 0
#np1 = nsec # remove for clarity. In fortran, np1 = nsec+1, python starts at 0 giving extra index

Z = s0 * chl - np.arange(nsec+1) * dx * s0 # bottom elev array
Y = np.ones_like(Z)*y0 # depth array
Y[(Z+y0) < yd] = yd - Z[(Z+y0) < yd]
V = q0/AR(Y)

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
Q_out.append(V*AR(Y))

while (T <= tlast) & (iflag == 0): # time loop, ntimes = tlast/dt
    print("Model time = {0:3.2f} s".format(T))
    ITER = 0
    if (iprint == ip):
        ip = 0
        PREISS_H_out = np.vstack((PREISS_H_out,np.concatenate([['T=',T,'H='],[float(x) for x in (Y+Z)]])))
        PREISS_Q_out = np.vstack((PREISS_Q_out,np.concatenate([['T=',T,'Q='],[float(x) for x in (V*AR(Y))]])))
    T = T + dt
    # !
    # !     GENERATE SYSTEM OF EQUATIONS
    # !
    
    ARi = AR(Y[:-1]) # calculate flow area at upstream section
    ARiP1 = AR(Y[1:]) # calculate flow area at downstream section
    C1 = dtx2*(1-alpha)*(ARiP1*V[1:]-ARi*V[:-1])-ARi-ARiP1
    C2 = make_C2(Y,V,ARi,ARiP1,cmn_squared,s0,grav)
      
    
    SUMM = tol+10
    for L in range(1,1000):
        plt.plot(Y,".",label=L-1)
        if (SUMM > tol):
            EQN = np.zeros((np11,np11+1),dtype=float) # should generate the same array?
            ITER = ITER+1    
            # !    
            # !        INTERIOR NODES
            # !    
            
            ARi = AR(Y[:-1]) # calculate flow area at upstream section
            ARiP1 = AR(Y[1:]) # calculate flow area at downstream section
            row_inds1 = 2*np.arange(nsec,dtype=int)+1 # every other row, starting at 1 (2nd row)
            EQN[row_inds1,np11]=-(ARi+ARiP1+dtx2*alpha*(V[1:]*ARiP1-V[:-1]*ARi)+C1) # sets last column
            
            sf1 = make_sf(Y[:-1],V[:-1],cmn_squared)
            sf2 = make_sf(Y[1:],V[1:],cmn_squared)
            term1 = term1 = dtx2*alpha*((V[1:]**2)*ARiP1 + grav*CENTR(Y[1:])-(V[:-1]**2)*ARi-grav*CENTR(Y[:-1]))
            term2 = -alpha*dt*grav*((s0-sf2)*ARiP1+(s0-sf1)*ARi)
            EQN[row_inds1+1,np11] = -(V[:-1]*ARi+V[1:]*ARiP1+term1+term2+C2) # every other row, starting at 2 (3rd row)
            
            daY1 = TOP(Y[:-1])
            daY2 = TOP(Y[1:])
            EQN[row_inds1,row_inds1-1] = daY1*(1-dtx2*alpha*V[:-1])
            EQN[row_inds1,row_inds1] = -dtx2*alpha*ARi
            EQN[row_inds1,row_inds1+1] = daY2*(1+dtx2*alpha*V[1:])
            EQN[row_inds1,row_inds1+2] = dtx2*alpha*ARiP1
            
            dcdY1 = DCENDY(Y[:-1])
            dcdY2 = DCENDY(Y[1:])
            dsdV1 = 2*V[:-1]*cmn_squared/HR(Y[:-1])**1.333
            dsdV2 = 2*V[1:]*cmn_squared/HR(Y[1:])**1.333
            
            PERi = ARi/HR(Y[:-1])
            term1 = 2*np.sqrt(1+s**2)*ARi - daY1*PERi
            term2 = HR(Y[:-1])**0.333*ARi**2
            dsdY1 = 1.333*V[:-1]*abs(V[:-1])*cmn_squared*term1/term2
            
            PERiP1 = ARiP1/HR(Y[1:])
            term1 = 2*np.sqrt(1+s**2)*ARiP1-daY2*PERiP1
            term2 = (HR(Y[1:])**0.333)*(ARiP1**2)
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
            # !       EQN(1,1)    = 1.0
            # !       EQN(2,NP11) = (V(1) - V0)                               ! ok


            # !
            # !         DOWNSTREAM END (NO OUTFLOW)
            # !
            EQN[-1,-2] = 1.
            EQN[-1,np11] = 0. - V[-1]
            # !       
            # !         DOWNSTREAM END (Y given)
            # !
            # !       EQN(2*NP1,2*NP1) = 1.
            # !       EQN(2*NP1,NP11) =  Y(NP1) - YD                          ! ok
       
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
    Q_out.append(V*AR(Y))

H_out = np.array(H_out)
Q_out = np.array(Q_out)

# IF(IFLAG.EQ.1) WRITE(6,"('MAXIMUM NUMBER OF ITERATIONS EXCEEDED')")
# STOP
# END

if (iflag == 1):
    print("Maximum number of iterations exceeded")
