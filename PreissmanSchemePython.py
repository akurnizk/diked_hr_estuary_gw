"""
FORTRAN code, translated
"""

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
# IMPLICIT NONE


# INTEGER I,J,K,L,LL,IP,ITER,MAXITER
# INTEGER K1,K2              
# INTEGER NSEC,IPRINT,NP1,NP11,IFLAG            
# REAL G,TLAST,CHL,CMN,S0,Q0,Y0,ALPHA
# REAL DT,DX,C,CMN2,T,ARI,ARIP1,PERI,PERIP1,YRES,YD
# REAL SF1,SF2,V0,TOL,TERM1,TERM2,TERM3,DTX2,SUMM  
# REAL DSDY1,DSDY2,DSDV1,DSDV2,DCDY1,DCDY2,DAY1,DAY2
# REAL Y(60),V(60),C1(60),C2(60),DF(60),EQN(124,125),Z(60)
"""
Initialize these arrays below variables based on nsec
"""
# REAL HR,AR,TOP,CENTR,DCENDY             
                                                  
# CHARACTER (len = 15)::  UNITS                     
# character (len = 80):: FILE_NAME                  
                                                  
# REAL B0,S                                         
# COMMON /CHANNEL/B0,S                              

# ! INPUT DATA
# call getarg(1,FILE_NAME)    
# OPEN(20,FILE=FILE_NAME)
# READ(20,*)
# READ(20,*) G,NSEC,TLAST,IPRINT
grav = 9.81 # m/s^2
nsec = 20 # this should be the channel-line distance from the dike to a point with no tidal influence, if grid cells are 1mx1m
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
# CLOSE(20)

Z = np.array([np.nan]*(nsec+1)) # initialized HR_bot elev array
Y, V = (Z.copy() for i in range(2))
C1, C2 = (Z[:-1].copy() for i in range(2))

# OPEN(10,FILE="G_H_out.csv")
# OPEN(11,FILE="G_Q_out.csv")

# T = 0
T = 0 # steady state, initial time
# IF(G.GT.10) THEN
#   CMN2 = (CMN*CMN)/2.22
# ELSE
#   CMN2 = CMN*CMN
# END IF
cmn_squared = cmn**2 # gravity units are metric, so no changes are necessary.

"""
Necessary functions, defined at end of Fortran code
"""
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

"""    ******************************************************************
!      SIMULTANEOUS SOLUTION OF THE SYSTEM OF EQUATIONS
"""
def MATSOL(N,A):

    X = np.zeros((N),dtype=int)
    NROW = np.array(range(0,42),dtype=int)
    
    i = 0
    while (i <= N-2): # loop through columns
        AMAX = A[NROW[i],i].copy() # 
        j = i
        ip = i
        while (j <= N-1): # loop through rows
            if(abs(A[NROW[j],i]) > AMAX):
                AMAX = abs(A[NROW[j],i])
                ip = j
            j = j+1
        
        # AMAX2 = np.max(np.abs(A[:,i]))
        
        # print(AMAX,AMAX2)
        
        if(abs(AMAX) <= 1E-08): # GO TO 100
            print('Singular matrix --> No unique solution exists')
            return X
        
        if(NROW[i] != NROW[ip]):
            NC = NROW[i].copy()
            NROW[i] = NROW[ip].copy()
            NROW[ip] = NC.copy()
        j = i+1
        while (j <= N-1):
            COEF = A[NROW[j],i]/A[NROW[i],i]
            jj = i+1
            while (jj <= N):
                A[NROW[j],jj] = A[NROW[j],jj] - COEF*A[NROW[i],jj]
                jj = jj + 1
            j = j+1
        i = i+1
        # print(i,NROW[i])
    
    if(abs(A[NROW[N-1],N-1]) <= 1E-08): # GO TO 100
        print('Singular matrix --> No unique solution exists')
        return X
    
    X[N-1] = A[NROW[N-1],N]/A[NROW[N-1],N-1]
    i = N-2
    while (i >= 0):
        # SUMM = 0.0
        # j = i+1
        
        SUMM = np.sum(A[NROW[i],:N]*X[:N])
        
        # while (j <= N-1):
        #     SUMM = A[NROW[i],j]*X[j] + SUMM
        #     j = j+1
        # print(SUMM,SUMM2)
        
        X[i] = (A[NROW[i],N] - SUMM)/A[NROW[i],i]
        i = i-1
    return X

"""
!====================================================================
!             NORMAL DEPTH
!====================================================================
"""
def NORMAL_D(YNORM,Q,CMAN,B0,S,S0):
  
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

# !
# !     STEADY STATE CONDITIONS
# !
"""
Steady State Conditions
"""
# C = SQRT(G*AR(Y0)/TOP(Y0))
# V0 = Q0/AR(Y0)
# DX = CHL/NSEC
# DT = DX/(V0+C)
# DTX2 = 2*DT/DX
# YRES = Y0
# I = 1
# NP1 = NSEC + 1
c = np.sqrt(grav*AR(y0)/TOP(y0)) # celerity
v0 = q0/AR(y0) # flow velocity
dx = chl/nsec
# dx = cell_spacing # assuming channel length and number of sections are the same
dt = dx/(v0+c) # time step length
dtx2 = 2*dt/dx
yres = y0
i = 0
np1 = nsec

# DO WHILE(I.LE.NP1)
#   Z(I) = S0*CHL - (I-1)*DX*S0  ! RIVERBED ELEVATION
#   IF((Z(I)+Y0).LT.YD)THEN
#     Y(I) = YD - Z(I)
#   ELSE
#     Y(I) = Y0
#   END IF
#   V(I) = Q0/AR(Y(I))
#   I = I+1
# END DO
# IF(G.LE.10) UNITS = 'SI'
# IF(G.GT.10) UNITS = 'ENGLISH'
while (i <= np1):
    Z[i] = s0*chl - (i)*dx*s0 # riverbed elevation (I should be able to use DEM for this)
    if (Z[i]+y0 < yd):
        Y[i] = yd - Z[i]
    else:
        Y[i] = y0
    V[i] = q0/AR(Y[i])
    i = i+1

# WRITE(*,"(5X,'PREISSMANN SCHEME')")
# WRITE(*,"(/5X,'UNITS = ',A10)")UNITS
# WRITE(*,"(5X,'CHANNEL LENGTH        = ',F7.2)")CHL
# WRITE(*,"(5X,'CHANNEL BOTTOM WIDTH  = ',F7.2)")B0
# WRITE(*,"(5X,'CHANNEL LATERAL SLOPE = ',F7.2)")S
# WRITE(*,"(5X,'CHANNEL BOTTOM SLOPE  = ',F7.2)")S0
# WRITE(*,"(5X,'MANNING  n            = ',F7.2)")CMN
# WRITE(*,"(5X,'INITIAL STEADY STATE DISCHARGE = ',F7.2)")Q0
# WRITE(*,"(5X,'UNIFORM FLOW DEPTH             = ',F7.2)")Y0
# WRITE(*,"(5X,'FLOW DEPTH AT LOWER END (IC)   = ',F7.2)")YD
# WRITE(*,"(5X,'NUMBER OF CHANNEL SECTIONS     = ',I3)")NP1
# WRITE(*,"(5X,'SIMULATION LENGHT              = ',F7.2)")TLAST
# !
# IFLAG = 0
# IP = IPRINT
iflag = 0
ip = iprint
# WRITE(10,"('T=,',F8.3,',Z=',60(',',F6.2))")0.,(Z(I),I = 1,NP1)
PREISS_H_out = np.concatenate([['T=',T,'Z='],[float(x) for x in Z]])
PREISS_Q_out = PREISS_H_out.copy() # added this to have a header row with the bottom elevations for the discharge array.

# !
# !     COMPUTE TRANSIENT CONDITIONS
# !
"""
Compute Transient Conditions
"""
# DO WHILE(T.LE.TLAST .AND. IFLAG.EQ.0)
#   ITER = 0
#   IF(IPRINT.EQ.IP) THEN
#     IP = 0
#     WRITE(10,"('T=,',F8.3,',H=',60(',',F6.2))")T,(Y(I) + Z(I),I = 1,NP1)
#     WRITE(11,"('T=,',F8.3,',Q=',60(',',F6.1))")T,(V(I)*AR(Y(I)),I = 1,NP1)
#   END IF
#   T = T+DT
while (T <= tlast) & (iflag == 0):
    ITER = 0
    if (iprint == ip):
        ip = 0
        PREISS_H_out = np.vstack((PREISS_H_out,np.concatenate([['T=',T,'H='],[float(x) for x in (Y+Z)]])))
        PREISS_Q_out = np.vstack((PREISS_Q_out,np.concatenate([['T=',T,'Q='],[float(x) for x in (V*AR(Y))]])))
    T = T + dt
# !
# !     GENERATE SYSTEM OF EQUATIONS
# !
    """
    Generate System of Equations
    """
#   I = 1
#   DO WHILE(I.LE.NSEC)
#     ARI = AR(Y(I))
#     ARIP1 = AR(Y(I+1))
#     C1(I)=DTX2*(1-ALPHA)*(ARIP1*V(I+1)-ARI*V(I))-ARI-ARIP1
#     SF1 = ABS(V(I))*V(I)*CMN2/HR(Y(I))**1.333
#     SF2 = ABS(V(I+1))*V(I+1)*CMN2/HR(Y(I+1))**1.333
#     TERM1 = -DT*(1-ALPHA)*(G*ARIP1*(S0-SF2)+G*ARI*(S0-SF1))
#     TERM2 = -(V(I)*ARI+V(I+1)*ARIP1)
#     TERM3 = DTX2*(1-ALPHA)*(V(I+1)**2*ARIP1+G*CENTR(Y(I+1))-  &
#     	      V(I)**2*ARI-G*CENTR(Y(I)))
#     C2(I) = TERM1 + TERM2 + TERM3
#     I = I+1
#   END DO
    i = 0
    while (i <= nsec-1):
        ARi = AR(Y[i]).copy()
        ARiP1 = AR(Y[i+1]).copy()
        C1[i]=dtx2*(1-alpha)*(ARiP1*V[i+1]-ARi*V[i])-ARi-ARiP1
        sf1 = abs(V[i])*V[i]*cmn_squared/HR(Y[i])**1.333
        sf2 = abs(V[i+1])*V[i+1]*cmn_squared/HR(Y[i+1])**1.333
        term1 = -dt*(1-alpha)*(grav*ARiP1*(s0-sf2)+grav*ARi*(s0-sf1))
        term2 = -(V[i]*ARi+V[i+1]*ARiP1)
        term3 = dtx2*(1-alpha)*(V[i+1]**2*ARiP1+grav*CENTR(Y[i+1])-V[i]**2*ARi-grav*CENTR(Y[i]))
        C2[i] = term1 + term2 + term3
        i = i+1     
#   NP11 = 2*NP1 + 1
#   SUMM = TOL+10
#   DO L = 1,1000
# !     WRITE(*,*)L,SUMM,TOL 
#      IF(SUMM.GT.TOL)THEN
#        I = 1
#        DO WHILE(I.LE.2*NP1)
#          J = 1
#          DO WHILE(J.LE.NP11)
#            EQN(I,J) = 0.0
#            J = J+1
#          END DO
#          I = I+1
#        END DO
#        ITER = ITER+1
    """
    Is this just filling an array of zeros?
    """ 
    """
    My code:
    """
    np11 = 2*np1 + 2
    SUMM = tol+10
    for L in range(1,1000):
# !     WRITE(*,*)L,SUMM,tol 
        if (SUMM > tol):
            EQN = np.zeros((np11,np11+1),dtype=float) # should generate the same array?
            ITER = ITER+1    
# !    
# !        INTERIOR NODES
# !    
            """
            Interior Nodes
            """
#        I = 1
#        DO WHILE(I.LE.NSEC)
#          ARI = AR(Y(I))
#          ARIP1 = AR(Y(I+1))
#          K = 2*I
#          EQN(K,NP11)=-(ARI+ARIP1+DTX2*ALPHA*(V(I+1)*ARIP1-V(I)*ARI)+C1(I))
         
#          SF1 = ABS(V(I))*V(I)*CMN2/HR(Y(I))**1.333
#          SF2 = ABS(V(I+1))*V(I+1)*CMN2/HR(Y(I+1))**1.333
#          TERM1 = DTX2*ALPHA*(V(I+1)**2*ARIP1+G*CENTR(Y(I+1))-V(I)**2*ARI-G*CENTR(Y(I)))
#          TERM2 = -ALPHA*DT*G*((S0-SF2)*ARIP1+(S0-SF1)*ARI)
#          EQN(K+1,NP11) = -(V(I)*ARI+V(I+1)*ARIP1+TERM1+TERM2+C2(I))
         
#          DAY1 = TOP(Y(I))
#          DAY2 = TOP(Y(I+1))
#          EQN(K,K-1) = DAY1*(1-DTX2*ALPHA*V(I))
#          EQN(K,K) = -DTX2*ALPHA*ARI
#          EQN(K,K+1) = DAY2*(1+DTX2*ALPHA*V(I+1))
#          EQN(K,K+2) = DTX2*ALPHA*ARIP1
         
#          DCDY1 = DCENDY(Y(I))
#          DCDY2 = DCENDY(Y(I+1))
#          DSDV1 = 2*V(I)*CMN2/HR(Y(I))**1.333
#          DSDV2 = 2*V(I+1)*CMN2/HR(Y(I+1))**1.333
#          PERI = ARI/HR(Y(I))
#          TERM1 = 2*SQRT(1+S**2)*ARI - DAY1*PERI
#          TERM2 = HR(Y(I))**0.333*ARI**2
#          DSDY1 = 1.333*V(I)*ABS(V(I))*CMN2*TERM1/TERM2
#          PERIP1 = ARIP1/HR(Y(I+1))
#          TERM1 = 2*SQRT(1+S**2)*ARIP1-DAY2*PERIP1
#          TERM2 = HR(Y(I+1))**0.333*ARIP1**2
#          DSDY2 = 1.333*V(I+1)*ABS(V(I+1))*CMN2*TERM1/TERM2
#          TERM1 = -DTX2*ALPHA*(V(I)**2*DAY1 + G*DCDY1)
#          TERM2 = -G*DT*ALPHA*(S0-SF1)*DAY1
#          EQN(K+1,K-1)=V(I)*DAY1+TERM1+TERM2+G*DT*ALPHA*ARI*DSDY1
#          EQN(K+1,K)=ARI-DTX2*ALPHA*2*V(I)*ARI+G*DT*ALPHA*ARI*DSDV1
         
#          TERM1 = DTX2*ALPHA*(V(I+1)**2*DAY2+G*DCDY2)
#          TERM2 = -G*DT*G*(S0-SF2)*DAY2
#          EQN(K+1,K+1) = V(I+1)*DAY2+TERM1+TERM2+ALPHA*DT*G*ARIP1*DSDY2
#          EQN(K+1,K+2) = ARIP1+DTX2*ALPHA*2*V(I+1)*ARIP1+ALPHA*DT*G*DSDV2*ARIP1
#          I = I+1
#        END DO
            i = 0
            while (i <= nsec-1):
                ARi = AR(Y[i]).copy()
                ARiP1 = AR(Y[i+1]).copy()
                k = 2*i+1
                EQN[k,np11]=-(ARi+ARiP1+dtx2*alpha*(V[i+1]*ARiP1-V[i]*ARi)+C1[i]) # sets last column
            
                sf1 = abs(V[i])*V[i]*cmn_squared/HR(Y[i])**1.333
                sf2 = abs(V[i+1])*V[i+1]*cmn_squared/HR(Y[i+1])**1.333
                term1 = dtx2*alpha*(V[i+1]**2*ARiP1+grav*CENTR(Y[i+1])-V[i]**2*ARi-grav*CENTR(Y[i]))
                term2 = -alpha*dt*grav*((s0-sf2)*ARiP1+(s0-sf1)*ARi)
                EQN[k+1,np11] = -(V[i]*ARi+V[i+1]*ARiP1+term1+term2+C2[i])
            
                daY1 = TOP(Y[i])
                daY2 = TOP(Y[i+1])
                EQN[k,k-1] = daY1*(1-dtx2*alpha*V[i])
                EQN[k,k] = -dtx2*alpha*ARi
                EQN[k,k+1] = daY2*(1+dtx2*alpha*V[i+1])
                EQN[k,k+2] = dtx2*alpha*ARiP1
            
                dcdY1 = DCENDY(Y[i])
                dcdY2 = DCENDY(Y[i+1])
                dsdV1 = 2*V[i]*cmn_squared/HR(Y[i])**1.333
                dsdV2 = 2*V[i+1]*cmn_squared/HR(Y[i+1])**1.333
                PERi = ARi/HR(Y[i])
                term1 = 2*np.sqrt(1+s**2)*ARi - daY1*PERi
                term2 = HR(Y[i])**0.333*ARi**2
                dsdY1 = 1.333*V[i]*abs(V[i])*cmn_squared*term1/term2
                PERiP1 = ARiP1/HR(Y[i+1])
                term1 = 2*np.sqrt(1+s**2)*ARiP1-daY2*PERiP1
                term2 = HR(Y[i+1])**0.333*ARiP1**2
                dsdY2 = 1.333*V[i+1]*abs(V[i+1])*cmn_squared*term1/term2
                term1 = -dtx2*alpha*(V[i]**2*daY1 + grav*dcdY1)
                term2 = -grav*dt*alpha*(s0-sf1)*daY1
                EQN[k+1,k-1]=V[i]*daY1+term1+term2+grav*dt*alpha*ARi*dsdY1
                EQN[k+1,k]=ARi-dtx2*alpha*2*V[i]*ARi+grav*dt*alpha*ARi*dsdV1
            
                term1 = dtx2*alpha*(V[i+1]**2*daY2+grav*dcdY2)
                term2 = -grav*dt*grav*(s0-sf2)*daY2
                EQN[k+1,k+1] = V[i+1]*daY2+term1+term2+alpha*dt*grav*ARiP1*dsdY2
                EQN[k+1,k+2] = ARiP1+dtx2*alpha*2*V[i+1]*ARiP1+alpha*dt*grav*dsdV2*ARiP1
                i = i+1
                plt.plot(Y,".",label=i-2)
# !
# !         UPSTREAM END (Y given)
# !
#        EQN(1,1) = 1.0
#        EQN(1,NP11) = -(Y(1)-YRES)                             !ok
            """
            For now... UPSTREAM END (Y given)
            """
            EQN[0,0] = 1.0
            EQN[0,np11] = -(Y[0]-yres)
# !
# !         UPSTREAM END (V given)
# !
# !       EQN(1,1)    = 1.0
# !       EQN(2,NP11) = (V(1) - V0)                               ! ok
            """
            Will use!
            """
# !
# !         DOWNSTREAM END (NO OUTFLOW)
# !
#        EQN(2*NP1,2*NP1) = 1.
#        EQN(2*NP1,NP11)  = 0. - V(NP1)                         ! ok
            """
            For now... DOWNSTREAM END (NO OUTFLOW)
            """
            EQN[2*np1+1,2*np1+1] = 1.
            EQN[2*np1+1,np11] = 0. - V[np1]
# !       
# !         DOWNSTREAM END (Y given)
# !
# !       EQN(2*NP1,2*NP1) = 1.
# !       EQN(2*NP1,NP11) =  Y(NP1) - YD                          ! ok
            """
            Will use!
            """          
#        CALL MATSOL(2*NP1,DF,EQN)
#        I = 1
#        SUMM = 0.0
#        DO WHILE(I.LE.2*NP1)
#          SUMM = ABS(DF(I))+SUMM
#          IF(MOD(I,2)==1) Y(I/2+1) = Y(I/2+1)+DF(I)
#          IF(MOD(I,2)==0) V(I/2) = V(I/2) + DF(I)
#          I = I+1
#        END DO
# !    
# !        CHECK NUMBER OF ITERATIONS
# !    
#        IF(ITER.GT.MAXITER) THEN
#          IFLAG = 1
#          SUMM = TOL
#        END IF
#      ELSE
#        EXIT
#      END IF
#   END DO
#   IP = IP+1
# END DO
            # 
            DF = MATSOL(2*np1+2,EQN)
            i = 1
            SUMM = 0.0
            while (i <= 2*np1+2):
                SUMM = abs(DF[i-1])+SUMM
                if (i%2==1):
                    Y[int(i/2)] = Y[int(i/2)] + DF[i-1]
                if (i%2==0):
                    V[int(i/2)-1] = V[int(i/2)-1] + DF[i-1]
                i = i + 1
                """   
                CHECK NUMBER OF ITERATIONS
                """    
            print(ITER)
            if (ITER > maxiter):
                iflag = 1
                SUMM = tol.copy()
        else:
            break
    ip = ip+1

# IF(IFLAG.EQ.1) WRITE(6,"('MAXIMUM NUMBER OF ITERATIONS EXCEEDED')")
# STOP
# END

if (iflag == 1):
    print("Maximum number of iterations exceeded")

# FUNCTION AR(D)
# REAL AR,D
# REAL B0,S
# COMMON /CHANNEL/B0,S
# AR = (B0+D*S)*D
# END

# FUNCTION HR(D)
# REAL HR,D
# REAL B0,S
# COMMON /CHANNEL/B0,S
# HR = (B0+D*S)*D/(B0+2*D*SQRT(1+S*S))
# END

# FUNCTION TOP(D)
# REAL TOP,D
# REAL B0,S
# COMMON /CHANNEL/B0,S
# TOP = B0+2*D*S
# END

# FUNCTION CENTR(D)
# REAL CENTR,D
# REAL B0,S
# COMMON /CHANNEL/B0,S
# CENTR = D*D*(B0/2+D*S/3)
# END

# FUNCTION DCENDY(D)
# REAL DCENDY,D
# REAL B0,S
# COMMON /CHANNEL/B0,S
# DCENDY = D*(B0+D*S)
# END

# !    ******************************************************************
# !      SIMULTANEOUS SOLUTION OF THE SYSTEM OF EQUATIONS
# !
# SUBROUTINE MATSOL(N,X,A)

# IMPLICIT NONE
# INTEGER I,N,IP,J,NC,JJ
# INTEGER NROW
# REAL X,A
# REAL AMAX,COEF,SUMM
# DIMENSION X(60),A(124,125),NROW(60)

# I = 1
# DO WHILE(I.LE.N)
#   NROW(I) = I
#   I = I + 1
# END DO
# I = 1
# DO WHILE(I.LE.N-1)
#    AMAX = A(NROW(I),I)
#    J = I
#    IP =I
#    DO WHILE(J.LE.N)
#      IF(ABS(A(NROW(J),I)) .GT. AMAX) THEN
#        AMAX = ABS(A(NROW(J),I))
#        IP = J
#      END IF
#      J = J+1
#    END DO
#    IF(ABS(AMAX) .LE. 1E-08)THEN !GO TO 100
#      WRITE(*,*) 'SINGULAR MATRIX --> NO UNIQUE SOLUTION EXISTS'
#      STOP
#    END IF
#    IF(NROW(I).NE.NROW(IP))  THEN
#      NC = NROW(I)
#      NROW(I) = NROW(IP)
#      NROW(IP) = NC
#    END IF
#    J = I+1
#    DO WHILE(J.LE.N)
#      COEF = A(NROW(J),I)/A(NROW(I),I)
#      JJ = I+1
#      DO WHILE(JJ.LE.N+1)
#        A(NROW(J),JJ)=A(NROW(J),JJ)-COEF*A(NROW(I),JJ)
#        JJ = JJ + 1
#      END DO
#      J = J+1
#    END DO
#    I = I+1
# END DO
# IF(ABS(A(NROW(N),N)) .LE. 1E-08)THEN !GO TO 100
#   WRITE(*,*) 'SINGULAR MATRIX --> NO UNIQUE SOLUTION EXISTS'
#   STOP
# END IF
# X(N) = A(NROW(N),N+1)/A(NROW(N),N)
# I = N-1
# DO WHILE(I.GE.1)
#   SUMM = 0.0
#   J = I+1
#   DO WHILE(J.LE.N)
#     SUMM = A(NROW(I),J)*X(J) + SUMM
#     J = J+1
#   END DO
#   X(I) = (A(NROW(I),N+1) - SUMM)/A(NROW(I),I)
#   I = I-1
# END DO
# RETURN
# END

# !====================================================================
# !             NORMAL DEPTH
# !====================================================================
# SUBROUTINE NORMAL_D(YNORM,Q,CMAN,B0,S,S0)

#   IMPLICIT NONE
#   INTEGER I
#   REAL YNORM,Q,CMAN,B0,S,S0
#   REAL YNEW,C1,C2,ERR,DFDY,FY
#   REAL AR,HR,BW
  
#   IF(Q.LT.0.)THEN
#     YNORM = 0.
#     RETURN
#   END IF
#   C1 = (CMAN*Q)/SQRT(S0)
#   C2 = 2*SQRT(1 + S*S)
#   YNORM = (CMAN**2*(Q/B0)**2/S0)**0.3333
#   DO I = 1,1000
#      FY = AR(YNORM)*HR(YNORM)**0.6667 - C1
#      DFDY = 1.6667*BW(YNORM)*HR(YNORM)**0.6667 &
#           - 0.6667*HR(YNORM)**1.6667*C2
#      YNEW = YNORM - FY/DFDY
#      ERR = ABS((YNEW - YNORM)/YNEW)
#      YNORM = YNEW
#      IF(ERR.LT.1.0E-06)EXIT
#   END DO
#   RETURN     
# END SUBROUTINE NORMAL_D

# FUNCTION BW(D)
# REAL BW,D
# REAL B0,S
# COMMON /CHANNEL/B0,S
# BW = B0 + 2*S*D
# END
