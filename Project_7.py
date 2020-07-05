# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:36:08 2020

@author: Saumya Dholakia
"""
import numpy as np
import matplotlib.pyplot as plt
from Riemann_mod import Riemann

R=Riemann()
rhoL,uL,pL,rhoR,uR,pR,max_time = R.get_cases()
case=4
gam=1.4
 
# Geometrical parameters:   
# When the interval length is given:
L=1
xth = L/2
dx = 0.025
x = np.arange(0,L+dx,dx)
nx = np.size(x)
aL = ((gam*pL[case])/rhoL[case])**(1/2)
aR = ((gam*pR[case])/rhoR[case])**(1/2)
umax = max(abs(uL[case])+aL, abs(uR[case])+aR)
time_end=max_time[case]

# Problem initialization:
def initial(rhoL,uL,pL,rhoR,uR,pR,nx,x,xth):
    u = np.zeros(nx)
    p = np.zeros(nx)
    rho = np.zeros(nx)
    for ii in range(nx):
        if x[ii] <= xth: 
            # LHS initial conditions for the Shock tube problem:
            rho[ii] = rhoL
            u[ii] = uL
            p[ii] = pL 
        else:
            # RHS initial conditions for the Shock tube problem:
            rho[ii] = rhoR
            u[ii] = uR
            p[ii] = pR
    return(rho,u,p)
rhoi,ui,pi = initial(rhoL[case],uL[case],pL[case],rhoR[case],uR[case],pR[case],nx,x,xth)

# Alternative way to implement the MUSCL algorithm
#Limiters
#class smooth():
#    def limit(self,r):
#        return((abs(r) + r)/(1.+abs(r)))
#class superbee():
#    def limit(self,r):
#        return(max(0,min(1,2.*r),min(r,2.)))

# MUSCL algorithm
#class MUSCL():
#    def _init_(self,limiter):
#        self.limiter = limiter
#        self.K = 1 
#        self.oneMK = 1. - self.K #(one minus K)
#        self.onePK = 1. + self.K #(one plus K)
#        
#    def get_LR_states(self,nx,var,varL,varR):
#        for ii in range(1,nx-2):
#            # Smoothness indicators:
#            riphM = (var[ii] - var[ii-1])/(var[ii+1] - var[ii] + 1e-30)
#            # r i plus half minus
#            rip32hM  = (var[ii+1] - var[ii])/(var[ii+2] - var[ii+1] + 1e-30)
#            # r i plus 3/2 minus
#            # Limiters:
#            psi_iphM = self.limiter.limit(riphM)
#            # psi i plus half minus
#            psi_imhP = psi_iphM/(riphM + 1e-30)
#            # psi i minus half plus
#            psi_ip32hM = self.limiter.limit(rip32hM)
#            # psi i plus 3/2 minus
#            psi_iphP = psi_ip32hM/(rip32hM + 1e-30)
#            # psi i plus half plus
#            # Differences:
#            delta_imhP = 0.5*(var[ii] - var[ii-1])*psi_imhP
#            # delta i minus half plus
#            delta_iphM = 0.5*(var[ii+1] - var[ii])*psi_iphM
#            # delta i plus half minus
#            delta_ip32hM = 0.5*(var[ii+2] - var[ii+1])*psi_ip32hM
#            # delta i plus 3/2 minus
#            delta_iphP = 0.5*(var[ii+1] - var[ii])*psi_iphP
#            # delta i plus half plus
#            deltaL = self.oneMK*delta_imhP + self.onePK*delta_iphM
#            deltaR = self.oneMK*delta_ip32hM + self.onePK*delta_iphP
#            # Calclulating the left and right state values:
#            varL[ii] = var[ii] + 0.5*deltaL
#            varR[ii] = var[ii+1] - 0.5*deltaR
#        # Boundary conditions:
#        varL[0] = var[0]
#        varR[0] = var[1]
#        varL[nx-1] = var[nx-1]
#        varR[nx-1] = var[nx]
#        varL[nx-2] = var[nx-2]
#        varR[nx-2] = var[nx-1]
        
#Limiters       
def smooth(r):
    r1 = (abs(r) + r)/(1.+abs(r))
    return(r1)
    
def superbee(r):
    r2 = max(0,min(1,2.*r),min(r,2.))
    return(r2)
    
# MUSCL algorithm
def get_LR_states(nx,var):
    K = -1
    varL = np.zeros(nx)
    varR = np.zeros(nx)
    for ii in range(1,nx-2):
        # Smoothness indicators:
        riphM = (var[ii] - var[ii-1])/(var[ii+1] - var[ii] + 1e-30)
        # r i plus half minus
        rip32hM  = (var[ii+1] - var[ii])/(var[ii+2] - var[ii+1] + 1e-30)
        # r i plus 3/2 minus
        # Limiters:
        psi_iphM = smooth(riphM)
#        psi_iphM = superbee(riphM)
        # psi i plus half minus
        psi_imhP = psi_iphM/(riphM + 1e-30)
        # psi i minus half plus
        psi_ip32hM = smooth(rip32hM)
#        psi_ip32hM = superbee(rip32hM)
        # psi i plus 3/2 minus
        psi_iphP = psi_ip32hM/(rip32hM + 1e-30)
        # psi i plus half plus
        # Differences:
        delta_imhP = 0.5*(var[ii] - var[ii-1])*psi_imhP
        # delta i minus half plus
        delta_iphM = 0.5*(var[ii+1] - var[ii])*psi_iphM
        # delta i plus half minus
        delta_ip32hM = 0.5*(var[ii+2] - var[ii+1])*psi_ip32hM
        # delta i plus 3/2 minus
        delta_iphP = 0.5*(var[ii+1] - var[ii])*psi_iphP
        # delta i plus half plus
        deltaL = (1-K)*delta_imhP + (1+K)*delta_iphM
        deltaR = (1-K)*delta_ip32hM + (1+K)*delta_iphP
        # Calclulating the left and right state values:
        varL[ii] = var[ii] + 0.5*deltaL
        varR[ii] = var[ii+1] - 0.5*deltaR
    # Boundary conditions:
    varL[0] = var[0]
    varR[0] = var[1]
    varL[nx-1] = var[nx-1]
    varR[nx-1] = var[nx-1]
    varL[nx-2] = var[nx-2]
    varR[nx-2] = var[nx-1]
    return(varL,varR) 
    
# Reconstructing the Q,F and a matrices using MUSCL:(Vanleer's method)
def cons_van_MUSCL(rhoi,ui,pi,nx,gam):
    F1_1 = np.zeros(nx)
    F2_1 = np.zeros(nx)
    F3_1 = np.zeros(nx)
    p = np.copy(pi)
    u = np.copy(ui)
    rho = np.copy(rhoi)
    a = np.sqrt(np.abs((gam*p)/rho))
    e = p/(rho*(gam-1))
    et = e + (u**2)/2
    Q = [rho,rho*u,rho*et]
    # Determine the +/- sign change for the respective variables
    # Using the MUSCL algorithm to calculate the left and right states
    rhop,rhom = get_LR_states(nx,rho)
    pp,pm = get_LR_states(nx,p)
    up,um = get_LR_states(nx,u)
    ap = np.sqrt(np.abs((gam*pp)/rhop))
    am = np.sqrt(np.abs((gam*pm)/rhom))
    Mp = (up/ap)
    Mm = (um/am)
    F1p = 0.25*rhop*ap*((1+Mp)**2)
    F1m = -0.25*rhom*am*((1-Mm)**2)
    F2p = F1p*((2*ap)/gam)*((0.5*(gam-1)*Mp)+1)
    F2m = F1m*((2*am)/gam)*((0.5*(gam-1)*Mm)-1)
    F3p = ((F2p**2)/F1p)*(0.5*((gam**2)/(gam**2 - 1)))
    F3m = ((F2m**2)/F1m)*(0.5*((gam**2)/(gam**2 - 1)))
    Fp = [F1p, F2p, F3p]
    Fm = [F1m, F2m, F3m]
    for ii in range(0,nx):
        F1_1[ii] = Fp[0][ii]+Fm[0][ii]
        F2_1[ii] = Fp[1][ii]+Fm[1][ii]
        F3_1[ii] = Fp[2][ii]+Fm[2][ii]
    F=[F1_1,F2_1,F3_1]
    return(Q,F,a)
    
# Reconstructing the Q,F and a matrices:(Vanleer's method)
def cons_van(rhoi,ui,pi,nx,sign,gam):
    p = np.copy(pi)
    u = np.copy(ui)
    rho = np.copy(rhoi)
    a = np.sqrt(np.abs(gam*p/rho))
    sign_change = ((-1)**sign)
    e = p/(rho*(gam-1))
    et = e + (u**2)/2
    M = (u/a)
    Q = [rho,rho*u,rho*et]
    F1 = sign_change*0.25*rho*a*((1+sign_change*M)**2)*1
    F2 = sign_change*0.25*rho*a*((1+sign_change*M)**2)*(2*a/gam)*(0.5*(gam-1)*M +sign_change*1)
    F3 = sign_change*0.25*rho*a*((1+sign_change*M)**2)*(2*a*a/((gam**2) -1))*((0.5*(gam-1)*M +sign_change*1)**2)
    F = [F1, F2, F3]
    return(Q,F,a)

# Reconstructing the Q,F and a matrices: (Roe's method)
def cons_roe(rhoi,ui,pi,max_time,nx):
    rho_avg = np.zeros(nx)
    u_avg = np.zeros(nx)
    H_avg = np.zeros(nx)
    p_avg = np.zeros(nx)
    deltap = np.zeros(nx)
    deltarho = np.zeros(nx)
    deltau = np.zeros(nx)
    F1_1 = np.zeros(nx)
    F2_1 = np.zeros(nx)
    F3_1 = np.zeros(nx)
    F1_2 = np.zeros(nx)
    F2_2 = np.zeros(nx)
    F3_2 = np.zeros(nx)
    p = np.copy(pi)
    u = np.copy(ui)
    rho = np.copy(rhoi)
    a = np.sqrt(np.abs((gam*p)/rho))
    e = p/(rho*(gam-1))
    et = e + (u**2)/2
    H = et + p/rho
    # Calculating the average quantities:
    for ii in range(0,nx-1):
        rho_avg[ii] = np.sqrt(rho[ii]*rho[ii+1])
        u_avg[ii] = (np.sqrt(rho[ii+1]/rho[ii])*u[ii+1] + u[ii])/(1+np.sqrt(rho[ii+1]/rho[ii]))
        H_avg[ii] = (np.sqrt(rho[ii+1]/rho[ii])*H[ii+1] + H[ii])/(1+np.sqrt(rho[ii+1]/rho[ii]))
        p_avg[ii] = (np.sqrt(rho[ii+1]/rho[ii])*p[ii+1] + p[ii])/(1+np.sqrt(rho[ii+1]/rho[ii]))
        deltap[ii] = p[ii+1]-p[ii]
        deltarho[ii] = rho[ii+1]-rho[ii]
        deltau[ii] = u[ii+1]-u[ii]
    a_avg = np.sqrt((gam -1)*(H_avg - (u_avg**2)/2))
#    a_avg = np.sqrt(np.abs((gam*p_avg)/rho_avg))
#    a_avg[nx-1] = a[nx-1]
    alpha = [(deltap-a_avg*rho_avg*deltau)/(2*a_avg*a_avg),(deltap+a_avg*rho_avg*deltau)/(2*a_avg*a_avg),deltarho-deltap/(a_avg*a_avg)]
    Q = [rho,rho*u,rho*et]
    F = [rho*u,rho*u*u + p, rho*u*H]
    # Averaging the left and right flux qunatities:
    for ii in range(0,nx-1):
        F1_1[ii] = F[0][ii]+F[0][ii+1]
        F2_1[ii] = F[1][ii]+F[1][ii+1]
        F3_1[ii] = F[2][ii]+F[2][ii+1]
    F_avg=[F1_1,F2_1,F3_1]
    # Entropy fix
    uma = u_avg - a_avg
    upa = u_avg + a_avg
    absuma = np.abs(uma)
    absupa = np.abs(upa)
    for ii in range(0,nx-1):
        eps = max(0,uma[ii]-(u[ii]-a[ii]),(u[ii+1]-a[ii+1])-uma[ii])
        if absuma[ii] < eps:
            absuma[ii] = eps
        eps = max(0,upa[ii]-(u[ii]+a[ii]),(u[ii+1]+a[ii+1])-upa[ii])
        if absupa[ii] < eps:
            absupa[ii] = eps
#    F1_2 = alpha[0]*np.abs(u_avg-a_avg) + alpha[1]*np.abs(u_avg+a_avg) + alpha[2]*np.abs(u_avg)
#    F2_2 = alpha[0]*(u_avg-a_avg)*np.abs(u_avg-a_avg) + alpha[1]*(u_avg+a_avg)*np.abs(u_avg+a_avg) + alpha[2]*u_avg*np.abs(u_avg)
#    F3_2 = alpha[0]*(H_avg-u_avg*a_avg)*np.abs(u_avg-a_avg) + alpha[1]*(H_avg+u_avg*a_avg)*np.abs(u_avg+a_avg) + 0.5*alpha[2]*u_avg*u_avg*np.abs(u_avg)
    F1_2 = alpha[0]*absuma + alpha[1]*absupa + alpha[2]*np.abs(u_avg)
    F2_2 = alpha[0]*(u_avg-a_avg)*absuma + alpha[1]*(u_avg+a_avg)*absupa + alpha[2]*u_avg*np.abs(u_avg)
    F3_2 = alpha[0]*(H_avg-u_avg*a_avg)*absuma + alpha[1]*(H_avg+u_avg*a_avg)*absupa + 0.5*alpha[2]*u_avg*u_avg*np.abs(u_avg)
    F_rho1 = 0.5*(F_avg[0]-F1_2)
    F_rho2 = 0.5*(F_avg[1]-F2_2)
    F_rho3 = 0.5*(F_avg[2]-F3_2)
    F_rho = [F_rho1,F_rho2,F_rho3]
    return(Q,F_rho,a)

# Reconstructing the Q,F and a matrices using MUSCL: (Roe's method)
def cons_roe_MUSCL(rhoi,ui,pi,max_time,nx):
    rho_avg = np.zeros(nx)
    u_avg = np.zeros(nx)
    H_avg = np.zeros(nx)
    deltap = np.zeros(nx)
    deltarho = np.zeros(nx)
    deltau = np.zeros(nx)
    F1_1 = np.zeros(nx)
    F2_1 = np.zeros(nx)
    F3_1 = np.zeros(nx)
    F1_2 = np.zeros(nx)
    F2_2 = np.zeros(nx)
    F3_2 = np.zeros(nx)
    p = np.copy(pi)
    u = np.copy(ui)
    rho = np.copy(rhoi)
    a = np.sqrt(np.abs(gam*p/rho))
    e = p/(rho*(gam-1))
    et = e + (u**2)/2
#    H = et + p/rho
    # Calculating the +/- states for the respective variables
    rhop,rhom = get_LR_states(nx,rho)
    pp,pm = get_LR_states(nx,p)
    up,um = get_LR_states(nx,u)
#    ep,em = get_LR_states(nx,e)
#    ap = np.sqrt(np.abs((gam*pp)/rhop))
#    am = np.sqrt(np.abs((gam*pm)/rhom))
    ep = pp/(rhop*(gam-1))
    em = pm/(rhom*(gam-1))
    etp = ep + (up**2)/2
    etm = em + (um**2)/2
    Hp = etp + pp/rhop
    Hm = etm + pm/rhom
    # Calculating the average quantities:
    rho_avg = np.sqrt(rhop*rhom)
    u_avg = (np.sqrt(rhom/rhop)*um + up)/(1+np.sqrt(rhom/rhop))
    p_avg = (np.sqrt(rhom/rhop)*pm + pp)/(1+np.sqrt(rhom/rhop))
    H_avg = (np.sqrt(rhom/rhop)*Hm + Hp)/(1+np.sqrt(rhom/rhop))
#    a_avg = (np.sqrt(rhom/rhop)*am + ap)/(1+np.sqrt(rhom/rhop))
    a_avg = np.sqrt(np.abs((gam*p_avg)/rho_avg))
#    a_avg = np.sqrt((gam -1)*(H_avg - (u_avg**2)/2))
    deltap = pm-pp
    deltarho = rhom-rhop
    deltau = um-up
    alpha = [(deltap-a_avg*rho_avg*deltau)/(2*a_avg*a_avg),(deltap+a_avg*rho_avg*deltau)/(2*a_avg*a_avg),deltarho-deltap/(a_avg*a_avg)]
    Q = [rho,rho*u,rho*et]
#    F = [rho*u,rho*u*u + p, rho*u*H]
    Fp = [rhop*up,rhop*up*up + pp, rhop*up*Hp]
    Fm = [rhom*um,rhom*um*um + pm, rhom*um*Hm]
    # Averaging the left and right flux qunatities:
#    F_avg = Fp + Fm
    for ii in range(0,nx):
        F1_1[ii] = Fp[0][ii]+Fm[0][ii]
        F2_1[ii] = Fp[1][ii]+Fm[1][ii]
        F3_1[ii] = Fp[2][ii]+Fm[2][ii]
    F_avg=[F1_1,F2_1,F3_1]
#    for ii in range(0,nx-1):
#        F1_1[ii] = F[0][ii]+F[0][ii+1]
#        F2_1[ii] = F[1][ii]+F[1][ii+1]
#        F3_1[ii] = F[2][ii]+F[2][ii+1]
#    F_avg=[F1_1,F2_1,F3_1]
    # Entropy fix
    uma = u_avg - a_avg
    upa = u_avg + a_avg
    absuma = np.abs(uma)
    absupa = np.abs(upa)
    for ii in range(0,nx-1):
        eps = max(0,uma[ii]-(u[ii]-a[ii]),(u[ii+1]-a[ii+1])-uma[ii])
        if absuma[ii] < eps:
            absuma[ii] = eps
        eps = max(0,upa[ii]-(u[ii]+a[ii]),(u[ii+1]+a[ii+1])-upa[ii])
        if absupa[ii] < eps:
            absupa[ii] = eps
    F1_2 = alpha[0]*absuma + alpha[1]*absupa + alpha[2]*np.abs(u_avg)
    F2_2 = alpha[0]*(u_avg-a_avg)*absuma + alpha[1]*(u_avg+a_avg)*absupa + alpha[2]*u_avg*np.abs(u_avg)
    F3_2 = alpha[0]*(H_avg-u_avg*a_avg)*absuma + alpha[1]*(H_avg+u_avg*a_avg)*absupa + 0.5*alpha[2]*u_avg*u_avg*np.abs(u_avg)
#    F1_2 = alpha[0]*np.abs(u_avg-a_avg) + alpha[1]*np.abs(u_avg+a_avg) + alpha[2]*np.abs(u_avg)
#    F2_2 = alpha[0]*(u_avg-a_avg)*np.abs(u_avg-a_avg) + alpha[1]*(u_avg+a_avg)*np.abs(u_avg+a_avg) + alpha[2]*u_avg*np.abs(u_avg)
#    F3_2 = alpha[0]*(H_avg-u_avg*a_avg)*np.abs(u_avg-a_avg) + alpha[1]*(H_avg+u_avg*a_avg)*np.abs(u_avg+a_avg) + 0.5*alpha[2]*u_avg*u_avg*np.abs(u_avg)
    F_roe1 = 0.5*(F_avg[0]-F1_2)
    F_roe2 = 0.5*(F_avg[1]-F2_2)
    F_roe3 = 0.5*(F_avg[2]-F3_2)
    F_roe = [F_roe1,F_roe2,F_roe3]
    return(Q,F_roe,a)
 
# Decoding the conservative variables: (Vanleer's method)
def decons_van(Q,nx,gam):
    e = np.zeros(nx)
    et = np.zeros(nx)
    p = np.zeros(nx)
    u = np.zeros(nx)
    rho = np.zeros(nx)
    rho = (Q[0])
    u = Q[1]/rho
    et = Q[2]/rho
    e = et - (u*u)/2
    p = rho*(gam - 1)*e
    return(rho,u,p,e)

# Decoding the conservative variables: (Roe's method)
def decons_roe(Q,nx,gam):
    e = np.zeros(nx)
    et = np.zeros(nx)
    p = np.zeros(nx)
    u = np.zeros(nx)
    rho = np.zeros(nx)
    rho = (Q[0])
    u = Q[1]/rho
    et = Q[2]/rho
    e = et - (u*u)/2
    p = rho*(gam - 1)*e
    return(rho,u,p,e) 
  
class Vanleer():
    def solve(self,time_end,nx,gam,cmax,umax,dx,Q):
        # Initializating parameters:
        rhonew,unew,pnew,enew = decons_van(Q,nx,gam)
        Q,F,a = cons_van(rhonew,unew,pnew,nx,2,gam) # Initial values
        Qi,Fi,a = cons_van(rhonew,unew,pnew,nx,2,gam) 
        # Initial values needed only for the boundary conditions
        Qnew,Fnew,a = cons_van(rhonew,unew,pnew,nx,2,gam)
        Q,Fp,a = cons_van(rhonew,unew,pnew,nx,2,gam)
        # These contain the positive flux values
        Q,Fn,a = cons_van(rhonew,unew,pnew,nx,1,gam)
        # These contain the negative flux values
        # Initializing time:
        dt = (cmax*dx)/umax
        it = 0
        t0 = 0
        # Initializing time loop:
        for t in range(10000):
            if t0 + dt > time_end:
                break
#            Q,Fp,a = cons_van(rhonew,unew,pnew,nx,2,gam)
#            Q,Fn,a = cons_van(rhonew,unew,pnew,nx,1,gam)
            for kk in range(3):
                for ii in range(1,nx-1): 
                    F[kk][ii] = (Fp[kk][ii]+Fn[kk][ii+1]) 
                F[kk][0] = (Fp[kk][0]+Fn[kk][1])
            for kk in range(3):
                for ii in range(1,nx-1):
                    Qnew[kk][ii] = (Q[kk][ii] - (dt/dx)*(F[kk][ii]-F[kk][ii-1]))
                Qnew[kk][nx-1] =  Qi[kk][nx-1]
                Qnew[kk][0] =  Qi[kk][0]
            rhonew,unew,pnew,enew = decons_van(Qnew,nx,gam)
            Q,Fp,a = cons_van(rhonew,unew,pnew,nx,2,gam)
            Q,Fn,a = cons_van(rhonew,unew,pnew,nx,1,gam)
            # Recompute dt dynamically to maintain stability
            dt = (dx*cmax)/(np.max(np.abs(unew) + a))
            it = it + 1
            print('for iteration =', it, 'for time step =',dt,end = '\n')
            # Update time:
            t0 = t0 + dt
        # Storing the final values at time level 'n+1'
        rhovan,uvan,pvan,evan = decons_van(Qnew,nx,gam)
        print('Van done')
        return(rhovan,uvan,pvan,evan) 
 
class Vanleer_MUSCL():
    def solve(self,time_end,nx,gam,cmax,umax,dx,Q):
        # Initializating parameters:
        rhonew,unew,pnew,enew = decons_van(Q,nx,gam)
        Q,F,a = cons_van_MUSCL(rhonew,unew,pnew,nx,gam) # Initial values
        Qi,Fi,a = cons_van_MUSCL(rhonew,unew,pnew,nx,gam) 
        # Initial values needed only for the boundary conditions
        Qnew,Fnew,a = cons_van_MUSCL(rhonew,unew,pnew,nx,gam)
        # Initializing time:
        dt = (cmax*dx)/umax
        it = 0
        t0 = 0
        # Initializing time loop:
        for t in range(10000):
            if t0 + dt > time_end:
                break
            for kk in range(3):
                for ii in range(1,nx-1):
                    Qnew[kk][ii] = (Q[kk][ii] - (dt/dx)*(F[kk][ii]-F[kk][ii-1]))
                Qnew[kk][nx-1] =  Qi[kk][nx-1]
                Qnew[kk][0] =  Qi[kk][0]
            rhonew,unew,pnew,enew = decons_van(Qnew,nx,gam)
            Q,F,a = cons_van_MUSCL(rhonew,unew,pnew,nx,gam)
            # Recompute dt dynamically to maintain stability
            dt = (dx*cmax)/(np.max(np.abs(unew) + a))
            it = it + 1
            print('for iteration =', it, 'for time step =',dt,end = '\n')
            # Update time:
            t0 = t0 + dt
        # Storing the final values at time level 'n+1'
        rhovan,uvan,pvan,evan = decons_van(Qnew,nx,gam)
        print('Van done')
        return(rhovan,uvan,pvan,evan) 
        
class Roe():
    def solve(self,time_end,nx,gam,cmax,umax,dx,Q):
        # Initializating parameters:
        rhonew,unew,pnew,enew = decons_roe(Q,nx,gam)
        Q,F_roe,a = cons_roe(rhonew,unew,pnew,max_time,nx) # Initial values
        Qi,Fi_roe,a = cons_roe(rhonew,unew,pnew,max_time,nx)
        # Initial values needed only for the boundary conditions
        Qnew,Fnew_roe,a = cons_roe(rhonew,unew,pnew,max_time,nx)
        # Initializing time:
        dt = (cmax*dx)/umax
        it = 0
        t0 = 0
        for t in range(10000):
            if t0 + dt > time_end:
                break
            for kk in range(3):
                for ii in range(1,nx-1): 
                    Qnew[kk][ii] = (Q[kk][ii] - (dt/dx)*(F_roe[kk][ii]-F_roe[kk][ii-1]))
                Qnew[kk][nx-1] =  Qi[kk][nx-1]
                Qnew[kk][0] =  Qi[kk][0]
            rhonew,unew,pnew,enew = decons_roe(Qnew,nx,gam)
            Q,F_roe,a = cons_roe(rhonew,unew,pnew,max_time,nx)
            # Recompute dt dynamically to maintain stability
            dt = (dx*cmax)/(np.max(np.abs(unew) + a))
            it = it + 1
            print('for iteration =', it, 'for time step =',dt,end = '\n')
            # Update time:
            t0 = t0 + dt
        # Storing the final values at time level 'n+1'
        rhoroe,uroe,proe,eroe = decons_roe(Qnew,nx,gam)
        print('Roe done')
        return(rhoroe,uroe,proe,eroe)
        
class Roe_MUSCL():
    def solve(self,time_end,nx,gam,cmax,umax,dx,Q):
        # Initializating parameters:
        rhonew,unew,pnew,enew = decons_roe(Q,nx,gam)
        Q,F_roe,a = cons_roe_MUSCL(rhonew,unew,pnew,max_time,nx) # Initial values
        Qi,Fi_roe,a = cons_roe_MUSCL(rhonew,unew,pnew,max_time,nx)
        # Initial values needed only for the boundary conditions
        Qnew,Fnew_roe,a = cons_roe_MUSCL(rhonew,unew,pnew,max_time,nx)
        # Initializing time:
        dt = (cmax*dx)/umax
        it = 0
        t0 = 0
        for t in range(10000):
            if t0 + dt > time_end:
                break
            for kk in range(3):
                for ii in range(1,nx-1): 
                    Qnew[kk][ii] = (Q[kk][ii] - (dt/dx)*(F_roe[kk][ii]-F_roe[kk][ii-1]))
                Qnew[kk][nx-1] =  Qi[kk][nx-1]
                Qnew[kk][0] =  Qi[kk][0]
            rhonew,unew,pnew,enew = decons_roe(Qnew,nx,gam)
            Q,F_roe,a = cons_roe_MUSCL(rhonew,unew,pnew,max_time,nx)
            # Recompute dt dynamically to maintain stability
            dt = (dx*cmax)/(np.max(np.abs(unew) + a))
            it = it + 1
            print('for iteration =', it, 'for time step =',dt,end = '\n')
            # Update time:
            t0 = t0 + dt
        # Storing the final values at time level 'n+1'
        rhoroe,uroe,proe,eroe = decons_roe(Qnew,nx,gam)
        print('Roe done')
        return(rhoroe,uroe,proe,eroe)
        
# Post processing:
#cmax = 1.
#Q,F,a = cons_van(rhoi,ui,pi,nx,2,gam)
#solver1 = Vanleer()
#rhovan,uvan,pvan,evan = solver1.solve(time_end,nx,gam,cmax,umax,dx,Q)
#R.plot_compare(x,xth,L,rhovan,pvan,uvan,evan,gam,time_end,case,"rusanov"+str(case),nx)

cmax = 0.25
Q,F,a = cons_van_MUSCL(rhoi,ui,pi,nx,gam)
solver2 = Vanleer_MUSCL()
rhovan_MUSCL,uvan_MUSCL,pvan_MUSCL,evan_MUSCL = solver2.solve(time_end,nx,gam,cmax,umax,dx,Q)
R.plot_compare(x,xth,L,rhovan_MUSCL,pvan_MUSCL,uvan_MUSCL,evan_MUSCL,gam,time_end,case,"rusanov"+str(case),nx)

#cmax = 1.
#Q,F_roe,a = cons_roe(rhoi,ui,pi,max_time,nx)
#solver3 = Roe()
#rhoroe,uroe,proe,eroe = solver3.solve(time_end,nx,gam,cmax,umax,dx,Q)
#R.plot_compare(x,xth,L,rhoroe,proe,uroe,eroe,gam,time_end,case,"rusanov"+str(case),nx)

cmax = 0.25
Q,F_roe,a = cons_roe_MUSCL(rhoi,ui,pi,max_time,nx)
solver4 = Roe_MUSCL()
rhoroe_MUSCL,uroe_MUSCL,proe_MUSCL,eroe_MUSCL = solver4.solve(time_end,nx,gam,cmax,umax,dx,Q)
R.plot_compare(x,xth,L,rhoroe_MUSCL,proe_MUSCL,uroe_MUSCL,eroe_MUSCL,gam,time_end,case,"rusanov"+str(case),nx)

rhoexact,uexact,pexact,eexact = R.plot(xth,gam,time_end,case,"rusanov"+str(case),nx)

# Plots considering MUSCL
f, ax = plt.subplots(2,2,figsize=(12,5))
ax[1][1].plot(x,evan_MUSCL,label='Van leer',color='red')
ax[1][1].plot(x,eroe_MUSCL,label='Roe',color='blue')
ax[1][1].plot(x,eexact,label='exact',color='black',linestyle='--')
ax[1][1].set_xlabel('$x(m)$',size=20)
ax[1][1].set_ylabel('$e(jKg-K)$',size=20)
ax[1][1].grid()
ax[1][1].legend(fontsize=14) 
 
ax[1][0].plot(x,pvan_MUSCL,label='Van leer',color='red')
ax[1][0].plot(x,proe_MUSCL,label='Roe',color='blue')
ax[1][0].plot(x,pexact,label='exact',color='black',linestyle='--')
ax[1][0].set_xlabel('$x(m)$',size=20)
ax[1][0].set_ylabel('$p(Pa)$',size=20)
ax[1][0].grid()
ax[1][0].legend(fontsize=14)

ax[0][0].plot(x,rhovan_MUSCL,label='Van leer',color='red')
ax[0][0].plot(x,rhoroe_MUSCL,label='Roe',color='blue')
ax[0][0].plot(x,rhoexact,label='exact',color='black',linestyle='--')
ax[0][0].set_ylabel(r'$\rho (kg/m^3)$',size=20)
ax[0][0].grid()
ax[0][0].legend(fontsize=14)

ax[0][1].plot(x,uvan_MUSCL,label='Van leer',color='red')
ax[0][1].plot(x,uroe_MUSCL,label='Roe',color='blue')
ax[0][1].plot(x,uexact,label='exact',color='black',linestyle='--')
ax[0][1].set_ylabel('$u(m/s)$',size=20)
ax[0][1].grid()
ax[0][1].legend(fontsize=14)

# Plots without considering MUSCL
#f, ax = plt.subplots(2,2,figsize=(12,5))
#ax[1][1].plot(x,evan,label='Van leer',color='red')
#ax[1][1].plot(x,eroe,label='Roe',color='blue')
#ax[1][1].plot(x,eexact,label='exact',color='black',linestyle='--')
#ax[1][1].set_xlabel('$x(m)$',size=20)
#ax[1][1].set_ylabel('$e(jKg-K)$',size=20)
#ax[1][1].grid()
#ax[1][1].legend(fontsize=14) 
# 
#ax[1][0].plot(x,pvan,label='Van leer',color='red')
#ax[1][0].plot(x,proe,label='Roe',color='blue')
#ax[1][0].plot(x,pexact,label='exact',color='black',linestyle='--')
#ax[1][0].set_xlabel('$x(m)$',size=20)
#ax[1][0].set_ylabel('$p(Pa)$',size=20)
#ax[1][0].grid()
#ax[1][0].legend(fontsize=14)
#
#ax[0][0].plot(x,rhovan,label='Van leer',color='red')
#ax[0][0].plot(x,rhoroe,label='Roe',color='blue')
#ax[0][0].plot(x,rhoexact,label='exact',color='black',linestyle='--')
#ax[0][0].set_ylabel(r'$\rho (kg/m^3)$',size=20)
#ax[0][0].grid()
#ax[0][0].legend(fontsize=14)
#
#ax[0][1].plot(x,uvan,label='Van leer',color='red')
#ax[0][1].plot(x,uroe,label='Roe',color='blue')
#ax[0][1].plot(x,uexact,label='exact',color='black',linestyle='--')
#ax[0][1].set_ylabel('$u(m/s)$',size=20)
#ax[0][1].grid()
#ax[0][1].legend(fontsize=14)

# Errors between numerical and exact values without MUSCL
#f, ax = plt.subplots(2,2,figsize=(12,5))
#ax[1][1].plot(x,evan - eexact,label='Van leer',color='red')
#ax[1][1].plot(x,eroe - eexact,label='Roe',color='blue')
#ax[1][1].set_xlabel('$x(m)$',size=20)
#ax[1][1].set_ylabel('$\epsilon = e-e_{exact}$',size=20)
#ax[1][1].grid()
#ax[1][1].legend(fontsize=14) 
# 
#ax[1][0].plot(x,pvan - pexact,label='Van leer',color='red')
#ax[1][0].plot(x,proe - pexact,label='Roe',color='blue')
#ax[1][0].set_xlabel('$x(m)$',size=20)
#ax[1][0].set_ylabel('$\epsilon = p-p_{exact}$',size=20)
#ax[1][0].grid()
#ax[1][0].legend(fontsize=14)
#
#ax[0][0].plot(x,rhovan - rhoexact,label='Van leer',color='red')
#ax[0][0].plot(x,rhoroe - rhoexact,label='Roe',color='blue')
#ax[0][0].set_ylabel('$\epsilon = rho-rho_{exact}$',size=20)
#ax[0][0].grid()
#ax[0][0].legend(fontsize=14)
#
#ax[0][1].plot(x,uvan - uexact,label='Van leer',color='red')
#ax[0][1].plot(x,uroe - uexact,label='Roe',color='blue')
#ax[0][1].set_ylabel('$\epsilon = u-u_{exact}$',size=20)
#ax[0][1].grid()
#ax[0][1].legend(fontsize=14)

# Errors between numerical and exact values with MUSCL
f, ax = plt.subplots(2,2,figsize=(12,5))
ax[1][1].plot(x,evan_MUSCL - eexact,label='Van leer',color='red')
ax[1][1].plot(x,eroe_MUSCL - eexact,label='Roe',color='blue')
ax[1][1].set_xlabel('$x(m)$',size=20)
ax[1][1].set_ylabel('$\epsilon = e-e_{exact}$',size=20)
ax[1][1].grid()
ax[1][1].legend(fontsize=14) 
 
ax[1][0].plot(x,pvan_MUSCL - pexact,label='Van leer',color='red')
ax[1][0].plot(x,proe_MUSCL - pexact,label='Roe',color='blue')
ax[1][0].set_xlabel('$x(m)$',size=20)
ax[1][0].set_ylabel('$\epsilon = p-p_{exact}$',size=20)
ax[1][0].grid()
ax[1][0].legend(fontsize=14)

ax[0][0].plot(x,rhovan_MUSCL - rhoexact,label='Van leer',color='red')
ax[0][0].plot(x,rhoroe_MUSCL - rhoexact,label='Roe',color='blue')
ax[0][0].set_ylabel('$\epsilon = rho-rho_{exact}$',size=20)
ax[0][0].grid()
ax[0][0].legend(fontsize=14)

ax[0][1].plot(x,uvan_MUSCL - uexact,label='Van leer',color='red')
ax[0][1].plot(x,uroe_MUSCL - uexact,label='Roe',color='blue')
ax[0][1].set_ylabel('$\epsilon = u-u_{exact}$',size=20)
ax[0][1].grid()
ax[0][1].legend(fontsize=14)

# For L2 plots
#def grid_gen():
#    ndx = 7
#    L=1
#    dx_start = 0.1
#    x_temp = []
#    x_new = []
#    deltax_temp = []
#    deltax = []
#    den = 0
#    nx_temp = []
#    nx_new = []
#    for ii in range(ndx):
#        den = 2.**ii
#        deltax_temp = dx_start/den
#        deltax.append(deltax_temp)
#        x_temp = np.arange(0,L+deltax_temp,deltax_temp)
#        nx_temp = np.size(x_temp)
#        nx_new.append(nx_temp)
#        x_new.append(x_temp)
##    print(deltax)
##    print(x)
##    print(nx_new)
#    return(x_new,deltax,nx_new)
#x_new,deltax,nx_new = grid_gen()
#
#def op_store_exact(case,max_time,gam,xth):
#    x_new,deltax,nx_new = grid_gen()
#    rhoexact_new = []
#    pexact_new  = []
#    uexact_new = []
#    eexact_new = []
#    for ii in range(len(deltax)):
#        rhoexact,uexact,pexact,eexact = R.plot(xth,gam,max_time[case],case,"rusanov"+str(case),nx_new[ii])
#        rhoexact_new.append(rhoexact)
#        pexact_new.append(pexact)
#        uexact_new.append(uexact)
#        eexact_new.append(eexact)
#    return(rhoexact_new,pexact_new,uexact_new,eexact_new)
#rhoexact_new,pexact_new,uexact_new,eexact_new = op_store_exact(case,max_time,gam,xth)
#
#def op_store_van(Vanleer,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth):
#    x_new,deltax,nx_new = grid_gen()
#    cmax = 1.
#    rhovan_new = []
#    pvan_new  = []
#    uvan_new = []
#    evan_new = []
#    for ii in range(len(deltax)):
#        rhoi,ui,pi = initial(rhoL[case],uL[case],pL[case],rhoR[case],uR[case],pR[case],nx_new[ii],x_new[ii],xth)
#        Q,F,a = cons_van(rhoi,ui,pi,nx_new[ii],2,gam)
#        solver = Vanleer()
#        rhovan,uvan,pvan,evan = solver.solve(time_end,nx_new[ii],gam,cmax,umax,deltax[ii],Q)
#        rhovan_new.append(rhovan)
#        pvan_new.append(pvan)
#        uvan_new.append(uvan)
#        evan_new.append(evan)
#    return(rhovan_new,pvan_new,uvan_new,evan_new)
#rhovan_new,pvan_new,uvan_new,evan_new = op_store_van(Vanleer,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth)
#
#def op_store_van_MUSCL(Vanleer_MUSCL,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth):
#    x_new,deltax,nx_new = grid_gen()
#    cmax = 0.25
#    rhovan_MUSCL = []
#    pvan_MUSCL  = []
#    uvan_MUSCL = []
#    evan_MUSCL = []
#    for ii in range(len(deltax)):
#        rhoi,ui,pi = initial(rhoL[case],uL[case],pL[case],rhoR[case],uR[case],pR[case],nx_new[ii],x_new[ii],xth)
#        Q,F,a = cons_van_MUSCL(rhoi,ui,pi,nx_new[ii],gam)
#        solver = Vanleer_MUSCL()
#        rhovan,uvan,pvan,evan = solver.solve(time_end,nx_new[ii],gam,cmax,umax,deltax[ii],Q)
#        rhovan_MUSCL.append(rhovan)
#        pvan_MUSCL.append(pvan)
#        uvan_MUSCL.append(uvan)
#        evan_MUSCL.append(evan)
#    return(rhovan_MUSCL,pvan_MUSCL,uvan_MUSCL,evan_MUSCL)
#rhovan_MUSCL,pvan_MUSCL,uvan_MUSCL,evan_MUSCL = op_store_van_MUSCL(Vanleer_MUSCL,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth)
#
#def op_store_roe(Roe,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth):
#    x_new,deltax,nx_new = grid_gen()
#    cmax = 1.
#    rhoroe_new = []
#    proe_new  = []
#    uroe_new = []
#    eroe_new = []
#    for ii in range(len(deltax)):
#        rhoi,ui,pi = initial(rhoL[case],uL[case],pL[case],rhoR[case],uR[case],pR[case],nx_new[ii],x_new[ii],xth)
#        Q,F_roe,a = cons_roe(rhoi,ui,pi,max_time,nx_new[ii])
#        solver = Roe()
#        rhoroe,uroe,proe,eroe = solver.solve(time_end,nx_new[ii],gam,cmax,umax,deltax[ii],Q)
#        rhoroe_new.append(rhoroe)
#        proe_new.append(proe)
#        uroe_new.append(uroe)
#        eroe_new.append(eroe)
#    return(rhoroe_new,proe_new,uroe_new,eroe_new)
#rhoroe_new,proe_new,uroe_new,eroe_new = op_store_roe(Roe,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth)
#
#def op_store_roe_MUSCL(Roe_MUSCL,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth):
#    x_new,deltax,nx_new = grid_gen()
#    cmax = 0.25
#    rhoroe_MUSCL = []
#    proe_MUSCL = []
#    uroe_MUSCL = []
#    eroe_MUSCL = []
#    for ii in range(len(deltax)):
#        rhoi,ui,pi = initial(rhoL[case],uL[case],pL[case],rhoR[case],uR[case],pR[case],nx_new[ii],x_new[ii],xth)
#        Q,F_roe,a = cons_roe_MUSCL(rhoi,ui,pi,max_time,nx_new[ii])
#        solver = Roe_MUSCL()
#        rhoroe,uroe,proe,eroe = solver.solve(time_end,nx_new[ii],gam,cmax,umax,deltax[ii],Q)
#        rhoroe_MUSCL.append(rhoroe)
#        proe_MUSCL.append(proe)
#        uroe_MUSCL.append(uroe)
#        eroe_MUSCL.append(eroe)
#    return(rhoroe_MUSCL,proe_MUSCL,uroe_MUSCL,eroe_MUSCL)
#rhoroe_MUSCL,proe_MUSCL,uroe_MUSCL,eroe_MUSCL = op_store_roe_MUSCL(Roe_MUSCL,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth)
#
#def L2_norm(var,varexact):
#    x_new,deltax,nx_new = grid_gen()
#    L2_temp = 0
#    L2_err = []
#    for kk in range(len(deltax)):
#        L2_temp = ((np.sum(np.power((var[kk]-varexact[kk]),2)))**1/2)/nx_new[kk]
##    for ii in range(nx_new[kk]):
##        L2_temp = L2_temp + (((var[ii][kk]-varexact[ii][kk])**2)**0.5)/nx_new[ii][kk]
#        L2_err.append(L2_temp)
#    return(L2_err)
#    
#def plot_L2(Vanleer,Roe,Vanleer_MUSCL,Roe_MUSCL,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth,case):
#    x_new,deltax,nx_new = grid_gen()
#    L2_rhovan = []
#    L2_pvan = []
#    L2_uvan = []
#    L2_evan = []
#    L2_rhovan_MUSCL = []
#    L2_pvan_MUSCL = []
#    L2_uvan_MUSCL = []
#    L2_evan_MUSCL = []
#    L2_rhoroe = []
#    L2_proe = []
#    L2_uroe = []
#    L2_eroe = []
#    L2_rhoroe_MUSCL = []
#    L2_proe_MUSCL = []
#    L2_uroe_MUSCL = []
#    L2_eroe_MUSCL = []
#    dx2_temp = []
#    one_dx_temp = []
#    dx2 = []
#    one_dx = []
#    for ii in range(len(deltax)):
#        rhoexact_new,pexact_new,uexact_new,eexact_new = op_store_exact(case,max_time,gam,xth)
#        rhovan_new,pvan_new,uvan_new,evan_new = op_store_van(Vanleer,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth)
#        rhoroe_new,proe_new,uroe_new,eroe_new = op_store_roe(Roe,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth)
#        rhovan_MUSCL,pvan_MUSCL,uvan_MUSCL,evan_MUSCL = op_store_van_MUSCL(Vanleer_MUSCL,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth)
#        rhoroe_MUSCL,proe_MUSCL,uroe_MUSCL,eroe_MUSCL = op_store_roe_MUSCL(Roe_MUSCL,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth)
#        one_dx_temp = 1/deltax[ii]
#        one_dx.append(one_dx_temp)
#        dx2_temp = deltax[ii]**2
#        dx2.append(dx2_temp)
#        
#    # Calculation of L2 errors
#    # Van leer
#    L2_rhovan = L2_norm(rhovan_new,rhoexact_new)
#    L2_uvan = L2_norm(uvan_new,uexact_new)
#    L2_pvan = L2_norm(pvan_new,pexact_new)
#    L2_evan = L2_norm(evan_new,eexact_new)
#    # Van leer with MUSCL
#    L2_rhovan_MUSCL = L2_norm(rhovan_MUSCL,rhoexact_new)
#    L2_uvan_MUSCL = L2_norm(uvan_MUSCL,uexact_new)
#    L2_pvan_MUSCL = L2_norm(pvan_MUSCL,pexact_new)
#    L2_evan_MUSCL = L2_norm(evan_MUSCL,eexact_new)
#    # Roe 
#    L2_rhoroe = L2_norm(rhoroe_new,rhoexact_new)
#    L2_uroe = L2_norm(uroe_new,uexact_new)
#    L2_proe = L2_norm(proe_new,pexact_new)
#    L2_eroe = L2_norm(eroe_new,eexact_new)
#    # Roe with MUSCL
#    L2_rhoroe_MUSCL = L2_norm(rhoroe_MUSCL,rhoexact_new)
#    L2_uroe_MUSCL = L2_norm(uroe_MUSCL,uexact_new)
#    L2_proe_MUSCL = L2_norm(proe_MUSCL,pexact_new)
#    L2_eroe_MUSCL = L2_norm(eroe_MUSCL,eexact_new)
#    
#    # Log log Plot:
#    # Methods without using MUSCL
#    f, ax = plt.subplots(2,2,figsize=(12,5))
#    ax[1][1].loglog(one_dx,deltax,label='first order',color='black')
#    ax[1][1].loglog(one_dx,dx2,label='second order',color='black', linestyle='--')
#    ax[1][1].loglog(one_dx,L2_evan,label='Van leer',color='red')
#    ax[1][1].loglog(one_dx,L2_eroe,label='Roe',color='blue')
#    ax[1][1].set_xlabel('$x(m)$',size=20)
#    ax[1][1].set_ylabel('$e(jKg-K)$',size=20)
#    ax[1][1].grid()
#    ax[1][1].legend(fontsize=14)  
#    
#    ax[1][0].loglog(one_dx,deltax,label='first order',color='black')
#    ax[1][0].loglog(one_dx,dx2,label='second order',color='black', linestyle='--')
#    ax[1][0].loglog(one_dx,L2_pvan,label='Van leer',color='red')
#    ax[1][0].loglog(one_dx,L2_proe,label='Roe',color='blue')
#    ax[1][0].set_xlabel('$x(m)$',size=20)
#    ax[1][0].set_ylabel('$p(Pa)$',size=20)
#    ax[1][0].grid()
#    ax[1][0].legend(fontsize=14) 
#    
#    ax[0][0].loglog(one_dx,deltax,label='first order',color='black')
#    ax[0][0].loglog(one_dx,dx2,label='second order',color='black', linestyle='--')
#    ax[0][0].loglog(one_dx,L2_rhovan,label='Van leer',color='red')
#    ax[0][0].loglog(one_dx,L2_rhoroe,label='Roe',color='blue')
#    ax[0][0].set_xlabel('$x(m)$',size=20)
#    ax[0][0].set_ylabel(r'$\rho (kg/m^3)$',size=20)
#    ax[0][0].grid()
#    ax[0][0].legend(fontsize=14) 
#    
#    ax[0][1].loglog(one_dx,deltax,label='first order',color='black')
#    ax[0][1].loglog(one_dx,dx2,label='second order',color='black', linestyle='--')
#    ax[0][1].loglog(one_dx,L2_uvan,label='Van leer',color='red')
#    ax[0][1].loglog(one_dx,L2_uroe,label='Roe',color='blue')
#    ax[0][1].set_xlabel('$x(m)$',size=20)
#    ax[0][1].set_ylabel('$u(m/s)$',size=20)
#    ax[0][1].grid()
#    ax[0][1].legend(fontsize=14)  
#    
#    # Methods using MUSCL   
#    f, ax = plt.subplots(2,2,figsize=(12,5))
#    ax[1][1].loglog(one_dx,deltax,label='first order',color='black')
#    ax[1][1].loglog(one_dx,dx2,label='second order',color='black', linestyle='--')
#    ax[1][1].loglog(one_dx,L2_evan_MUSCL,label='Van leer',color='red')
#    ax[1][1].loglog(one_dx,L2_eroe_MUSCL,label='Roe',color='blue')
#    ax[1][1].set_xlabel('$x(m)$',size=20)
#    ax[1][1].set_ylabel('$e(jKg-K)$',size=20)
#    ax[1][1].grid()
#    ax[1][1].legend(fontsize=14)  
#    
#    ax[1][0].loglog(one_dx,deltax,label='first order',color='black')
#    ax[1][0].loglog(one_dx,dx2,label='second order',color='black', linestyle='--')
#    ax[1][0].loglog(one_dx,L2_pvan_MUSCL,label='Van leer',color='red')
#    ax[1][0].loglog(one_dx,L2_proe_MUSCL,label='Roe',color='blue')
#    ax[1][0].set_xlabel('$x(m)$',size=20)
#    ax[1][0].set_ylabel('$p(Pa)$',size=20)
#    ax[1][0].grid()
#    ax[1][0].legend(fontsize=14) 
#    
#    ax[0][0].loglog(one_dx,deltax,label='first order',color='black')
#    ax[0][0].loglog(one_dx,dx2,label='second order',color='black', linestyle='--')
#    ax[0][0].loglog(one_dx,L2_rhovan_MUSCL,label='Van leer',color='red')
#    ax[0][0].loglog(one_dx,L2_rhoroe_MUSCL,label='Roe',color='blue')
#    ax[0][0].set_xlabel('$x(m)$',size=20)
#    ax[0][0].set_ylabel(r'$\rho (kg/m^3)$',size=20)
#    ax[0][0].grid()
#    ax[0][0].legend(fontsize=14) 
#    
#    ax[0][1].loglog(one_dx,deltax,label='first order',color='black')
#    ax[0][1].loglog(one_dx,dx2,label='second order',color='black', linestyle='--')
#    ax[0][1].loglog(one_dx,L2_uvan_MUSCL,label='Van leer',color='red')
#    ax[0][1].loglog(one_dx,L2_uroe_MUSCL,label='Roe',color='blue')
#    ax[0][1].set_xlabel('$x(m)$',size=20)
#    ax[0][1].set_ylabel('$u(m/s)$',size=20)
#    ax[0][1].grid()
#    ax[0][1].legend(fontsize=14)  
#
#plot_L2(Vanleer,Roe,Vanleer_MUSCL,Roe_MUSCL,rhoL,rhoR,uL,uR,pL,pR,gam,time_end,umax,xth,case)

