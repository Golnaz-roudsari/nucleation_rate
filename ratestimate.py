#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:05:52 2021

@author: Roudsari

Nucleation rate using Cox method
https://doi.org/10.1063/1.4919714

"""
import numpy as np
import matplotlib.pyplot as plt
import io
plt.rcParams["font.family"] = "Arial"
from matplotlib.pyplot import figure






#%%
def compressed_decay_fitting(t,P):    
    P_=np.log(-np.log(P))
    t_=np.log(t)
    p=np.polyfit(t_,P_,1)
    S=np.exp(p[1])
    R=S**(1/p[0])
    g=p[0]
    return g,R
 
def rateestimate(t_sim,N_nuc,eps,N_sim):
    if N_nuc==N_sim:
        P_liq= np.ones([1,N_sim])- np.linspace(1/N_sim, 1, num=N_sim)
        P_liq[0,-1]=eps
        list1= ((1-eps)*np.ones([1, 2])) 
        t_sim=np.concatenate((np.array([[1,t_sim[0,0]/1.1]]), t_sim),axis=1)
        P_liq= np.concatenate((list1,P_liq),axis=1)
    elif N_nuc < N_sim:
         step= np.ones([1,N_sim])- np.linspace(1/N_sim, 1, num=N_sim)
         P_liq= step[0,:N_nuc].reshape((1,N_nuc))
         list1= ((1-eps)*np.ones([1, 2])) 
         t_sim=np.concatenate((np.array([[1,t_sim[0,0]/1.1]]), t_sim),axis=1)
         P_liq=np.concatenate((list1,P_liq),axis=1)
    #plt.plot(np.ravel(t_sim), np.ravel(P_liq), 'b*')
    [g,R]=compressed_decay_fitting(np.ravel(t_sim),np.ravel(P_liq))
    P_liq = np.append(P_liq, np.repeat(np.nan, N_sim-N_nuc))
    t_sim = np.append(t_sim, np.repeat(np.nan, N_sim-N_nuc))
 #  if g<=1:
  #      g= float("NAN")
   #     R = 1/(np.mean(t_sim))       
    return g,R,P_liq,t_sim   

def rateestimate_bs(Nss,x,N_nuc,eps,Nx):
    R=np.zeros([1,Nss])
    G=np.zeros([1,Nss])
    for i in range(Nss):
        Nx=x.shape[1] 
        inds = np.random.randint(Nx, size=Nx)
        y = np.sort(x[0,inds]).reshape(1,-1)
        [g,r,TT,YY] = rateestimate(y,N_nuc,eps,N_sim)
        R[0,i]=r
        G[0,i]=g
    r_bs= np.mean(R)
    g_bs= np.mean(G)
    return g_bs,r_bs
        

#%%
N_sim=15
eps=1e-4
Res_gr = np.zeros([8,3])
Res_gr_s = np.zeros([8,1])
Res_P = np.zeros([8,N_sim+2])
Res_t = np.zeros([8,N_sim+2])
count=0
with io.open("/home/local/roudasri/Python_codes/nuc_rate/rate_wedge.dat","r") as T:
    for line in T:
        ang=line.split(' ')[0]
        t_str=line.split(' ')[1:]
        t_float= np.array([[float(i) for i in t_str]])
        t_float=np.sort(t_float)
        N_nuc=np.size(t_float)
        ang=float(ang)
        [g,r,P,t] = rateestimate(t_float,N_nuc,eps,N_sim)
        Res_gr[count] = np.array([ang, g, r]).reshape((1,3))
        Res_gr_s[count]= np.array([r/(10.31*1e-18)]).reshape((1,1))
        Res_P[count] = P.reshape(1,2+N_sim)
        Res_t[count] = t.reshape(1,2+N_sim)
        count+=1


#%%
Clr1=['b','g','r','c','m','y', 'k', 'C0'] 
Clr2=['b','g','r','c','m--','y--', 'k--', 'C0'] 
figure(figsize=(4, 3), dpi=300)
#LS= ['solid','solid','solid','solid','solid','dotted', 'dotted', 'dotted']
for i in range(8):
    t=Res_t[i,2:]
    p=Res_P[i,2:] 
    x = np.arange(1,np.nanmax(t),0.1)
    r=Res_gr[i,2]
    g=Res_gr[i,1]
    ang=Res_gr[i,0]
    y = np.exp(-(np.power((r*x),g)))
    plt.plot(np.ravel(t), np.ravel(p), Clr1[i]+'*',label=str(int(ang))+'$^\circ$',markersize=4)
    plt.plot(np.ravel(x), np.ravel(y), Clr2[i],label=str(int(ang))+'$^\circ$-fitting',markersize=4)
    
plt.title("")
plt.xlabel("Time (ns)")
plt.ylabel("$P_{liq}(t)$")
#plt.set_xlim([0, 150])
plt.grid()
#lgd=plt.legend(bbox_to_anchor=(1.35, 1.04), loc='upper right', prop={'size': 7})
lgd=plt.legend(loc='lower center',ncol=4, shadow= False, bbox_to_anchor=(0.5,-0.52),borderpad=0.2, prop={'size':8})    

plt.xlim(0, 150)
plt.ylim(0, 1)
plt.savefig('nuc_rate.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()     



#%%
defect= np.array([69.27, 106.36, 175.53, 72.34, 76.96, 87.00, 90.52]).reshape(1,7)
area= np.array([194.24, 219.16, 277.07, 102.91, 109.58, 92.10, 92.10]).reshape(1,7)
r_def=np.zeros([2,8])
for i in range(7):
    r_def[0,i]= Res_gr[i,2]/(defect[0,i]*1e-27)
    r_def[1,i]= Res_gr[i,2]/(area[0,i]*1e-27)
r_sa= print(r_def)

    

#%%

#bootstraping
Nss=10000
N_sim=15
eps=1e-4
Res_gr_bs = np.zeros([7,3])
count=0
with io.open("/home/local/roudasri/Python_codes/nuc_rate/rate_wedge.dat","r") as T:
    for line in T:
        ang=line.split(' ')[0]
        t_str=line.split(' ')[1:]
        t_float= np.array([[float(i) for i in t_str]])
        t_float=np.sort(t_float)
        N_nuc=np.size(t_float)
        ang=float(ang)
        [g_bs,r_bs] = rateestimate_bs(Nss,t_float,N_nuc,eps,N_sim)
        Res_gr_bs[count] = np.array([ang, g_bs, r_bs]).reshape((1,3))
        count+=1


    
   




