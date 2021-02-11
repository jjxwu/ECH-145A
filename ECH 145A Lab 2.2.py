import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import scipy.stats as ss
import pandas as pd
import glob
import os
from uncertainties import ufloat as u

def func(t, P0, c, k, omega, phi):
    return P0 + c*np.exp(-k*t/2)*np.cos(omega*t+phi)

def correction(P_array, T_array):
    diff = abs(np.diff(P_array))
    index = np.where(diff[0:30]>0.025)[0]
    if len(index) == 0:
        index_val = 0
    else:
        index_val = index[0]
    corrected_P = P_array[index_val:]
    corrected_T = T_array[index_val:]
    return corrected_P ,corrected_T

def get_data_fit(P, T):
    emptyP = [[], [], []]
    emptyT = [[], [], []]
    emptyt = [[], [], []]
    fit = [[], [], []]
    fit_err = [[], [], []]
    for i in range(len(P)):
        for j in range(len(P[i])):
            P[i][j] = P[i][j][np.logical_not(np.isnan(P[i][j]))]
            start = np.argmax(P[i][j])
            end = start + 650
            Parray, Tarray = correction(P[i][j][start:end],T[i][j][start:end])
            emptyP[i].append(Parray)
            emptyT[i].append(Tarray-T[i][j][start])

            if emptyT[i][j][0] != 0:
                emptyT[i][j] -= emptyT[i][j][0]
            emptyt[i].append(np.linspace(emptyT[i][j][0], emptyT[i][j][-1], 1000))
            fit[i].append(so.curve_fit(func, emptyT[i][j], emptyP[i][j])[0])
            fit_err[i].append(np.sqrt(np.diag(so.curve_fit(func, emptyT[i][j], emptyP[i][j])[1])))
    return emptyP, emptyT, emptyt, fit, fit_err

#%%
data_cal = [glob.glob('Calibration_1_??.csv'),glob.glob('Calibration_2_??.csv'),glob.glob('Calibration_3_??.csv'),glob.glob('Calibration_4_??.csv')]
data_ar = [glob.glob('Osc_Ar_NoWeight_??.csv'),glob.glob('Osc_Ar_Mass1_??.csv'),glob.glob('Osc_Ar_Mass2_??.csv')]
data_co2 = [glob.glob('Osc_CO2_NoWeight_??.csv'),glob.glob('Osc_CO2_Mass1_??.csv'),glob.glob('Osc_CO2_Mass2_??.csv')]
data = data_ar + data_co2
#%%
weights = []
mV_V_cal = [[], [], [], []]
mV_err = [[], [], [], []]

weight_data = pd.read_csv('Weights.csv').values.tolist()

for i in range(3):
    weights.append(weight_data[i][1:])
    
weights = np.array(weight_data,dtype=float)
mean_weight_data = np.mean(weights,axis=1)/1000 #kg
weight_cfgs = np.cumsum(mean_weight_data) 

g = 9.8 #m/s**2
plunger_area = np.pi*(0.03256)**2/4  #m**2

pressure_data = ((weight_cfgs * g)/plunger_area)/1000 #kPa

for i in range(len(data_cal)):
    for j in data_cal[i]:
        array = pd.read_csv(j, header = 6).to_numpy()
        mV_V_cal[i].append(np.mean(array[:,2])/9)
        mV_err[i].append(ss.sem(array[:,2])/9)
a = ss.linregress(pressure_data, np.mean(mV_V_cal, axis=1))
#%%
plt.plot(pressure_data, np.mean(mV_V_cal,axis=1), label = 'Calibration (Air)', marker='o', ms = 5)
plt.plot(np.linspace(0.41194141, 3.05431013, 100), a.slope*np.linspace(0.41194141, 3.05431013, 100)+a.intercept, label = 'Curve Fit')
plt.plot([0.41346, 1.35604, 2.22659, 3.15154, 4.02878 ], [0.02159, 0.05542, 0.08702, 0.12101, 0.15300], label = 'Calibration (Ar, CO$_2$)', marker='o', ms = 5)
a2 = ss.linregress([0.41346, 1.35604, 2.22659, 3.15154, 4.02878 ], [0.02159, 0.05542, 0.08702, 0.12101, 0.15300])
plt.plot(np.linspace(0.41346, 4.02878, 100), a2.slope*np.linspace(0.41346, 4.02878, 100)+a2.intercept, label = 'Curve Fit')
plt.xlabel('Pressure (kPa)')
plt.ylabel('Voltage (mV/V)')
calibration = a.slope/0.145038 #mV/(psi V)
intercept = a.intercept/0.145038
plt.legend()
print('The calibration curve is of the form y = ', a.slope,'x + ', a.intercept, '(y = mV/V), x = kPa')
#plt.savefig('Lab2.2-Figure1.png', dpi = 300)

#%%
P_ar = [[], [], []] #psi
time_ar = [[], [], []] #s

P_co2 = [[], [], []] #psi
time_co2 = [[], [], []] #s

for i in range(len(data_ar)):
    for j in data_ar[i]:
        array = pd.read_csv(j, header=6).to_numpy() #mV/V
        P_ar[i].append(array[:,2]/9/calibration) #psi
        time_ar[i].append(array[:,1])
        
for i in range(len(data_co2)):
    for j in data_co2[i]:
        array = pd.read_csv(j, header=6).to_numpy() #mV/V
        P_co2[i].append(array[:,2]/9/calibration) #psi
        time_co2[i].append(array[:,1])
        
        
#%%

osc_P_ar, osc_T_ar, time_array_ar, fit_ar, fit_err_ar = get_data_fit(P_ar, time_ar)
osc_P_co2, osc_T_co2, time_array_co2, fit_co2, fit_err_co2 = get_data_fit(P_co2, time_co2)

#%%
fit_wErr = []
for i in range(len(fit_ar)):
    fit_wErr.append([])
    for j in range(len(fit_ar[i])):
        fit_wErr[i].append([])
        for k in range(len(fit_ar[i][j])):
            fit_wErr[i][j].append(u(fit_ar[i][j][k], fit_err_ar[i][j][k]))
for i in range(len(fit_co2)):
    fit_wErr.append([])
    for j in range(len(fit_co2[i])):
        fit_wErr[i+3].append([])
        for k in range(len(fit_co2[i][j])):
            fit_wErr[i+3][j].append(u(fit_co2[i][j][k], fit_err_co2[i][j][k]))
            
fit_wErr_avg = []

#P0, c, k, omega, phi
for i in range(len(fit_wErr)):
    fit_wErr_avg.append([])
    for j in range(len(fit_wErr[i])):
        param = [[], [], [], [], []]
        for k in range(5):
            param[k].append(fit_wErr[i][j][k])
    for l in range(5):
        fit_wErr_avg[i].append(np.mean(param[l]))
        
kappa_ar = []
kappa_co2 = []

h_ar = [u(0.068, 0.001), u(0.056, 0.008), u(0.056, 0.003)]
h_co2 = [u(0.068, 0.001), u(0.060, 0.003), u(0.057, 0.002)]
mass = [u(0.035, 0.0005), u(0.110695, 0.000001), u(0.184862, 0.000001)] #kg
A = np.pi*(u(0.03256, 0.00001)/2)**2 #m**2
V_ar = np.pi*(u(0.03256, 0.00001)/2)**2*np.array(h_ar) + np.pi*(u(0.00359, 0.00002)/2)**2*u(0.280, 0.001) #m**3
V_co2 = np.pi*(u(0.03256, 0.00001)/2)**2*np.array(h_co2) + np.pi*(u(0.00359, 0.00002)/2)**2*u(0.280, 0.001) #m**3

for i in range(len(fit_ar)):
    kappa_ar.append([])
    for j in range(len(fit_ar[i])):
        kappa_ar[i].append(((fit_wErr[i][j][3])**2+((fit_wErr[i][j][2])**2/4))*(mass[i]*V_ar[i])/(((fit_wErr[i][j][0]+14.6959)*6894.76)*A**2))
        
for i in range(len(fit_co2)):
    kappa_co2.append([])
    for j in range(len(fit_co2[i])):
        kappa_co2[i].append(((fit_wErr[i+3][j][3])**2+((fit_wErr[i+3][j][2])**2/4))*(mass[i]*V_co2[i])/(((fit_wErr[i+3][j][0]+14.6959)*6894.76)*A**2))

kappa_avg = []
for i in range(len(kappa_ar+kappa_co2)):
    if i <3:
        kappa_avg.append(np.mean(kappa_ar[i]))
    else:
        kappa_avg.append(np.mean(kappa_co2[i-3]))
#%%
for i in range(len(osc_P_ar)):
    for j in range(len(osc_P_ar[i])):
        tarray = np.linspace(osc_T_ar[i][j][0], osc_T_ar[i][j][-1], 1000)
        if i ==0:
            string = 'Mass1'
        if i ==1:
            string = 'Mass2'
        if i ==2:
            string = 'Mass3'
        plt.plot(osc_T_ar[i][j], osc_P_ar[i][j], 'o', label = f'Ar {string} Trial {str(j+1)}', ms = 3)
        plt.plot(tarray, func(tarray, *fit_ar[i][j]), label = 'Fitted Curve')
        plt.xlabel('Time (s)')
        plt.ylabel(' Pressure (psi$_g$)')
        plt.legend()
        plt.savefig(f'Figures/Argon-{string}-Trial-{str(j+1)}.jpg', dpi=300)
        plt.clf()
        
for i in range(len(osc_P_co2)):
    for j in range(len(osc_P_co2[i])):
        tarray = np.linspace(osc_T_co2[i][j][0], osc_T_co2[i][j][-1], 1000)
        if i ==0:
            string = 'Mass1'
        if i ==1:
            string = 'Mass2'
        if i ==2:
            string = 'Mass3'
        plt.plot(osc_T_co2[i][j], osc_P_co2[i][j], 'o', label = f'CO_2 {string} Trial {str(j+1)}', ms = 3)
        plt.plot(tarray, func(tarray, *fit_co2[i][j]), label = 'Fitted Curve')
        plt.xlabel('Time $(s)$')
        plt.ylabel('Pressure (psi$_g$)')
        plt.legend()
        plt.savefig(f'Figures/CO_2-{string}-Trial-{str(j+1)}.jpg', dpi=300)
        plt.clf()
#%%
for i in range(len(fit_wErr_avg)):
    if i-3>=0:
        gas = 'CO_2'
    else:
        gas = 'Argon'
    if i % 3 == 0:
        string = 'No Weight'
    elif i % 3 ==1:
        string = 'Mass 1'
    elif i % 3 ==2:
        string = 'Mass 2'
    print(gas, string + ':')
    print('kappa = ', kappa_avg[i])
    print('P_0 = ', fit_wErr_avg[i][0])
    print('C = ', fit_wErr_avg[i][1])
    print('k = ', fit_wErr_avg[i][2])
    print('omega = ', fit_wErr_avg[i][3])
    print('phi = ', fit_wErr_avg[i][4], '\n')
        
        
        
        
        
        
        
        
        
        
        
        
