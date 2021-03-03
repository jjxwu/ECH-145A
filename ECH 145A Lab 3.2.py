import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import pandas as pd
import os
from uncertainties import ufloat as u

ttb6 = pd.read_csv('TTB_Table_6.csv')
ttb1 = pd.read_csv('TTB_Table_1.csv')

ttb6 = ttb6.dropna(axis=1)
ttb1 = ttb1.fillna(0)
ttb6 = np.array(ttb6)
ttb1 = np.array(ttb1)
ttb1 = ttb1[:,1:]


#%%

def true_proof(T, rho):
    #Temperature in Fahrenheit
    #Density in kg/m^3
    #reference data
    ref_rho_e = 793.13 #kg/m**3 at 60 F
    ref_rho_w = 999.04 #kg/m**3 at 60 F
    
    mw_etOH = 0.04606844 #kg/mol
    mw_water = 0.01801528 #kg/mol
    alpha = 210e-6 #linear coefficient of thermal expansion for water (1/C)
    alpha_F = 5/9 * alpha #in 1/F
    
    rho_corrected = rho*(1+alpha_F*(T-60))
    
    sg_corrected = rho_corrected/ref_rho_w
    
    proof = ttb6[:,0]
    alcohol = ttb6[:,1]
    water = ttb6[:,2]
    sg = ttb6[:,4]
    
    sg_interp = si.interp1d(proof, sg, fill_value = 'extrapolate')
    proof_interp = si.interp1d(sg, proof, fill_value = 'extrapolate')
    alcohol_interp = si.interp1d(proof, alcohol, fill_value = 'extrapolate')
    
    temp = np.arange(1,101,1)
    proof_array = np.arange(0,207,1) 
    trueProof = si.interp2d(temp, proof_array, ttb1)
    
    proof_value = proof_interp(sg_corrected)
    
    tp = trueProof(T, proof_value)
    
    tsg =  sg_interp(tp) #unitless
    
    te_rho = tsg * ref_rho_w #kg/m**3
    
    alc_content = alcohol_interp(tp)
    
    eth_massfrac = alc_content*te_rho/(alc_content * te_rho + (100-alc_content)*ref_rho_w)
    water_massfrac = 1-eth_massfrac
    
    etOH_mol = alc_content * (te_rho/mw_etOH)
    water_mol = (100-alc_content) * (ref_rho_w/mw_water)
    
    etOH_mol_frac = etOH_mol / (etOH_mol + water_mol)
    water_mol_frac = 1 - etOH_mol_frac
    
    true_dens = tsg*ref_rho_w
    return tp[0], etOH_mol_frac[0], true_dens[0]

#%%
txy_data = np.array(pd.read_csv('ethanol_water_temp_xy_data_perrys_1963.csv'))

T = txy_data[:,0]
x_etOH = txy_data[:,1]
y_etOH = txy_data[:,2]

xy = si.interp1d(x_etOH,y_etOH,kind='cubic')

xT = si.interp1d(x_etOH,T,kind='cubic')
yT = si.interp1d(y_etOH,T,kind='cubic')

new_x = np.linspace(0,1,1000)
new_T_x = xT(new_x)
new_T_y = yT(new_x)

new_y = xy(new_x)
#%%
# pyplot.plot(x_etOH,y_etOH)
#R=2
plt.figure(0, figsize = (6,6))
plt.plot(new_x,new_y)
plt.plot(x_etOH,x_etOH,'k')
plt.plot(0.0331603,0.0331603, 'o', color = 'tab:red')
plt.plot(0.7681719,0.7681719, 'o', color = 'tab:green')
plt.plot(np.linspace(0.063377, 0.063377, 50), np.linspace(0.063377, 0.2983086, 50), color = 'tab:purple', label = 'q line')
plt.plot(new_x[63:768], 2/3*new_x[63:768]+0.2560573, color = 'tab:green', label = 'Rectifying Line')
plt.plot(new_x[32:64], 8.049343139*(new_x[32:64]-0.063377)+0.2983086, color = 'tab:red', label = 'Stripping Line')
plt.xlabel('X - Liquid Mole Fraction Ethanol')
plt.ylabel('Y - Vapor Mole Fraction Ethanol')
plt.axis('equal')
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,1.1,0.1))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.legend()
#plt.savefig('McCabe_Thiele_R2.png',dpi=300,bbox_inches='tight')
#%%
plt.figure(0, figsize = (6,6))
plt.plot(new_x,new_y)
plt.plot(x_etOH,x_etOH,'k')
plt.plot(0.029637,0.029637, 'o', color = 'tab:red')
plt.plot(0.777236,0.777236, 'o', color = 'tab:green')
plt.plot(np.linspace(0.063377, 0.063377, 50), np.linspace(0.063377, 0.20433598, 50), color = 'tab:purple', label = 'q line')
plt.plot(new_x[63:777], 4/5*new_x[63:777]+0.15363438, color = 'tab:green', label = 'Rectifying Line')
plt.plot(new_x[29:64], 5.177800237*(new_x[29:64]-0.029637)+0.029637, color = 'tab:red', label = 'Stripping Line')
plt.xlabel('X - Liquid Mole Fraction Ethanol')
plt.ylabel('Y - Vapor Mole Fraction Ethanol')
plt.axis('equal')
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,1.1,0.1))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.legend()
#plt.savefig('McCabe_Thiele_R4.png',dpi=300,bbox_inches='tight')
#%%
bottom = np.array(984.7, 990.6, 989.7, 990.4])
bottomT = [17.1, 15.7, 12.3, 11.9]

top = [u(830.4, 1), u(829.4, 1), u(821.6, 1), u(826, 1)]
topT = [u(13.9, 0.2), u(16.2, 0.2), u(20.7, 0.2), u(18.6, 0.2)]
#%%
molfrac = [[], []]
molfrac_error = [[],[]]
for i in range(4):
    molfrac[0].append(true_proof((bottomT[i]*9/5)+32, bottom[i].value)[1])
#    molfrac[1].append(true_proof((topT[i]*9/5)+32, top[i])[1])

#%%
# R = 2

R2_trays = [[0.8312, 0.8951, 0.9593, 0.976],[0.83, 0.9, 0.963, 0.9777]]
R2_temp = [[21, 18.8, 17, 15.1],[21.6, 17, 15.6, 14.2]]
R2_molfrac = [[], []]
for i in range(2):
    for j in range(4):
        R2_molfrac[i].append(true_proof((R2_temp[i][j]*9/5)+32, R2_trays[i][j]*1000)[1])
#%%
# r = 4

R4_trays = [[0.827, 0.8791, 0.957, 0.9727], [0.8275, 0.88, 0.9623, 0.9755]]
R4_temp = [[17.4, 14.4, 13.3, 10.3], [17, 14.6, 11, 9.2]]
R4_molfrac = [[], []]
for i in range(2):
    for j in range(4):
        R4_molfrac[i].append(round(true_proof((R4_temp[i][j]*9/5)+32, R4_trays[i][j]*1000)[1],5))