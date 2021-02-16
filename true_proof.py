import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import pandas as pd
import os

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
    
    return tp[0], etOH_mol_frac[0]