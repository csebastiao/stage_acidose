# -*- coding: utf-8 -*-
"""
Ce code lit et permet d'observer les résultats obtenus
"""
# Modules
from matplotlib import pyplot as plt
import numpy as np

# Functions
"Calcule la moyenne par split"
def mean_per_split(val):
    """
    :param val: array, ensemble des données
    :return mean: array, ensemble des moyennes par split
    """
    mean=[[0 for y in range(4)] for x in range(5)]
    for i in range(5):
        for j in range(10):
            for k in range(4):
                mean[i][k]+=val[i*10+j][k]/10
        
    return mean

"Calcule la moyenne globale"
def mean_all(val):
    """
    :param val: array, ensemble des données
    :return mean: array, moyenne globale
    """
    mean=[0 for y in range(4)]
    for i in range(50):
        for j in range(4):
            mean[j]+=val[i][j]/50
    return mean


# Main
if __name__ == "__main__":
    "On récupère le nom des fichiers"
    names=['./results_S1.txt',
           './results_S2.txt',
           './results_all.txt']
    
    "On forme les arrays que nous allons remplir des informations"
    ct_S1=[[None for y in range(4)] for x in range(50)]
    ca_S1=[[None for y in range(4)] for x in range(50)]
    ct_S2=[[None for y in range(4)] for x in range(50)]
    ca_S2=[[None for y in range(4)] for x in range(50)]
    ct_all=[[None for y in range(4)] for x in range(50)]
    ca_all=[[None for y in range(4)] for x in range(50)]
    mt_S1=[None for x in range(50)]
    ma_S1=[None for x in range(50)]
    mt_S2=[None for x in range(50)]
    ma_S2=[None for x in range(50)]
    mt_all=[None for x in range(50)]
    ma_all=[None for x in range(50)]
    cmat=[ct_S1,ca_S1,ct_S2,ca_S2,ct_all,ca_all]
    MCC=[mt_S1,ma_S1,mt_S2,ma_S2,mt_all,ma_all]
    
    "On récupère les informations dans les fichiers"
    for i in range(3):
        file=open(names[i])
        content=file.readlines()
        content=content[0].split()
        for j in range(5):
            for k in range(10):
                for l in range(4):
                    cmat[i*2][j*10+k][l]=int(content[j*100+k*10+l])
                    cmat[i*2+1][j*10+k][l]=int(content[j*100+k*10+4+l])
                MCC[i*2][j*10+k]=float(content[j*100+k*10+8])
                MCC[i*2+1][j*10+k]=float(content[j*100+k*10+9])
      
    "On calcule l'ensemble des moyennes, par split et globaux"         
    mean_ts_S1=mean_per_split(ct_S1)
    mean_ta_S1=mean_all(ct_S1)
    mean_as_S1=mean_per_split(ca_S1)
    mean_aa_S1=mean_all(ca_S1)
    mean_ts_S2=mean_per_split(ct_S2)
    mean_ta_S2=mean_all(ct_S2)
    mean_as_S2=mean_per_split(ca_S2)
    mean_aa_S2=mean_all(ca_S2)
    mean_ts_all=mean_per_split(ct_all)
    mean_ta_all=mean_all(ct_all)
    mean_as_all=mean_per_split(ca_all)
    mean_aa_all=mean_all(ca_all)
    
    "On visualise les MCC"
    plt.figure()
    plt.scatter(np.arange(0,50),ma_S1,label='Première phase',s=100)
    plt.scatter(np.arange(0,50),ma_S2,label='Deuxième phase',s=100)
    plt.scatter(np.arange(0,50),ma_all,label='Phases mélangées',s=100)
    plt.plot(np.arange(0,51),np.zeros(51),color='k')
    plt.ylabel('MCC')
    plt.xlabel('Modèle')
    plt.ylim([-1,1])
    plt.xlim([0,50])
    plt.title("MCC sur l'ensemble des données par phase")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.scatter(np.arange(0,50),mt_S1,label='Première phase',s=100)
    plt.scatter(np.arange(0,50),mt_S2,label='Deuxième phase',s=100)
    plt.scatter(np.arange(0,50),mt_all,label='Phases mélangées',s=100)
    plt.plot(np.arange(0,51),np.zeros(51),color='k')
    plt.ylabel('MCC')
    plt.xlabel('Modèle')
    plt.ylim([-1,1])
    plt.xlim([0,50])
    plt.title("MCC sur les données de test par phase")
    plt.legend()
    plt.show()
        