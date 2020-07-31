# -*- coding: utf-8 -*-
"""
Ce code permet de tester un modèle simple de GRU sur une sinusoïde 
bruitée.
"""

# Modules
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Functions
"Forme une sinusoïde bruitée"
def noisy_sin(time,coeff=10):
    """
    :param time: array, pas de temps utilisés
    :param coeff: float, coefficient augmentant la taille de la
    sinusoïde sans augmenter le bruit
    :return np.asarray(values): array, valeurs de la sinusoïde bruitée
    """
    values=[]
    for t in time:
        values.append(np.sin(t)*coeff+random.random())
    return np.asarray(values)

"Prédit le signal obtenu à partir du modèle entraîné"
def predict_signal(model,time,begin,true_signal):
    """
    :param model: modèle entraîné
    :param time: array, pas de temps utilisés
    :param begin: array, début du signal à prédire
    :param true_signal: array, signal que l'on devrait obtenir
    :return predicted_signal: array, signal prédit par le modèle
    """
    predicted_signal=begin
    for i in range(true_signal.shape[0]-begin.shape[0]):
        result=model(predicted_signal[i:600+i])
        predicted_signal.append(result)
    return predicted_signal
        
# Main
if __name__ == "__main__":
    "On choisit une seed pour conserver le même bruit sur notre sinus"
    random.seed(20)
    temps=np.arange(0,1400,0.1)
    valeurs=noisy_sin(temps,coeff=3)
    val_train=[]
    lab_train=[]
    val_test=[]
    lab_test=[]
    
    """
    On construit les données que l'on enverra dans le modèle, des
    ensemble de 600 points translaté à chaque fois de 1 point, avec
    le point suivant comme valeur à prédire par le modèle
    """
    for i in range(11400):
        val_train.append(valeurs[i:600+i])
        lab_train.append(valeurs[600+i])
    for i in range(11400,13400):
        val_test.append(valeurs[i:600+i])
        lab_test.append(valeurs[600+i])
    val_train=np.asarray(val_train).reshape(11400,600,1)
    lab_train=np.asarray(lab_train)
    val_test=np.asarray(val_test).reshape(2000,600,1)
    lab_test=np.asarray(lab_test)
    
    "On construit notre modèle"
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(8,input_shape=(600,1)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError())
    model.summary()
    
    "On entraîne et évalue notre modèle"
    model.fit(val_train,lab_train,epochs=10)
    model.evaluate(val_test,lab_test,verbose=2)
    
    "On observe les résultats"
    lims=[-4,5]
    sample=val_test
    sample_label=lab_test
    result=model(sample)
    
    plt.figure()
    plt.scatter(result,sample_label,s=5,color='k')
    plt.xlabel('Valeurs prédites')
    plt.ylabel('Valeurs réelles')
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xticks(np.arange(-4,5,1),labels=[])
    plt.yticks(np.arange(-4,5,1),labels=[])
    plt.plot(lims,lims,color='r',label='Prédiction exacte')
    plt.title('Comparaison entre valeurs prédites et valeurs réelle')
    plt.legend()
    
    plt.figure()
    plt.plot(temps[12000:12500],valeurs[12000:12500],label='Signal réel',color='k')
    plt.plot(temps[12000:12500],result[:500],label='Signal prédit',color='r')
    plt.xlabel('Temps')
    plt.ylabel('Valeur de la sinusoïde bruitée')
    plt.ylim(lims)
    plt.xlim([1200,1250])
    plt.xticks(np.arange(1200,1250,5),labels=[])
    plt.yticks(np.arange(-4,5,1),labels=[])
    plt.title('Comparaison entre signal prédit et signal réel')
    plt.legend()
    
    plt.show()