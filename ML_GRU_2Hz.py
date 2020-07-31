# -*- coding: utf-8 -*-
"""
Ce code fait un premier modèle de machine learning sur 
l'ensemble des données séparées en 2 phases, afin d'obtenir 2 ou 3
classes correspondant à des valeurs de pH.
"""

# Modules
import numpy as np
import sys
import os
import scipy.io
import scipy.signal
import math
import tensorflow as tf
from matplotlib import pyplot as plt
import time

# Functions
"Récupère le nom de chaque fichier dans un dossier"
def getnames(directory):
    """
    :param directory: dossier que l'on veut lister
    :return NAMES: list, ensemble des noms des fichiers du dossier
    """
    NAMES=[]
    for root, dirs, files in os.walk(directory):
        for filename in files:
            NAMES.append(filename)
    return NAMES

"Permet de split les données en données de test et d'entraînement"
def split_data(data,labels,ratio=0.1):
    """
    :param data: array, ensemble des données
    :param labels: array, ensemble des résultats des données
    :param ratio: float, optionnel, permet de définir le ratio de
    données de test
    :return SIG_train,LAB_train,SIG_test,LAB_test: respectivement
    données et résultats d'entraînement et de test
    """
    countneg=labels.count(0)
    countpos=labels.count(1)
    SIG_train=[]
    LAB_train=[]
    SIG_test=[]
    LAB_test=[]
    for i in range(len(data)):
        if labels[i]==0:
            if countneg>labels.count(0)*ratio:
                countneg+=-1
                SIG_train.append(data[i])
                LAB_train.append(labels[i])
            elif countneg<=labels.count(0)*ratio:
                SIG_test.append(data[i])
                LAB_test.append(labels[i])
        if labels[i]==1:
            if countpos>labels.count(1)*ratio:
                countpos+=-1
                SIG_train.append(data[i])
                LAB_train.append(labels[i])
            elif countpos<=labels.count(1)*ratio:
                SIG_test.append(data[i])
                LAB_test.append(labels[i])
    SIG_train=np.asarray(SIG_train)
    SIG_test=np.asarray(SIG_test)
    LAB_train=np.asarray(LAB_train)
    LAB_test=np.asarray(LAB_test)
    return SIG_train,LAB_train,SIG_test,LAB_test

"Calcule le déséquilibrage des classes"
def class_imbalance(labels,verbose=0):
    """
    :param labels: array, ensemble des résultats
    :param verbose: 0 ou 1, optionnel, affiche les résultats
    :return output_bias: permet d'initialiser le biais de sortie du
    modèle
    :return class_weight: permet de balancer le poids de chaque
    classe dans le modèle
    """
    pos=labels.count(1)
    neg=labels.count(0)
    if verbose==1: 
        print('Nombre de cas positifs pour S1:', pos)
        print('Nombre de cas négatifs pour S1:', neg)
    total=pos+neg
    output_bias=np.log(pos/neg)
    weight_for_0=(1/neg)*(total)/2.0 
    weight_for_1=(1/pos)*(total)/2.0
    class_weight={0:weight_for_0,1:weight_for_1}
    if verbose==1:
         print('Poids de la classe 0 : {:.2f}'.format(weight_for_0))
         print('Poids de la classe 1 : {:.2f}'.format(weight_for_1))
    return output_bias,class_weight
    
    
"Construit un modèle de machine learning"
def make_model(output_bias=None):
    """
    :param output_biais: float,initialise le biais de l'output
    :return model: modèle de machine learning
    """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model=tf.keras.Sequential()
    #Mettre 600 ou 1200
    model.add(tf.keras.layers.GRU(8,input_shape=(1200,1)))
    #Mettre 2 ou 3
    model.add(tf.keras.layers.Dense(2,bias_initializer=output_bias))
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
    return model

"Permet de calculer la matrice de confusion et le MCC"
def check_cmat(model,data,labels,verbose=0):
    """
    :param model: modèle de machine learning entraîné
    :param data: array, ensemble des données
    :param labels: array, ensemble des résultats
    :param verbose: 0 ou 1, optionnel, affiche les résultats
    :return cmat: array, matrice de confusion
    :return MCC: float, Coefficient de Corrélation de Matthews
    """
    result_raw=model(np.asarray(data))
    result=[None for x in range(len(data))]
    for i in range(len(data)):
        result[i]=tf.math.argmax(result_raw[i])
    cmat=tf.math.confusion_matrix(labels,result,num_classes=2)
    if verbose==1:
        print("Matrice de confusion :")
        print(cmat.numpy())
    TP=cmat[1][1].numpy()
    TN=cmat[0][0].numpy()
    FP=cmat[1][0].numpy()
    FN=cmat[0][1].numpy()
    if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)==0:
        MCC=np.nan
        if verbose==1:
            print('Impossibilité de calculer le coefficient de corrélation de Matthews (division par 0)')
    else:
        MCC=(TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if verbose==1:
            print('Coefficient de Corrélation de Matthews :',MCC)
    return cmat.numpy(),MCC

"Permet d'évaluer les capacités d'un modèle entraîné"
def eval_model(model,data_test,labels_test,data_all,labels_all):
    """
    :param model: modèle de machine learning entraîné
    :param data_test: array, ensemble des données de test
    :param labels_test: array, ensemble des résultats réels de test
    :param data_all: array, ensemble des données
    :param labels_all: array, ensemble des résultats 
    :return cmat_test: array, matrice de confusion des données
    de test
    :return MCC_test: float, Coefficient de Corrélation de Matthews
    des données de test
    :return cmat_all: array, matrice de confusion
    :return MCC_all: float, Coefficient de Corrélation de Matthews
    """
    evaluation=model.evaluate(data_test,labels_test,verbose=2)
    print('Loss de {} et précision de {}%'.format(round(evaluation[0],4),round(evaluation[1]*100,2)))
    cmat_test,MCC_test=check_cmat(model,data_test,labels_test)
    cmat_all,MCC_all=check_cmat(model,data_all,labels_all)
    return cmat_test,MCC_test,cmat_all,MCC_all
    
"Permet d'afficher le coût pour les données de test et d'entraînement"
def plot_loss(history, label):
    """
    :param history: mettre le model.fit() donnant l'historique
    :param label: string, nom du fichier
    """ 
    plt.plot(history.epoch,history.history['loss'], 
                 label='Train '+label)
    plt.plot(history.epoch,history.history['val_loss'],
             label='Val '+label, linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()  

# Main
if __name__ == "__main__":
    start_time = time.time()
    
    "On récupère les métainfos et les noms des fichiers"
    metainfo=scipy.io.loadmat('.\data\metainfo_currentDB.mat')['metainfo']
    NOMS=getnames('.\data\matfiles')
    
    "On récupère les informations qui nous seront utiles"
    DUREE_EXP=metainfo[0][0][34]
    PH=metainfo[0][0][1]
    
    "On construit un système à 2 et 3 classes pour le pH"
    ACIDOSE=np.zeros(PH.shape)
    for i in range(PH.shape[0]):
        if PH[i]<=7.05:
            ACIDOSE[i]=1
    CLASSES=np.zeros(PH.shape)
    for i in range(PH.shape[0]):
        if PH[i]<=7.05:
            CLASSES[i]=1
        elif PH[i]<7.2:
            CLASSES[i]=2
            
    "On récupère les données et on ré-échantillonne à 2Hz."
    SIGNAUX=[]
    count=0
    for name in NOMS:
        count+=1
        sys.stdout.write("\r{0}%".format(round((float(count)/1819)*100,2)))
        sys.stdout.flush()
        SIGNAUX.append(scipy.io.loadmat('.\data\matfiles\{}'.format(name))['bpm'])
    print(' : Données récupérées')
    
    "On ré-échantillonne à 2Hz le signal initialement à 10Hz"
    N=81920
    freq=2
    SIGNAUX_res=[None for x in range(len(SIGNAUX))]
    
    """On doit adapter le resample à la taille réelle de chaque 
    signal. En effet, ils font tous 81920 cependant certains ont 
    au début des valeurs nan.
    """
    for i in range(len(SIGNAUX)):
        sys.stdout.write("\r{0}%".format(round((float(i+1)/1819)*100,2)))
        sys.stdout.flush()
        SIZE=N
        beginning=0
        for j in range(N):
            if np.isnan(SIGNAUX[i][j])==True:
                SIZE+=-1
            else:
                beginning=j
                break
        SIGNAUX_res[i]=scipy.signal.resample(SIGNAUX[i][beginning:],SIZE//5)
    print(' : Données ré-échantillonnées')
    
    "On peut mixer les données pour modifier le split des données"
    index_all=list(zip(SIGNAUX_res,DUREE_EXP,PH,ACIDOSE,NOMS,CLASSES))
    np.random.shuffle(index_all)
    SIGNAUX_res,DUREE_EXP,PH,ACIDOSE,NOMS,CLASSES=zip(*index_all)
    
    "On sépare toutes les données suivant la durée du stage 2"
    SIG_FILTERED_S1=[]
    SIG_FILTERED_S2=[]
    SIG_FILTERED_all=[]
    PH_S1=[]
    PH_S2=[]
    PH_all=[]
    ACIDOSE_S1=[]
    ACIDOSE_S2=[]
    ACIDOSE_all=[]
    NOMS_S1=[]
    NOMS_S2=[]
    NOMS_all=[]
    CLASSES_S1=[]
    CLASSES_S2=[]
    CLASSES_all=[]
    
    longueur=600
    for i in range(len(DUREE_EXP)):
        if math.isnan(DUREE_EXP[i][0])==True:
            pass
        elif DUREE_EXP[i][0]<=15:
            SIG_FILTERED_S1.append(SIGNAUX_res[i][-((120+longueur)*freq):-(120*freq)])
            PH_S1.append(PH[i])
            ACIDOSE_S1.append(ACIDOSE[i])
            CLASSES_S1.append(CLASSES[i])
            NOMS_S1.append(NOMS[i][:-9])
            SIG_FILTERED_all.append(SIGNAUX_res[i][-((120+longueur)*freq):-(120*freq)])
            PH_all.append(PH[i])
            ACIDOSE_all.append(ACIDOSE[i])
            CLASSES_all.append(CLASSES[i])
            NOMS_all.append(NOMS[i][:-9])
        elif DUREE_EXP[i][0]>15:
            SIG_FILTERED_S2.append(SIGNAUX_res[i][-((300+longueur)*freq):-(300*freq)])
            PH_S2.append(PH[i])
            ACIDOSE_S2.append(ACIDOSE[i])
            CLASSES_S2.append(CLASSES[i])
            NOMS_S2.append(NOMS[i][:-9])
            SIG_FILTERED_all.append(SIGNAUX_res[i][-((300+longueur)*freq):-(300*freq)])
            PH_all.append(PH[i])
            ACIDOSE_all.append(ACIDOSE[i])
            CLASSES_all.append(CLASSES[i])
            NOMS_all.append(NOMS[i][:-9])
        else:
            print('ERROR in {}'.format(i))
    
    SIG_FILTERED_S1_train,ACIDOSE_S1_train,SIG_FILTERED_S1_test,ACIDOSE_S1_test=split_data(SIG_FILTERED_S1,ACIDOSE_S1)
    SIG_FILTERED_S2_train,ACIDOSE_S2_train,SIG_FILTERED_S2_test,ACIDOSE_S2_test=split_data(SIG_FILTERED_S2,ACIDOSE_S2)
    SIG_FILTERED_all_train,ACIDOSE_all_train,SIG_FILTERED_all_test,ACIDOSE_all_test=split_data(SIG_FILTERED_all,ACIDOSE_all)
    
    """
    On construit le modèle pour la phase 1 et pour la phase 2. On 
    commence par initialiser un biais pour que l'apprentissage soit
    plus facile avec les classes déséquilibrées.
    On va ensuite modifier le poids de chaque classe pour qu'elle
    corresponde au nombre de cas. Les cas d'acidose étant plus rares,
    il faut leur donner plus de poids pour que le modèle y fasse
    plus attention
    """
    output_biais_S1,class_weight_S1=class_imbalance(ACIDOSE_S1)
    output_biais_S2,class_weight_S2=class_imbalance(ACIDOSE_S2)
    output_biais_all,class_weight_all=class_imbalance(ACIDOSE_all)
    
    "On entraîne et évalue notre modèle"
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        verbose=1,
        patience=4,
        mode='min',
        restore_best_weights=True)
    
    
    # model_S1=make_model()
    # history_S1=model_S1.fit(SIG_FILTERED_S1_train,ACIDOSE_S1_train,
    #                         epochs=100, callbacks=[early_stopping],
    #                         class_weight=class_weight_S1,
    #                         validation_data=(SIG_FILTERED_S1_test,
    #                                          ACIDOSE_S1_test))
    # cmat_test_S1,MCC_test_S1,cmat_all_S1,MCC_all_S1=eval_model(
    #     model_S1,SIG_FILTERED_S1_test,ACIDOSE_S1_test,
    #     SIG_FILTERED_S1,ACIDOSE_S1)
    # print(cmat_test_S1)
    # plt.figure()
    # plot_loss(history_S1,'Stage 1')
    
    
    model_S2=make_model()
    history_S2=model_S2.fit(SIG_FILTERED_S2_train,ACIDOSE_S2_train,
                            epochs=10,callbacks=[early_stopping],
                            class_weight=class_weight_S2,
                            validation_data=(SIG_FILTERED_S2_test,
                                              ACIDOSE_S2_test))
    cmat_test_S2,MCC_test_S2,cmat_all_S2,MCC_all_S2=eval_model(
        model_S2,SIG_FILTERED_S2_test,ACIDOSE_S2_test,
        SIG_FILTERED_S2,ACIDOSE_S2)
    plt.figure()
    plot_loss(history_S2,'Stage 2')
    
    # model_all=make_model()
    # history_all=model_S1.fit(SIG_FILTERED_all_train,ACIDOSE_all_train,
    #                         epochs=10, callbacks=[early_stopping],
    #                         class_weight=class_weight_all,
    #                         validation_data=(SIG_FILTERED_all_test,
    #                                          ACIDOSE_all_test))
    # cmat_test_all,MCC_test_all,cmat_all_all,MCC_all_all=eval_model(
    #     model_all,SIG_FILTERED_all_test,ACIDOSE_all_test,
    #     SIG_FILTERED_all,ACIDOSE_all)
    # plt.figure()
    # plot_loss(history_all,'Tous les stages')

    print("--- %s seconds ---" % (time.time() - start_time))
    
    """
    Pour sauvegarder les poids d'un modèle, utiliser 
    model.save_weights('./checkpoints/noms_du_checkpoint').
    Pour charger des poids, utiliser
    model.load_weights('./checkpoints/noms_du_checkpoint').
    """
