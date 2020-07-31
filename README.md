ML_GRU_2Hz.py : Programme principal, récupérant les signaux et formant un modèle GRU qui s'entraîne sur les données. Il faut adapter le nom du dossier contenant les signaux et les métainformations.

Noisy_sin.py : Programme montrant le fonctionnement d'un modèle GRU sur une sinusoïde bruitée

read_results.py : Programme lisant les résultats obtenus à l'aide du programme principal, avec 5 split différents des données et 10 modèles entraînés par split, qui permet de mieux observer et manipuler les données de results_S1.txt,results_S2.txt et results_all.txt

results_S1.txt : Résultats pour la phase 1. Se lit de la manière suivante : les 4 premiers nombres sont les Vrai négatifs, faux négatifs, faux positifs et vrai positif des données de test, les 4 suivants de l'ensemble des données, et les 2 suivants (soit un nombre soit nan) sont les coefficient de corrélation de Matthews des données de test ou de l'ensemble des données. 50 ensemble de données du genre en tout, soit 10 modèle entraîné pour 5 split différents. 

results_S2.txt : Voir results_S1.txt, semblable mais sur la phase 2

results_all.txt : Voir results_all.txt, semblable mais sur les phases mélangées
