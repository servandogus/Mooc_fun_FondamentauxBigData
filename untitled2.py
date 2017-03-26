#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 18:08:12 2017

@author: srn
"""

import numpy as np

# ---------------
# Estimation de moindre carré
# ---------------

# X est la matrice des observations
X = np.eye(8)
X[0,-1] = 1

# y est le vecteur des résultats
y = np.array([1,1,1,1,1,1,1,1])

# Estimateur des moindres carrés : Beta
Beta_X = np.dot(np.transpose(X), X)
Beta_X = np.linalg.inv(Beta_X)
Beta_X = np.dot(Beta_X, np.transpose(X))
Beta_X = np.dot(Beta_X, y)

# Question 1 : somme des Beta(i)
print("\nQuestion 1: somme des Beta_X(i) = {}".format(sum(Beta_X)))
np
# Question 2
temp = np.dot(np.transpose(X), X) - np.eye(8)
print("\nQuestion 2: np.linalg.matrix_rank(Xt.X - I8) = {}".format(np.linalg.matrix_rank(temp)))

# Question 3
#importation des données
f = open("winequality-white.csv", "r")
Z = np.genfromtxt("winequality-white.csv", delimiter=";")
# on retire la ligne d'en tete, et la dernière colonne qui représente le vecteur des résutlats
y = Z[1:,-1]
Z = Z[1:,:-1] 
# estimateur des moindres carrés
Beta_Z = np.dot(np.transpose(Z), Z)
Beta_Z = np.linalg.inv(Beta_Z)
Beta_Z = np.dot(Beta_Z, np.transpose(Z))
Beta_Z = np.dot(Beta_Z, y)
# estimation:
res = np.dot(Z, Beta_Z)
# erreur:
err = res - y
# Somme des ecarts au carré
err_s = np.linalg.norm(err)**2
print("\nQuestion 3: Somme des erreurs au carré = {}".format(err_s))

# Question 4, normer et centrer les observations
def standardize(z):
    m = np.mean(z)
    s = np.std(z)
    res = [(i-m)/s for i in z]
    return res

for i in range(Z.shape[1]):
    Z[:,i] = standardize(Z[:,i])
 
# estimateur des moindres carrés
Beta_Z2 = np.dot(np.transpose(Z), Z)
Beta_Z2 = np.linalg.inv(Beta_Z2)
Beta_Z2 = np.dot(Beta_Z2, np.transpose(Z))
Beta_Z2 = np.dot(Beta_Z2, y)
# estimation:
res = np.dot(Z, Beta_Z2)
# erreur:
err = res - y
# Somme des ecarts au carré
err_s = np.linalg.norm(err)**2
print("\nQuestion 3: Somme des erreurs au carré = {}".format(err_s))

