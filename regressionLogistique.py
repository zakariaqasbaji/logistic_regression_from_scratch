# -*- coding: utf-8 -*-
"""
# 1)-L'importation de dataset

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
labels = data['target']
print(labels)

le dataset du fichier sklearn.datasets est déja préparé et les données
catégorielles sont déja étiquetées .

on va télécharger le dataset du siteweb (non préparé) et on va le préparer à 
zéro pour pouvoir appliquer LabelEncoder.

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


# Préparation de notre dataset:
dataset = pd.read_csv('data.csv',sep=',')


dataset.drop(['Unnamed: 32'], axis=1, inplace=True)#champ inutile =>supression

dataset.drop(['id'], axis=1, inplace=True)#champ inutile =>supression

y=dataset.diagnosis.values

x = dataset.drop(['diagnosis'], axis=1)
#feature scalling:
x=(x-np.mean(x))/np.std(x)
#les données ne sont pas étiquetées


#-------------------------------------
#2)- Division du dataset

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


#-------------------------------------
#3)-étiquetage des données


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
y_test = labelencoder.fit_transform(y_test)
#-------------------------------------
#4)-Implementation et Entrainement
"""Remarque: sklearn.linear_model contient une classe nommée LogisticRegression 
qu'on utilise souvent pour faire la regression logistique mais dans cette
question on va implémenter à zéro les fonctions de regression logistique """


x_train = x_train.T
x_train=np.vstack((x_train, np.ones(x_train.shape)))

x_test = x_test.T
x_test=np.vstack((x_test, np.ones(x_test.shape)))

y_train = y_train.T
y_test = y_test.T
#initialisation des variables:
theta = np.full((x_train.shape[0],1),0.0)
epsilon=1e-10#on va ajouter epsilon dans la foncion log pour eviter RuntimeWarning: divide by zero encountered in log



#fonction sigmoid:

def sigmoid(z):
     return 1/( 1 + np.exp(-z) )

#fonction cout:
     
def cost_function(x,y,theta):
    y_pre = sigmoid(np.dot(theta.T,x))
    loss = -y*np.log(y_pre+epsilon)-(1-y)*np.log(1-y_pre+epsilon)
    return (np.sum(loss))/x.shape[1]
    
#gradient:
def grad(x,y,theta):
    y_pre = sigmoid(np.dot(theta.T,x))
    return (np.dot(x,((y_pre-y).T)))/x.shape[1]

def gradient_descent(x_train,y_train,theta,learning_rate,n_iterations):
    cost_history=[]
    for i in range(0,n_iterations):
        theta=theta-learning_rate*grad(x_train,y_train,theta)
        cost_history.append(cost_function(x_train,y_train,theta))
        if i % 10 == 0: #aprés chaque 10 itérations on affiche le cost
            print ("Le cost apres l'iteration %i est: %f" %(i,cost_function(x_train,y_train,theta)))
    return theta,cost_history

def predict(x_test,theta):
    y_pre=sigmoid(np.dot(theta.T,x_test))
    y_predicted=[1 if i>=0.5 else 0 for i in y_pre.T]
    return y_predicted

#Fonction de précision :
    
def accuracy(y_true,y_pred):
    accuracy=np.sum( y_true==y_pred)/len(y_true)
    return accuracy*100



#entrainement:
n_iterations=2500
print("******************************La descente de gradient******************************")

start_time=time.time()
theta_final,cost_history=gradient_descent(x_train, y_train, theta, learning_rate=0.01,n_iterations=n_iterations)
print("le temps d'execution est:",time.time()-start_time)
#La précision de notre modèle:
y_pred=predict(x_test, theta_final)
print("la precision du test:",accuracy(y_test,y_pred),"%" )

#tracage de cout en fonction des itérations:
plt.plot(range(n_iterations),cost_history)
plt.title("Le cout en \n fonction de nombre d'iterations")
plt.xlabel("nombre d'iterations")
plt.ylabel("le cout")
plt.show()

#-------------------------------------
#5)-gradient stochastique:

def stochastic_gradient_descent(x_train,y_train,theta,learning_rate, n_iterations):
    m = len(y_train)
    cost_history = []
 
    for i in range(n_iterations):        
        cost=0.0        
        for j in range(0,m):
            rand_indice=np.random.randint(0,m)
            x_i = x_train[:,rand_indice].reshape(x_train.shape[0],1)
            y_i = y_train[rand_indice].reshape(1,1)
            theta=theta-learning_rate*grad(x_i,y_i,theta)
            cost+=cost_function(x_i,y_i,theta)
        cost_history.append(cost)
        #on affiche le cout a chaque iteration
        print ("Le cost apres l'iteration %i est: %f" %(i,cost_history[i]))
    return theta, cost_history 
n_iterations_sgd=5
#entrainement:   
print("**********************La descente de gradient stochastique**********************")
start_time=time.time()
theta_final_stochastic,cost_history_stochastic =stochastic_gradient_descent(x_train,y_train,theta,learning_rate=0.01, n_iterations=n_iterations_sgd)
print("le temps d'execution est:",time.time()-start_time)
#La précision de notre modèle:
y_pred_stochastic=predict(x_test, theta_final_stochastic)
print("la precision du test:",accuracy(y_test,y_pred_stochastic) )
plt.plot(cost_history_stochastic)
plt.title("Le cout en \n fonction de nombre d'iterations")
plt.xlabel("nombre d'iterations")
plt.ylabel("le cout")
plt.show()
