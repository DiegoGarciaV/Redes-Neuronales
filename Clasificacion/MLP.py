import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as mt

class MLP:
    
    #Coleccion de numero de neuronas por capa
    T = []
    
    #coleccion de transformaciones lineales
    F = []
    L = 0
    
    #Coleccion de umbrales 
    B = []
    
    #Coleccion de salidas de cada capa (Se utilizarán para el Backpropagation)
    
    A = []
    
    def __init__(self,F0,B0):
        
        self.F = F0
        self.B = B0
        
    def __init__(self,Top):
        
        L = len(Top)
        self.F = []
        self.B = []
        self.T = Top
        for t in range(0,len(Top)-1):
            
            
            #R = []
            
            #for i in range(0,Top[t+1]):
            #    R.append(np.random.normal(0,1,Top[t]))
                
            #self.B.append(np.random.normal(0,3,Top[t+1]))
            #self.F.append(np.array(R))
            self.B.append(np.random.rand(1,Top[t+1])* 4 -2 ) 
            self.F.append(np.random.rand(Top[t],Top[t+1])*4 -2)
            
            
            
        
    #funcion de activación
    def sigmoide(self,x):
        
            sig = 1 / (1 + np.exp(-x))
            return sig
        
    def sigVector(self,X):
        
        p = []
        for x in X:
            p.append(self.sigmoide(x))
            
        return np.array(p)
    
    def aplicar(self,X):
        
        Fi = self.F
        Bi = self.B
        self.A = []
        
        if(len(Fi) != len(Bi)):
            return "Matrices incompatibles"
        
        Xi = X

        
        for i in range(0,len(Fi)):
            
            Ti = Fi[i]
            Y =  Xi @ Ti + Bi[i]
            
            Xi = self.sigmoide(Y)
            self.A.append(Xi)
            
            
        return np.array(Xi)
            
        
    def agregaCapa(self,A,b):
        
        dSalida =len(self.F[len(self.F)-1])
        dEntrada = len(A[0])
        if(dSalida == dEntrada):
            self.F.append(A)
            self.B.append(b)
        else:
            return "Los tamaños de matrices no coinciden"
        
    def ajustaParam(self,capa,h,k,p):
        
        Fi = self.F
        
        if(capa < len(Fi)):
            if(h < len(Fi[capa]) and k < len(Fi[capa][0])):
                self.F[capa][h][k] = p
                return 1
            
        return 0
    
    def obtieneParam(self,capa,h,k):
        
        Fi = self.F
        
        if(capa < len(Fi)):
            if(h < len(Fi[capa]) and k < len(Fi[capa][0])):
                return self.F[capa][h][k]
            
        return 0
    
    def ajustaBias(self,capa,i,b):
        
        Bi = self.B
        
        if(capa < len(Bi)):
            if(i < len(Bi[capa])):
                self.B[capa][i] = b
                return 1
            
        return 0
    
    def obtieneBias(self,capa,i):
        
        Bi = self.B
        
        if(capa < len(Bi)):
            if(i < len(Bi[capa])):
                return self.B[capa][i]
            
        return 0
    
    
    
    def numNeuronas(self):
        
        k = 0
        for Fi in self.F:
            k = k + len(Fi)*(len(Fi[0])+1)
            
        return k
    
    
    def entrenamiento(self,X,Y,costo,dT):

        a = self.aplicar(X)

        #backProp

        deltas = []

        for i in reversed(range(0,len(self.F))):

            a = self.A[i]

            if(i == len(self.F)-1):
                deltas.insert(0,costo(a,Y) * (a*(1-a)))
            else:
                deltas.insert(0,deltas[0] @ Fi * (a*(1-a)))

            Fi = self.F[i].T

            self.B[i] = self.B[i] - np.mean(deltas[0],axis=0,keepdims=True) * dT

            if(i>0):
                ai = self.A[i-1].T
            else:
                ai = X.T
            self.F[i] = self.F[i] - ai  @ deltas[0] * dT
    
        return  error(self.aplicar(X),Y)