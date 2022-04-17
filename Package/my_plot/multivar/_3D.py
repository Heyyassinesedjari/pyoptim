import numpy as np                        #Manipulation des Matrices
from numpy.linalg import norm             #Calcule de la norme euclidienne comme critère d'arrêt du programme      
from sympy import diff, symbols           #Calcule symbolique de la dérivée partielle de f ou g afin de calculer le gradient stochastique selon une direction donnée 
import random as r                        #Choix aléatoire de la direction de descente du gradient stochastique
import time                               #Mesure de la performance temporelle des programmes
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d          #Dessin 3D des fonctions à deux variables de test et des points choisis par les differente methodes de la descente du gradient
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as spo
import math as m
import numdifftools as nd  #Pour calculer le gradient d'une fonction multivariable pas d'une liste  nd.Gradient(f)(Xk_1)
from time import perf_counter


def sgd(f,X,tol,step_size=0.01):
    """+ sgd takes 5 arguments:
- f : a unimodal function that takes a numpy array an argument
*Important: the numpy array must be of size 2
- X : a starting column vector of size 2
- tol : the tolerence
- initial_step_size : the depth of the descent, it's a real value.
- c: Armijo dividing coefficient c>1
       + It returns the argmin(f): a numpy array of size 2 & the 3D plot of the search"""
    
#-----------------I) AVANT LA BOUCLE WHILE-------------------------------------------------------------------------------------    
    #-----------------I.1) Initialisation--------------------------------------------------------------------

    Coordonnées_x=[]                 #Listes des coordonees x et y des points calculés par la MGDS a chaque iteration.Ca sera utile par la site pour effectuer le plot
    Coordonnées_y=[]   
    n=len(X)                         #On a choisi n==2 pour pouvoir representer graphiquement la MGDS
    grad=np.zeros(n)                 #[0,0] 
    Dict_eval={}                     #un dictionnaire pour evaluer la derivée partielle par rapport a une direction xi avec la methode evalf(subs=Dict_eval)
    alpha=step_size                  # le pas fixe de descente (pour garantire la convergence on a pris le pas relativement faible)
    x, y=symbols('x y', real=True)   #x and y sont des symbols qui aideront a calculer le dérivée partielle 
    l=[0,0]             #n==2
    
    #-------------------I.2) Choix aléatoire de i pour la direction xi-------------------------------------------------------------- 

    X0=X
    Coordonnées_x.append(X0[0])
    Coordonnées_y.append(X0[1])
    i=r.randint(0,n-1)               #le premier indice i aléatoire pour choisir la direction de descente, an d'autre terme choix aléatoire de la variable sur laquelle en effectuera la dérivée partielle df/dxi   
    
    #------------------I.3) Calcule de la dérivée partielle par rapport xi=Xk_1[i] évaluée au point X0 ------------------------

    l[i]=x
    l[int(not i)]=y
    dfi_x0=diff(f(l),x)               #l'expression symbolique de la dérivée partielle par rapport a xi=x
    Dict_eval[x]=X0[i]
    Dict_eval[y]=X0[int(not i)]
    dfi_x0=dfi_x0.evalf(subs=Dict_eval)     #La dérivée partielle evalué en X0    
    
    #-----------------I.4) Construction du gradient gradf(X0) selon la direction xi, [0 ... 0 ,df/dxi(X0), 0 ... 0] ------------------------ 

    gradfi_x0=grad.copy()           # [0         ...             0]
    gradfi_x0[i]=dfi_x0             #[0 ... 0 , df/dxi(X0) , 0 ... 0] Le vecteur gradient selon une seule direction, celle de xi calculé juste au dessus
        #                                           ^
        #                                     i ème direction
        
    #-----------------I.5) Calcule de X1------------------------------------------------------------------------------

    X1=X0-alpha*gradfi_x0
    Coordonnées_x.append(X1[0])     # pour le plot
    Coordonnées_y.append(X1[1])

    #-------I.6) Choix aléatoire de  i ---------------------------------------------------------------------
    i_1=i
    i=r.randint(0,n-1)

    #-------I.7) Calcule de la dérivée partielle par rapport xi évaluée au point X1 ------------------------  
    l[i_1]=0
    l[i]=x
    l[int(not i)]=y
    dfi_x1=diff(f(l),x)                    #l'expression symbolique de la dérivée partielle par rapport a xi=x
    
    Dict_eval[x]=X1[i]
    Dict_eval[y]=X1[int(not i)]
    dfi_x1=dfi_x1.evalf(subs=Dict_eval)    #La dérivée partielle evalué en X1  

    #-------I.8) Calcule du gradient gradf(X1) selon la direction xi = [0 ... 0 ,df/dxi(X1), 0 ... 0] ------------------------ 

    gradfi_x1=grad.copy()                  # [0         ...             0]
    gradfi_x1[i]=dfi_x1                    #[0 ... 0 , df/dxi(X1) , 0 ... 0] Le vecteur gradient selon une seule direction, celle de xi calculé juste au dessus
        #                                                  ^
        #                                          i ème direction
        

    #--------I.9) Calcule de la norme euclidienne -------------------------------------------------------------------------------------------------
    
    n_gradfi_x1=norm(gradfi_x1)
    n_distance=norm(X1-X0)

#--------II. MDGS-loop---------------------------------------------------------------------------------------------------------------
    
    num_iter=1
    while(n_gradfi_x1>tol or n_distance>tol):
        X0=X1
        gradfi_x0=gradfi_x1                     
    
        X1=X0-alpha*gradfi_x0
        
        Coordonnées_x.append(X1[0])
        Coordonnées_y.append(X1[1])
        
    #-------II.1) Choix aléatoire de i entre 0 et n-1 inclusifs ----------------- 
        i_1=i
        i=r.randint(0,n-1)
        
    #-------II.2) Calcule de la dérivée partielle par rapport xi évaluée au point X1 ------------------------ 
        l[i_1]=0
        l[i]=x
        l[int(not i)]=y
        dfi_x1=diff(f(l),x)                                #l'expression symbolique de la dérivée partielle par rapport a xi=x
        
        Dict_eval[x]=X1[i]
        Dict_eval[y]=X1[int(not i)]
        
        dfi_x1=dfi_x1.evalf(subs=Dict_eval)                #La dérivée partielle evalué en X1
        
    #-------II.3) Calcule du gradient gradf(X1) selon la direction xi = [0 ... 0 ,df/dxi(X1), 0 ... 0] ------------------------   

        gradfi_x1=grad.copy()
        gradfi_x1[i]=dfi_x1
        
    #--------II.4) Calcule de la norme euclidienne-------------------------------------------------------------------------------------------------        
        
        n_gradfi_x1= norm(gradfi_x1)
        n_distance = norm(X1-X0)
        
        num_iter+=1
        #--------Fin de la boucle-------------------------------------------------------------------------------------------------                
    w=tol
    p=0
    while(w<1):
        w*=10
        p+=1
    X1=np.round_(X1,decimals=p)
    print('Y* = ',X1)
    print('Le plot 3D apparaitra dans quelques secondes ...')
    print('Merci de patienter !')
    #--------Debut du plot 3D------------------------------------------------------------------------
    fig = plt.figure(figsize = (20, 20))
    ax = plt.axes(projection='3d')

    X=np.linspace(-1100,1100,1000)
    Y=np.linspace(-1100,1100,1000)

    Z=np.empty((1000,1000))
    for i in range(1000):
        for j in range(1000):
            Z[i,j]=f([X[i],Y[j]])

    ax.contourf(X, Y, Z, 300)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    x = np.array(Coordonnées_x)
    y = np.array(Coordonnées_y)
    z = f(np.array([Coordonnées_x,Coordonnées_y]))

    ax.scatter(x,y,z,c='r',s= 100)

    plt.show()



#============================================================================================================================================================
#============================================================================================================================================================

def sgd_with_bls(f,X,tol=0.01,initial_step_size=0.01,c=2):
    """ + sgd_with_bls takes 5 arguments:
- f : a unimodal function that takes a numpy array an argument
*Important: the numpy array must be of size 2
- X : a starting column vector of size 2
- tol : the tolerence
- initial_step_size : de depth of the descent, it's a real value.
- c: Armijo dividing coefficient c>1
       + It returns the argmin(f): a numpy array of size 2 & the 3D plot of the search """
#-----------------I) AVANT LA BOUCLE WHILE-------------------------------------------------------------------------------------    
    #-----------------I.1) Initialisation--------------------------------------------------------------------

    Coordonnées_x=[]    #Listes des coordonees x et y des points calculés par la MGDS a chaque iteration.Ca sera utile par la site pour effectuer le plot
    Coordonnées_y=[]
    n=len(X)            #On a choisi n==2 pour pouvoir representer graphiquement la MGDS
    grad=np.zeros(n)    #[0,0]
    Dict_eval={}        #un dictionnaire pour evaluer la derivée partielle par rapport a une direction xi avec la methode evalf(subs=Dict_eval)
    x, y=symbols('x y', real=True) #x and y sont des symbols qui aideront a calculer le dérivée partielle
    l=[0,0]             #n==2
    
    #-------------------I.2) Choix aléatoire de i pour la direction xi-------------------------------------------------------------- 

    X0=X
    Coordonnées_x.append(X0[0])
    Coordonnées_y.append(X0[1])
    i=r.randint(0,n-1)   #le premier indice i aléatoire pour choisir la direction de descente, 
                    # -> en d'autre terme c'est le choix aléatoire de la variable sur laquelle en effectuera la dérivée partielle df/dxi   
    
    #------------------I.3) Calcule de la dérivée partielle par rapport xi=Xk_1[i] évaluée au point X0 ------------------------
    
    l[i]=x
    l[int(not i)]=y
    dfi_x0=diff(f(l),x)               #l'expression symbolique de la dérivée partielle par rapport a xi=x
    
    Dict_eval[x]=X0[i]
    Dict_eval[y]=X0[int(not i)]
    
    dfi_x0=dfi_x0.evalf(subs=Dict_eval)     #La dérivée partielle evalué en X0     
    
    #-----------------I.4) Construction du gradient gradf(X0) selon la direction xi, [0 ... 0 ,df/dxi(X0), 0 ... 0] ------------------------ 

    gradfi_x0=grad.copy()           # [0         ...             0]
    gradfi_x0[i]=dfi_x0             #[0 ... 0 , df/dxi(X1) , 0 ... 0] Le vecteur gradient selon une seule direction, celle de xi calculé juste au dessus
        #                                                  ^
        #                                          i ème direction
        
#_____________________________________________"Backtracking line search using Armijo condition"_________________________________
    
    def phi(alpha):
        xalpha=X0-alpha*gradfi_x0                    #la fonction à minimiser
        return f(xalpha)
    
    alpha=initial_step_size                    #le pas initiale = 100 assez grand pour ne pas satisfaire la condition d'armijo dès le debut 
    eps=tol                                    #espilone = tolérence
    N=c                                       # A chaque fois que la condition d'Armijo n'est pas vérifié On divise le pas par N 


#                                                                phi(0 + epsilone_très_faible) - phi(0)
    phi_prime_de_0 = (phi(1.e-6)-phi(0))/1.e-6   #phi'(0) = ----------------------------------------
#                                                                        epsilone_très_faible                
    
    
    while(phi(alpha) > phi(0)+eps*alpha*phi_prime_de_0):      #tant que la condition d'Armijo n'est pas vérifié on divise le pas par N
        alpha=alpha/N
#__________________________________________________________________________________________________________________________  

    #---------I.5) Calcule de X1------------------------------------------------------------------------------

    X1=X0-alpha*gradfi_x0
    
    Coordonnées_x.append(X1[0])         # pour le plot
    Coordonnées_y.append(X1[1])

    #-------I.6) Choix aléatoire de  i ---------------------------------------------------------------------
    i_1=i
    i=r.randint(0,n-1)

    #-------I.7) Calcule de la dérivée partielle par rapport xi évaluée au point X1 ------------------------  

    l[i_1]=0
    l[i]=x
    l[int(not i)]=y
    dfi_x1=diff(f(l),x)                        #l'expression symbolique de la dérivée partielle par rapport a xi=x
    
    Dict_eval[x]=X1[i]
    Dict_eval[y]=X1[int(not i)]
    
    dfi_x1=dfi_x1.evalf(subs=Dict_eval)        #La dérivée partielle evalué en X1  

    #-------I.8) Calcule du gradient gradf(X1) selon la direction xi = [0 ... 0 ,df/dxi(X1), 0 ... 0] ------------------------ 

    gradfi_x1=grad.copy()                 # [0         ...             0]
    gradfi_x1[i]=dfi_x1                   #[0 ... 0 , df/dxi(X1) , 0 ... 0] Le vecteur gradient selon une seule direction, celle de xi calculé juste au dessus
        #                                                  ^
        #                                          i ème direction

    #--------I.9) Calcule de la norme euclidienne -------------------------------------------------------------------------------------------------
    
    n_gradfi_x1=norm(gradfi_x1)
    n_distance=norm(X1-X0)

#--------SGD-loop---------------------------------------------------------------------------------------------------------------

    num_iter=1
    while(n_gradfi_x1>tol or n_distance>tol):
        X0=X1
        gradfi_x0=gradfi_x1
        i_1=i
#_______________________________"Backtracking line search using Armijo condition"______________________________________________
    
        def phi(alpha):
            xalpha=X0-alpha*gradfi_x0              #la fonction à minimiser
            return f(xalpha)

        alpha=initial_step_size                                  #le pas initiale = 100 assez grand pour ne pas satisfaire la condition d'armijo dès le debut 
        eps=tol                                    #espilone = tolérence
        N=c                                        # A chaque fois que la condition d'Armijo n'est pas vérifié On divise le pas par N 


    #                      phi(0 + epsilone_très_faible) - phi(0)
    #           phi'(0) = ----------------------------------------
    #                               epsilone_très_faible
    
        phi_prime_de_0=(phi(1.e-6)-phi(0))/1.e-6

        while(phi(alpha) > phi(0)+eps*alpha*phi_prime_de_0):        #tant que la condition d'Armijo n'est pas vérifié on divise le pas par N
            alpha=alpha/N
#__________________________________________________________________________________________________________________________________

        X1=X0-alpha*gradfi_x0
        
        Coordonnées_x.append(X1[0])
        Coordonnées_y.append(X1[1])
        
        #-------II.1) Choix aléatoire de i entre 0 et n-1 inclusifs ----------------- 
        
        i=r.randint(0,n-1)
        
         #-------II.2) Calcule de la dérivée partielle par rapport xi évaluée au point X1------------------------ 
        
        l[i_1]=0
        l[i]=x
        l[int(not i)]=y
        dfi_x1=diff(f(l),x)
        
        Dict_eval[x]=X1[i]
        Dict_eval[y]=X1[int(not i)]
        
        dfi_x1=dfi_x1.evalf(subs=Dict_eval)
        
        #-------II.3) Calcule du gradient gradf(X1) selon la direction xi = [0 ... 0 ,df/dxi(X1), 0 ... 0] ------------------------ 

        gradfi_x1=grad.copy()
        gradfi_x1[i]=dfi_x1
        
        #--------II.4) Calcule de la norme euclidienne-------------------------------------------------------------------------------------------------        
        
        n_gradfi_x1= norm(gradfi_x1)
        n_distance = norm(X1-X0)
        num_iter+=1
        
        #--------Fin de la boucle-------------------------------------------------------------------------------------------------                
    w=tol
    p=0
    while(w<1):
        w*=10
        p+=1
    X1=np.round_(X1,decimals=p)
    print('Y* = ',X1)
    print('Le nombre de points parcourus lors de la descente est ',len(Coordonnées_x))
    print('Le plot 3D apparaitra dans quelques secondes ...')
    print('Merci de patienter !')
    #--------Debut du plot 3D------------------------------------------------------------------------
    fig = plt.figure(figsize = (20, 20))
    ax = plt.axes(projection='3d')

    X=np.linspace(-1100,1100,1000)
    Y=np.linspace(-1100,1100,1000)

    Z=np.empty((1000,1000))
    for i in range(1000):
        for j in range(1000):
            Z[i,j]=f([X[i],Y[j]])

    ax.contourf(X, Y, Z, 300)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    x = np.array(Coordonnées_x)
    y = np.array(Coordonnées_y)
    z = f(np.array([Coordonnées_x,Coordonnées_y]))

    ax.scatter(x,y,z,c='r',s= 100)

    plt.show()

#============================================================================================================================================================
#============================================================================================================================================================  
def gradient_descent(f,X,tol=0.01,alpha=0.01):
    """+ gradient_descent function takes 4 arguments:
- f : a unimodal function that takes a numpy array as an argument
- X : a starting numpy array
- tol : the tolerence
- alpha : the depth of the descent, it's a real value.
       + It returns the argmin(f): a numpy array of size 2 & the 3D plot of the search"""
    
    Coordonnées_x=[]
    Coordonnées_y=[]
    Xk_1=X
    Coordonnées_x.append(Xk_1[0])
    Coordonnées_y.append(Xk_1[1])
    
    gradfXk_1=nd.Gradient(f)(Xk_1)

    Xk=Xk_1-alpha*gradfXk_1

    Coordonnées_x.append(Xk[0])
    Coordonnées_y.append(Xk[1])
    
    gradfXk=nd.Gradient(f)(Xk)
    n_gradfXk=m.sqrt(gradfXk[0]**2 + gradfXk[1]**2)
    n_distance=m.sqrt((Xk-Xk_1)[0]**2 + (Xk-Xk_1)[1]**2)

    
    
    while(n_gradfXk>tol or n_distance>tol):
        Xk_1=Xk
        gradfXk_1=gradfXk
        
        Xk=Xk_1-alpha*gradfXk_1

        Coordonnées_x.append(Xk[0])
        Coordonnées_y.append(Xk[1])
        
        gradfXk=nd.Gradient(f)(Xk)
        
        n_gradfXk = m.sqrt(gradfXk[0]**2 + gradfXk[1]**2)
        n_distance = m.sqrt((Xk-Xk_1)[0]**2 + (Xk-Xk_1)[1]**2)
    w=tol
    p=0
    while(w<1):
        w*=10
        p+=1
    Xk=np.round_(Xk,decimals=p)
    print('Y* = ',Xk)
    print('Le nombre de points parcourus lors de la descente est ',len(Coordonnées_x))
    print('Le plot 3D apparaitra dans quelques secondes ...')
    print('Merci de patienter !')
    #--------Debut du plot 3D------------------------------------------------------------------------
    fig = plt.figure(figsize = (15, 15))
    ax = plt.axes(projection='3d')

    X=np.linspace(-1100,1100,1000)
    Y=np.linspace(-1100,1100,1000)

    Z=np.empty((1000,1000))
    for i in range(1000):
        for j in range(1000):
            Z[i,j]=f([X[i],Y[j]])

    ax.contourf(X, Y, Z, 300)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    x = np.array(Coordonnées_x)
    y = np.array(Coordonnées_y)
    z = f(np.array([Coordonnées_x,Coordonnées_y]))

    ax.scatter(x,y,z,c='r',s= 50)

    plt.show()
#============================================================================================================================================================
#============================================================================================================================================================
def gradient_conjugate(f,X,tol=0.01):   
    """+ gradient_conjugate function takes 3 arguments:
- f : a unimodal function that takes  a column vector as an argument
       *the Hessian must be positive definite 
- X : a starting column vector
- tol : the tolerence
       + It returns the argmin(f): a column vector of size 2 & the 3D plot of the search"""
    Coordonnées_x=[]
    Coordonnées_y=[]   
    X0=X
    Coordonnées_x.append(X0[0])
    Coordonnées_y.append(X0[1])
    d0=-nd.Gradient(f)(X0)
    Q=nd.Gradient(nd.Gradient(f))(np.array([1,1]))
    n=len(X)
    for k in range(0,n-1):
        alpha=(d0.T@d0)/(d0.T@Q@d0)
        X0=X0+alpha*d0
        Coordonnées_x.append(X0[0])
        Coordonnées_y.append(X0[1])
        beta=(nd.Gradient(f)(X0).T@Q@d0)/(d0.T@Q@d0)
        d0=-nd.Gradient(f)(X0)+beta*d0
    w=tol
    p=0
    while(w<1):
        w*=10
        p+=1
    X0=np.round_(X0,decimals=p)
    print('Y* = ',X0)
    print('Le nombre de points parcourus lors de la descente est ',len(Coordonnées_x))
    print('Le plot 3D apparaitra dans quelques secondes ...')
    print('Merci de patienter !')
    print('!!!!! This  method is not accurate in dimension 3 because it computes only one iteration, it\'s effective when the dimensoin n is very high > 1.e6')
    #--------Debut du plot 3D------------------------------------------------------------------------
    fig = plt.figure(figsize = (15, 15))
    ax = plt.axes(projection='3d')

    X=np.linspace(-1100,1100,1000)
    Y=np.linspace(-1100,1100,1000)

    Z=np.empty((1000,1000))
    for i in range(1000):
        for j in range(1000):
            Z[i,j]=f([X[i],Y[j]])

    ax.contourf(X, Y, Z, 300)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    x = np.array(Coordonnées_x)
    y = np.array(Coordonnées_y)
    z = f(np.array([Coordonnées_x,Coordonnées_y]))

    ax.scatter(x,y,z,c='r',s= 50)

    plt.show()
#============================================================================================================================================================
#============================================================================================================================================================
def newton(f,X,tol=0.01):
    """+ newton function takes 3 arguments:
- f : a unimodal function that takes  a numpy array as an argument
       *the Hessian must be invertible
- X : a starting numpy array
- tol : the tolerence
       + It returns the argmin(f): a column vector of size 2 & the 3D plot of the search"""
    Coordonnées_x=[]
    Coordonnées_y=[]   
    X0=X
    Coordonnées_x.append(X0[0])
    Coordonnées_y.append(X0[1])
    grad2fx0=nd.Gradient(nd.Gradient(f))(X0)
    if(np.linalg.det(grad2fx0)==0):
        return "gradient2f(X0) is not invertible"
    else:
        d0=-np.linalg.inv(grad2fx0)@nd.Gradient(f)(X0)
        n_d0=np.linalg.norm(d0)
        while(n_d0>tol):
            
            def phi(alpha):
                return f(X0-alpha*d0)
            
            alpha=spo.minimize(phi,1)
            alpha=alpha.x[0]
            X0=X0-alpha*d0
            Coordonnées_x.append(X0[0])
            Coordonnées_y.append(X0[1])
            grad2fx0=nd.Gradient(nd.Gradient(f))(X0)
            if(all(np.linalg.eigvals(grad2fx0)>0)):
                d0=-np.linalg.inv(grad2fx0)@nd.Gradient(f)(X0)
            else:
                w,=np.linalg.eig(grad2fx0)
                lambda_min=np.amin(w)
                n=len(X)
                inverse=np.linalg.inv(-lambda_min*np.identity(n)+grad2fx0)
                d0=-inverse@nd.Gradient(f)(X0)
            n_d0=np.linalg.norm(d0)
        w=tol
        p=0
        while(w<1):
            w*=10
            p+=1
        X0=np.round_(X0,decimals=p)
    print('Y* = ',X0)
    print('Le nombre de points parcourus lors de la descente est ',len(Coordonnées_x))
    print('Le plot 3D apparaitra dans quelques secondes ...')
    print('Merci de patienter !')
    #--------Debut du plot 3D------------------------------------------------------------------------
    fig = plt.figure(figsize = (15, 15))
    ax = plt.axes(projection='3d')

    X=np.linspace(-1100,1100,1000)
    Y=np.linspace(-1100,1100,1000)
    Z=np.empty((1000,1000))
    for i in range(1000):
        for j in range(1000):
            Z[i,j]=f([X[i],Y[j]])

    ax.contourf(X, Y, Z, 300)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    x = np.array(Coordonnées_x)
    y = np.array(Coordonnées_y)
    z = f(np.array([Coordonnées_x,Coordonnées_y]))

    ax.scatter(x,y,z,c='r',s= 50)

    plt.show()
#============================================================================================================================================================
#============================================================================================================================================================

def quasi_newton_dfp(f,X,tol=0.01):
    """+ quasi_newton_dfp takes 3 arguments:
- f : a unimodal function that takes  a numpy array as an argument
- X : a starting numpy array
- tol : the tolerence
       + It returns the argmin(f): a column vector of size 2 & the 3D plot of the search"""
    Coordonnées_x=[]
    Coordonnées_y=[]   
    X0=X
    Coordonnées_x.append(X0[0])
    Coordonnées_y.append(X0[1])
    g0=nd.Gradient(f)(X0).T

    H0=np.identity(len(X))
    n_g0=np.linalg.norm(g0)
    
    while(n_g0>tol):
        d0=-H0@g0

        def phi(a):
            return f(X0+a*d0)
        
        a=spo.minimize(phi,1)
        a=a.x[0]

        X0=X0+a*d0
        Coordonnées_x.append(X0[0])
        Coordonnées_y.append(X0[1])
        g_1=g0
        g0=nd.Gradient(f)(X0).T

        #Calculating H0 with DFP formula
        y0=g0-g_1
        
        A0=a*(np.outer(d0,d0.T))*(1/(d0.T@y0))
        B0=-(np.outer((H0@y0),(H0@y0).T)/(y0.T@H0@y0))
        
        H0 = H0 + A0 + B0
        n_g0=np.linalg.norm(g0)
    w=tol
    p=0
    while(w<1):
        w*=10
        p+=1
    X0=np.round_(X0,decimals=p)    
    print('Y* = ',X0)
    print('Le nombre de points parcourus lors de la descente est ',len(Coordonnées_x))
    print('Le plot 3D apparaitra dans quelques secondes ...')
    print('Merci de patienter !')
    #--------Debut du plot 3D------------------------------------------------------------------------
    fig = plt.figure(figsize = (15, 15))
    ax = plt.axes(projection='3d')

    X=np.linspace(-1100,1100,1000)
    Y=np.linspace(-1100,1100,1000)
    Z=np.empty((1000,1000))
    for i in range(1000):
        for j in range(1000):
            Z[i,j]=f([X[i],Y[j]])

    ax.contourf(X, Y, Z, 300)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    x = np.array(Coordonnées_x)
    y = np.array(Coordonnées_y)
    z = f(np.array([Coordonnées_x,Coordonnées_y]))
    ax.scatter(x,y,z,c='r',s= 50)

    plt.show()
#============================================================================================================================================================
#============================================================================================================================================================


def compare_all_time(f,X,tol,alpha,xstart_bls,n_bls=2):
    T=[0 for i in range(7)]
    P=[0 for i in range(6)]
    
    T[0]=perf_counter()
    P[0]=_gd_(f,X,tol,alpha)
    
    T[1]=perf_counter()
    P[1]=_gc_(f,X,tol)
    
    T[2]=perf_counter()
    P[2]= _n_(f,X,tol)
    
    T[3]=perf_counter()
    P[3]=_qndfp_(f,X,tol)
    
    T[4]=perf_counter()
    P[4]=_sgd_(f,X,tol,alpha)
    
    T[5]=perf_counter()
    P[5]=_sgdbls_(f,X,tol,xstart_bls,n_bls)
    
    T[6]=perf_counter()
    
    
    Optimization_Methods=["gradient_descent","gradient_conjugate","newton","quasi_newton_dfp","stochastic_gd","stochastic_gd_with_bls"]
    Time=[T[1]-T[0], T[2]-T[1], T[3]-T[2], T[4]-T[3], T[5]-T[4], T[6]-T[5]]
    plt.style.use('ggplot')
    plt.figure(figsize=(12,7))
    plt.barh(Optimization_Methods, Time)
    plt.title('Comparaison Temporelle')
    plt.ylabel("Les Méthodes d'Optimisations à 2 Variables et Sans Contraintes" )
    plt.xlabel('Temps en (s)')
    plt.show()



import scipy.optimize as spo

def compare_all_precision(f, X, tol, alpha, xstart_bls, n_bls=2):
    T=[0 for i in range(7)]
    P=[0 for i in range(6)]
    
    P[0]= _gd_(f,X,tol,alpha)    
    P[1]=  _n_(f,X,tol)    
    P[2]= _qndfp_(f,X,tol)
    P[3]= _sgd_(f,X,tol,alpha)
    P[4]= _sgdbls_(f,X,tol,xstart_bls,n_bls)
    
    Xopt=spo.minimize(f,X)

    Optimization_Methods=["gradient_descent","newton","quasi_newton_dfp","stochastic_gd","stochastic_gd_with_bls"]
    Precision_error=[norm(P[i]-Xopt.x) for i in range(5)]
    plt.style.use('ggplot')
    plt.figure(figsize=(12,7))
    plt.barh(Optimization_Methods, Precision_error)
    plt.title("Comparaison des erreurs")
    plt.ylabel("Les Méthodes d'Optimisations à 2 Variables et Sans Contraintes" )
    plt.xlabel("les écarts entre la valeur recherché et les valeurs trouvés")
    plt.show()



























#-------------------------------------------------------------------------------------------------------------------------------
def _gd_(f,X,tol,alpha):

    Xk_1=X
    gradfXk_1=nd.Gradient(f)(Xk_1)

    Xk=Xk_1-alpha*gradfXk_1
    gradfXk=nd.Gradient(f)(Xk)
    n_gradfXk=m.sqrt(gradfXk[0]**2 + gradfXk[1]**2)
    n_distance=m.sqrt((Xk-Xk_1)[0]**2 + (Xk-Xk_1)[1]**2)

    
    
    while(n_gradfXk>tol or n_distance>tol):
        Xk_1=Xk
        gradfXk_1=gradfXk        
        
        Xk=Xk_1-alpha*gradfXk_1
        gradfXk=nd.Gradient(f)(Xk)
        
        n_gradfXk = m.sqrt(gradfXk[0]**2 + gradfXk[1]**2)
        n_distance = m.sqrt((Xk-Xk_1)[0]**2 + (Xk-Xk_1)[1]**2)
    w=tol
    p=0
    while(w<1):
        w*=10
        p+=1
    Xk=np.round_(Xk,decimals=p)
    return Xk
#-------------------------------------------------------------------------------------------------------------------------------
def _gc_(f,X,tol):

    X0=X
    d0=-nd.Gradient(f)(X0)
    Q=nd.Gradient(nd.Gradient(f))(np.array([1,1]))
    n=len(X)
    for k in range(0,n-1):
        alpha=(d0.T@d0)/(d0.T@Q@d0)
        X0=X0+alpha*d0
        beta=(nd.Gradient(f)(X0).T@Q@d0)/(d0.T@Q@d0)
        d0=-nd.Gradient(f)(X0)+beta*d0
    w=tol
    p=0
    while(w<1):
        w*=10
        p+=1
    X0=np.round_(X0,decimals=p)
    return X0  
#-------------------------------------------------------------------------------------------------------------------------------
def _n_(f,X,tol):

    X0=X
    grad2fx0=nd.Gradient(nd.Gradient(f))(X0)
    if(np.linalg.det(grad2fx0)==0):
        return "gradient2f(X0) is not invertible"
    else:
        d0=-np.linalg.inv(grad2fx0)@nd.Gradient(f)(X0)
        n_d0=np.linalg.norm(d0)
        while(n_d0>tol):
            
            def phi(alpha):
                return f(X0-alpha*d0)
            
            alpha=spo.minimize(phi,1)
            alpha=alpha.x[0]
            X0=X0-alpha*d0
            grad2fx0=nd.Gradient(nd.Gradient(f))(X0)
            if(all(np.linalg.eigvals(grad2fx0)>0)):
                d0=-np.linalg.inv(grad2fx0)@nd.Gradient(f)(X0)
            else:
                w,=np.linalg.eig(grad2fx0)
                lambda_min=np.amin(w)
                n=len(X)
                inverse=np.linalg.inv(-lambda_min*np.identity(n)+grad2fx0)
                d0=-inverse@nd.Gradient(f)(X0)
            n_d0=np.linalg.norm(d0)
        w=tol
        p=0
        while(w<1):
            w*=10
            p+=1
        X0=np.round_(X0,decimals=p)
        return X0
#-------------------------------------------------------------------------------------------------------------------------------
    
def _qndfp_(f,X,tol):

    X0=X
    g0=nd.Gradient(f)(X0).T

    H0=np.identity(len(X))
    n_g0=np.linalg.norm(g0)
    
    while(n_g0>tol):
        d0=-H0@g0

        def phi(a):
            return f(X0+a*d0)
        
        a=spo.minimize(phi,1)
        a=a.x[0]

        X0=X0+a*d0
        g_1=g0
        g0=nd.Gradient(f)(X0).T

        #Calculating H0 with DFP formula
        y0=g0-g_1
        
        A0=a*(np.outer(d0,d0.T))*(1/(d0.T@y0))
        B0=-(np.outer((H0@y0),(H0@y0).T)/(y0.T@H0@y0))
        
        H0 = H0 + A0 + B0
        n_g0=np.linalg.norm(g0)
    w=tol
    p=0
    while(w<1):
        w*=10
        p+=1
    X0=np.round_(X0,decimals=p)    
    return X0



#-------------------------------------------------------------------------------------------------------------------------------

def _sgd_(f,X,tol,step_size):

    
#-----------------I) AVANT LA BOUCLE WHILE-------------------------------------------------------------------------------------    
    #-----------------I.1) Initialisation--------------------------------------------------------------------

    Coordonnées_x=[]                 #Listes des coordonees x et y des points calculés par la MGDS a chaque iteration.Ca sera utile par la site pour effectuer le plot
    Coordonnées_y=[]   
    n=len(X)                         #On a choisi n==2 pour pouvoir representer graphiquement la MGDS
    grad=np.zeros(n)                 #[0,0] 
    Dict_eval={}                     #un dictionnaire pour evaluer la derivée partielle par rapport a une direction xi avec la methode evalf(subs=Dict_eval)
    alpha=step_size                  # le pas fixe de descente (pour garantire la convergence on a pris le pas relativement faible)
    x, y=symbols('x y', real=True)   #x and y sont des symbols qui aideront a calculer le dérivée partielle 
    l=[0,0]             #n==2
    
    #-------------------I.2) Choix aléatoire de i pour la direction xi-------------------------------------------------------------- 

    X0=X
    Coordonnées_x.append(X0[0])
    Coordonnées_y.append(X0[1])
    i=r.randint(0,n-1)               #le premier indice i aléatoire pour choisir la direction de descente, an d'autre terme choix aléatoire de la variable sur laquelle en effectuera la dérivée partielle df/dxi   
    
    #------------------I.3) Calcule de la dérivée partielle par rapport xi=Xk_1[i] évaluée au point X0 ------------------------

    l[i]=x
    l[int(not i)]=y
    dfi_x0=diff(f(l),x)               #l'expression symbolique de la dérivée partielle par rapport a xi=x
    Dict_eval[x]=X0[i]
    Dict_eval[y]=X0[int(not i)]
    dfi_x0=dfi_x0.evalf(subs=Dict_eval)     #La dérivée partielle evalué en X0    
    
    #-----------------I.4) Construction du gradient gradf(X0) selon la direction xi, [0 ... 0 ,df/dxi(X0), 0 ... 0] ------------------------ 

    gradfi_x0=grad.copy()           # [0         ...             0]
    gradfi_x0[i]=dfi_x0             #[0 ... 0 , df/dxi(X0) , 0 ... 0] Le vecteur gradient selon une seule direction, celle de xi calculé juste au dessus
        #                                           ^
        #                                     i ème direction
        
    #-----------------I.5) Calcule de X1------------------------------------------------------------------------------

    X1=X0-alpha*gradfi_x0
    Coordonnées_x.append(X1[0])     # pour le plot
    Coordonnées_y.append(X1[1])

    #-------I.6) Choix aléatoire de  i ---------------------------------------------------------------------
    i_1=i
    i=r.randint(0,n-1)

    #-------I.7) Calcule de la dérivée partielle par rapport xi évaluée au point X1 ------------------------  
    l[i_1]=0
    l[i]=x
    l[int(not i)]=y
    dfi_x1=diff(f(l),x)                    #l'expression symbolique de la dérivée partielle par rapport a xi=x
    
    Dict_eval[x]=X1[i]
    Dict_eval[y]=X1[int(not i)]
    dfi_x1=dfi_x1.evalf(subs=Dict_eval)    #La dérivée partielle evalué en X1  

    #-------I.8) Calcule du gradient gradf(X1) selon la direction xi = [0 ... 0 ,df/dxi(X1), 0 ... 0] ------------------------ 

    gradfi_x1=grad.copy()                  # [0         ...             0]
    gradfi_x1[i]=dfi_x1                    #[0 ... 0 , df/dxi(X1) , 0 ... 0] Le vecteur gradient selon une seule direction, celle de xi calculé juste au dessus
        #                                                  ^
        #                                          i ème direction
        

    #--------I.9) Calcule de la norme euclidienne -------------------------------------------------------------------------------------------------
    
    n_gradfi_x1=norm(gradfi_x1)
    n_distance=norm(X1-X0)

#--------II. MDGS-loop---------------------------------------------------------------------------------------------------------------
    
    num_iter=1
    while(n_gradfi_x1>tol or n_distance>tol):
        X0=X1
        gradfi_x0=gradfi_x1                     
    
        X1=X0-alpha*gradfi_x0
        
        Coordonnées_x.append(X1[0])
        Coordonnées_y.append(X1[1])
        
    #-------II.1) Choix aléatoire de i entre 0 et n-1 inclusifs ----------------- 
        i_1=i
        i=r.randint(0,n-1)
        
    #-------II.2) Calcule de la dérivée partielle par rapport xi évaluée au point X1 ------------------------ 
        l[i_1]=0
        l[i]=x
        l[int(not i)]=y
        dfi_x1=diff(f(l),x)                                #l'expression symbolique de la dérivée partielle par rapport a xi=x
        
        Dict_eval[x]=X1[i]
        Dict_eval[y]=X1[int(not i)]
        
        dfi_x1=dfi_x1.evalf(subs=Dict_eval)                #La dérivée partielle evalué en X1
        
    #-------II.3) Calcule du gradient gradf(X1) selon la direction xi = [0 ... 0 ,df/dxi(X1), 0 ... 0] ------------------------   

        gradfi_x1=grad.copy()
        gradfi_x1[i]=dfi_x1
        
    #--------II.4) Calcule de la norme euclidienne-------------------------------------------------------------------------------------------------        
        
        n_gradfi_x1= norm(gradfi_x1)
        n_distance = norm(X1-X0)
        
        num_iter+=1
        #--------Fin de la boucle-------------------------------------------------------------------------------------------------                
    w=tol
    p=0
    while(w<1):
        w*=10
        p+=1
    X1=np.round_(X1,decimals=p)   
    return X1

#-------------------------------------------------------------------------------------------------------------------------------

def _sgdbls_(f,X,tol,initial_step_size,c):

    
    
#-----------------I) AVANT LA BOUCLE WHILE-------------------------------------------------------------------------------------    
    #-----------------I.1) Initialisation--------------------------------------------------------------------

    Coordonnées_x=[]    #Listes des coordonees x et y des points calculés par la MGDS a chaque iteration.Ca sera utile par la site pour effectuer le plot
    Coordonnées_y=[]
    n=len(X)            #On a choisi n==2 pour pouvoir representer graphiquement la MGDS
    grad=np.zeros(n)    #[0,0]
    Dict_eval={}        #un dictionnaire pour evaluer la derivée partielle par rapport a une direction xi avec la methode evalf(subs=Dict_eval)
    x, y=symbols('x y', real=True) #x and y sont des symbols qui aideront a calculer le dérivée partielle
    l=[0,0]             #n==2
    
    #-------------------I.2) Choix aléatoire de i pour la direction xi-------------------------------------------------------------- 

    X0=X
    Coordonnées_x.append(X0[0])
    Coordonnées_y.append(X0[1])
    i=r.randint(0,n-1)   #le premier indice i aléatoire pour choisir la direction de descente, 
                    # -> en d'autre terme c'est le choix aléatoire de la variable sur laquelle en effectuera la dérivée partielle df/dxi   
    
    #------------------I.3) Calcule de la dérivée partielle par rapport xi=Xk_1[i] évaluée au point X0 ------------------------
    
    l[i]=x
    l[int(not i)]=y
    dfi_x0=diff(f(l),x)               #l'expression symbolique de la dérivée partielle par rapport a xi=x
    
    Dict_eval[x]=X0[i]
    Dict_eval[y]=X0[int(not i)]
    
    dfi_x0=dfi_x0.evalf(subs=Dict_eval)     #La dérivée partielle evalué en X0     
    
    #-----------------I.4) Construction du gradient gradf(X0) selon la direction xi, [0 ... 0 ,df/dxi(X0), 0 ... 0] ------------------------ 

    gradfi_x0=grad.copy()           # [0         ...             0]
    gradfi_x0[i]=dfi_x0             #[0 ... 0 , df/dxi(X1) , 0 ... 0] Le vecteur gradient selon une seule direction, celle de xi calculé juste au dessus
        #                                                  ^
        #                                          i ème direction
        
#_____________________________________________"Backtracking line search using Armijo condition"_________________________________
    
    def phi(alpha):
        xalpha=X0-alpha*gradfi_x0                    #la fonction à minimiser
        return f(xalpha)
    
    alpha=initial_step_size                    #le pas initiale = 100 assez grand pour ne pas satisfaire la condition d'armijo dès le debut 
    eps=tol                                    #espilone = tolérence
    N=c                                       # A chaque fois que la condition d'Armijo n'est pas vérifié On divise le pas par N 


#                                                                phi(0 + epsilone_très_faible) - phi(0)
    phi_prime_de_0 = (phi(1.e-6)-phi(0))/1.e-6   #phi'(0) = ----------------------------------------
#                                                                        epsilone_très_faible                
    
    
    while(phi(alpha) > phi(0)+eps*alpha*phi_prime_de_0):      #tant que la condition d'Armijo n'est pas vérifié on divise le pas par N
        alpha=alpha/N
#__________________________________________________________________________________________________________________________  

    #---------I.5) Calcule de X1------------------------------------------------------------------------------

    X1=X0-alpha*gradfi_x0
    
    Coordonnées_x.append(X1[0])         # pour le plot
    Coordonnées_y.append(X1[1])

    #-------I.6) Choix aléatoire de  i ---------------------------------------------------------------------
    i_1=i
    i=r.randint(0,n-1)

    #-------I.7) Calcule de la dérivée partielle par rapport xi évaluée au point X1 ------------------------  

    l[i_1]=0
    l[i]=x
    l[int(not i)]=y
    dfi_x1=diff(f(l),x)                        #l'expression symbolique de la dérivée partielle par rapport a xi=x
    
    Dict_eval[x]=X1[i]
    Dict_eval[y]=X1[int(not i)]
    
    dfi_x1=dfi_x1.evalf(subs=Dict_eval)        #La dérivée partielle evalué en X1  

    #-------I.8) Calcule du gradient gradf(X1) selon la direction xi = [0 ... 0 ,df/dxi(X1), 0 ... 0] ------------------------ 

    gradfi_x1=grad.copy()                 # [0         ...             0]
    gradfi_x1[i]=dfi_x1                   #[0 ... 0 , df/dxi(X1) , 0 ... 0] Le vecteur gradient selon une seule direction, celle de xi calculé juste au dessus
        #                                                  ^
        #                                          i ème direction

    #--------I.9) Calcule de la norme euclidienne -------------------------------------------------------------------------------------------------
    
    n_gradfi_x1=norm(gradfi_x1)
    n_distance=norm(X1-X0)

#--------SGD-loop---------------------------------------------------------------------------------------------------------------

    num_iter=1
    while(n_gradfi_x1>tol or n_distance>tol):
        X0=X1
        gradfi_x0=gradfi_x1
        i_1=i
#_______________________________"Backtracking line search using Armijo condition"______________________________________________
    
        def phi(alpha):
            xalpha=X0-alpha*gradfi_x0              #la fonction à minimiser
            return f(xalpha)

        alpha=initial_step_size                                  #le pas initiale = 100 assez grand pour ne pas satisfaire la condition d'armijo dès le debut 
        eps=tol                                    #espilone = tolérence
        N=c                                        # A chaque fois que la condition d'Armijo n'est pas vérifié On divise le pas par N 


    #                      phi(0 + epsilone_très_faible) - phi(0)
    #           phi'(0) = ----------------------------------------
    #                               epsilone_très_faible
    
        phi_prime_de_0=(phi(1.e-6)-phi(0))/1.e-6

        while(phi(alpha) > phi(0)+eps*alpha*phi_prime_de_0):        #tant que la condition d'Armijo n'est pas vérifié on divise le pas par N
            alpha=alpha/N
#__________________________________________________________________________________________________________________________________

        X1=X0-alpha*gradfi_x0
        
        Coordonnées_x.append(X1[0])
        Coordonnées_y.append(X1[1])
        
        #-------II.1) Choix aléatoire de i entre 0 et n-1 inclusifs ----------------- 
        
        i=r.randint(0,n-1)
        
         #-------II.2) Calcule de la dérivée partielle par rapport xi évaluée au point X1------------------------ 
        
        l[i_1]=0
        l[i]=x
        l[int(not i)]=y
        dfi_x1=diff(f(l),x)
        
        Dict_eval[x]=X1[i]
        Dict_eval[y]=X1[int(not i)]
        
        dfi_x1=dfi_x1.evalf(subs=Dict_eval)
        
        #-------II.3) Calcule du gradient gradf(X1) selon la direction xi = [0 ... 0 ,df/dxi(X1), 0 ... 0] ------------------------ 

        gradfi_x1=grad.copy()
        gradfi_x1[i]=dfi_x1
        
        #--------II.4) Calcule de la norme euclidienne-------------------------------------------------------------------------------------------------        
        
        n_gradfi_x1= norm(gradfi_x1)
        n_distance = norm(X1-X0)
        num_iter+=1
        
        #--------Fin de la boucle-------------------------------------------------------------------------------------------------                
    w=tol
    p=0
    while(w<1):
        w*=10
        p+=1
    X1=np.round_(X1,decimals=p)
    return X1


