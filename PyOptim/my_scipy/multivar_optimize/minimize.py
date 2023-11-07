import numpy as np                        #Manipulation des Matrices
from numpy.linalg import norm             #Calcule de la norme euclidienne comme critère d'arrêt du programme      
from sympy import diff, symbols           #Calcule symbolique de la dérivée partielle de f ou g afin de calculer le gradient stochastique selon une direction donnée 
import random as r
import scipy.optimize as spo
import math as m
import numdifftools as nd  #Pour calculer le gradient d'une fonction multivariable pas d'une liste  nd.Gradient(f)(Xk_1)


#-------------------------------------------------------------------------------------------------------------------------------
def gradient_descent(f,X,tol=0.01,alpha=0.01):
    """+ gradient_descent function takes 4 arguments:
- f : a unimodal function that takes a numpy array as an argument
- X : a starting numpy array
- tol : the tolerence
- alpha : the fixed step size
       + It returns the argmin(f):  a numpy array  """
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
def gradient_conjugate(f,X,tol=0.01):
    """+ gradient_conjugate function takes 3 arguments:
- f : a unimodal function that takes  a column vector as an argument
       !!! *the Hessian must be positive definite  !!! 
- X : a starting column vector
- tol : the tolerence
       + It returns the argmin(f): column vector  """
    X0=np.array(X).T
    d0=np.array([-nd.Gradient(f)(X0)]).T
    Q=nd.Gradient(nd.Gradient(f))(X0)
    n=len(X0)
    for k in range(0,n-1):
        a=(d0.T@d0)[0,0]
        b=(d0.T@Q@d0)[0,0]
        alpha=a/b
        X0=X0+alpha*d0
        beta=(np.array([nd.Gradient(f)(X0)])@Q@d0)/(d0.T@Q@d0)
        d0=np.array([-nd.Gradient(f)(X0)]).T + beta*d0
        if(d0.all()==0):
            break
    w=tol
    p=0
    while(w<1):
        w*=10
        p+=1
    X0=np.round_(X0,decimals=p)
    return X0  
#-------------------------------------------------------------------------------------------------------------------------------
def newton(f,X,tol=0.01):
    """+ newton function takes 3 arguments:
- f : a unimodal function that takes  a numpy array as an argument
       *the Hessian must be invertible
- X : a starting numpy array
- tol : the tolerence
       + It returns the argmin(f): numpy array  """
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
    
def quasi_newton_dfp(f,X,tol=0.01):
    """+ quasi_newton_dfp takes 3 arguments:
- f : a unimodal function that takes  a numpy array as an argument
- X : a starting numpy array
- tol : the tolerence
       + It returns the argmin(f): numpy array  """
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

def sgd_2var(f,X,tol=0.01,step_size=0.01):
    """+ stochatstic_gradient_descent_2var takes 4 arguments:
- f : a unimodal function that takes a numpy array as an argument
*Important the column vector must be of size 2
- X : a starting numpy array of size 2
- tol : the tolerence
- step_size : de depth of de descent, it's a real value. 
       + It returns the argmin(f): a numpy array"""
    
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

def sgd_with_bls_2var(f,X,tol=0.01,initial_step_size=0.01,c=2):
    """+ SGD_with_BLS_2var takes 5 arguments:
- f : a unimodal function that takes a numpy array an argument
*Important: the numpy array must be of size 2
- X : a starting column vector of size 2
- tol : the tolerence
- initial_step_size : de depth of the descent, it's a real value.
- c: Armijo dividing coefficient c>1
       + It returns the argmin(f): a numpy array of size 2"""
    
    
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

