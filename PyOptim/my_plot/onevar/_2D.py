import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
#----------------1a Unrestricted_fixed_step_search----------------------------------------------------------------------------------------------


def fixed_step(f,x0,epsilon=0.01):
    """+ fixed_step function takes 3 arguments:
- f : a one variable & unimodal function 
- x0 : the initial starting point
- epsilon : which is the target precision
       + It returns the argmin(f) & the 2D plot of the search
       * the step size is fixed at epsilon """
    X=x0
    Coordonnées_x=[x0]
    step=epsilon
    x1=x0+step
    x_1=x0-step
    f1=f(x1)
    f0=f(x0)
    f_1=f(x_1)
    if(f1<f0):
        Coordonnées_x.append(x1)
        while(f1<f0):
            x0=x1
            x1+=step
            Coordonnées_x.append(x1)
            f1=f(x1)
            f0=f(x0)
        result = (x0 + x1)/2  #here xk==x0 et xk+1==x1 and |xk - xk+1|==step==epsilon
    #so we can stop searching and choose any x in between xk and xk+1, I chosed the x in middle.
    
    elif(f_1<f0):
        Coordonnées_x.append(x_1)
        while(f_1<f0):    
            x0=x_1
            x_1=x0-step
            Coordonnées_x.append(x_1)
            f0=f(x0)
            f_1=f(x_1)
        result = (x0 + x_1)/2 #here xk==x0 et xk+1==x1 and |xk - xk+1|==step==epsilon
    #so we can stop searching and choose any x in between xk and xk+1, I chosed the x in middle.
    
    else:
        Coordonnées_x.append(x1)
        result = (x0 + x1)/2   #here xk==x0 et xk+1==x1 and |xk - xk+1|==step==epsilon
    #so we can directly choose any x in between xk and xk+1, I chosed the x in middle.    
    
    print('x* = ',result)
    print('Le nombre de points parcourus avant d\'arriver au minimum est ',len(Coordonnées_x))
    print('Le plot apparaitra dans quelques secondes, merci de patienter !')
    fig, ax = plt.subplots(figsize=(10,10))

    theta1_grid = np.linspace(-10-abs(X),10+abs(X),50)
    J_grid=[]
    for i in range(len(theta1_grid)):
        J_grid.append(f(theta1_grid[i]))

    ax.plot(theta1_grid, J_grid, 'k')


    for j in range(1,len(Coordonnées_x)):
        ax.annotate('', xy=(Coordonnées_x[j], f(Coordonnées_x[j])), xytext=(Coordonnées_x[j-1],f(Coordonnées_x[j-1])), arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1} , va='center', ha='center')
    # Labels, titles and a legend.
    ax.scatter(Coordonnées_x, f(np.array(Coordonnées_x)), c='b', s=40, lw=1)
    ax.set_xlim(-5-abs(X),abs(X)+5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('f(x)')

    plt.tight_layout()
    plt.show()


#====================================================================================================================================================
#====================================================================================================================================================

#--------------1b Unrestricted_accelerated_step_search------------------------------------------------------------------------------------------
def accelerated_step(f,x0,epsilon=0.01):
    """+ accelerated_step function takes 3 arguments:
- f : a one variable & unimodal function 
- x0 : the initial starting point
- epsilon : which is the target precision
       + It returns the argmin(f) & the 2D plot of the search
       * the step size is initialized at epsilon """
    X=x0
    Coordonnées_x=[x0]
    init_step=epsilon
    step=init_step
    x1=x0+step
    x_1=x0-step
    f1=f(x1)
    f0=f(x0)
    f_1=f(x_1)
    if(f1<f0):
        Coordonnées_x.append(x1)
        while(1):
            while(f1<f0):
                x0=x1
                step*=2
                x1=x0+step
                Coordonnées_x.append(x1)
                f1=f(x1)
                f0=f(x0)
            if(step==epsilon):
                result = (x0 + x1)/2  #here we found xk==x0 et xk+1==x1 such as |xk - xk+1|==epsilon
                break            #so we can stop searching and choose any x in between xk and xk+1, I chosed the x in middle.
            step=init_step #we reduce the step length after bracketing the minimum
            x1=x0+step
            f1=f(x1)
    elif(f1>f0):
        Coordonnées_x.append(x_1)
        while(1):
            while(f_1<f0):
                x0=x_1
                step*=2
                x_1=x0-step
                Coordonnées_x.append(x_1)
                f0=f(x0)
                f_1=f(x_1)
            if(step==epsilon):
                result = (x0 + (x_1))/2 #here we found xk==x0 et xk+1==x1 such as |xk - xk+1|==epsilon
                break
            #so we can stop searching and choose any x in between xk and xk+1, I chosed the x in middle.
            step=init_step #we reduce the step length after bracketing the minimum
            x_1=x0-step
    else:
        Coordonnées_x.append(x1)
        result = (x0 + x1)/2   #here xk==x0 et xk+1==x1 and |xk - xk+1|==init_step==epsilon
    #so we can stop searching and choose any x in between xk and xk+1, I chosed the x in middle.
    print('x* = ',result)
    print('Le nombre de points parcourus avant d\'arriver au minimum est ',len(Coordonnées_x))
    print('Le plot apparaitra dans quelques secondes, merci de patienter !')
    fig, ax = plt.subplots(figsize=(10,10))

    theta1_grid = np.linspace(-10-abs(X),10+abs(X),50)
    J_grid=[]
    for i in range(len(theta1_grid)):
        J_grid.append(f(theta1_grid[i]))

    ax.plot(theta1_grid, J_grid, 'k')


    for j in range(1,len(Coordonnées_x)):
        ax.annotate('', xy=(Coordonnées_x[j], f(Coordonnées_x[j])), xytext=(Coordonnées_x[j-1],f(Coordonnées_x[j-1])), arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1} , va='center', ha='center')
    # Labels, titles and a legend.
    ax.scatter(Coordonnées_x, f(np.array(Coordonnées_x)), c='b', s=40, lw=1)
    ax.set_xlim(-5-abs(X),abs(X)+5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('f(x)')

    plt.tight_layout()
    plt.show()


#====================================================================================================================================================
#====================================================================================================================================================
#----------------1c Exhaustive search method--------------------------------------------------------------------------------------
def exhaustive_search(f,xs,xf,epsilon=0.01):
    """+ The exhaustive_search function takes 4 arguments:
- f : a one variable & unimodal function 
- xs : the starting point
- xf : the finishing point
- epsilon : which is the target precision
       + It returns the argmin(f) & the 2D plot of the search
       * the step size is fixed at epsilon """
    Coordonnées_x=[xs]
    X=xs
    step=epsilon
    num_of_equ_spaced_points=((xf-xs)/step)+1
    num_of_equ_spaced_points=int(num_of_equ_spaced_points)
    list_of_evaluations=[]
    xi=-10000
    if(f(xs)>f(xs+step)):
        for i in range(0,num_of_equ_spaced_points):
            x=xs+i*step
            Coordonnées_x.append(x)
            list_of_evaluations.append(f(x))
        min=list_of_evaluations[0]
        p=0
        for i in range(0,num_of_equ_spaced_points):
            if(list_of_evaluations[i]<=min):
                min=list_of_evaluations[i]
                p=i
        xi=xs+p*step
        xi_1=xs+(p-1)*step
        xi1=xs+(p+1)*step
        #the final interval of uncertitude is [xi-1,xi+1] of length 2*step which is epsilon so i chosed the middle of this interval for instance xi as a minimum  
    elif(f(xs)==f(xs+step)):
        xi=(xs+xf)/2
    Coordonnées_x.append(xi)
    result = xi
    print('x* = ',result)
    print('Le nombre de points parcourus avant d\'arriver au minimum est ', len(Coordonnées_x))
    print('Le plot apparaitra dans quelques secondes, merci de patienter !')
    fig, ax = plt.subplots(figsize=(10,10))
    theta1_grid = np.linspace(-10-abs(X),10+abs(X),50)
    J_grid=[]
    for i in range(len(theta1_grid)):
        J_grid.append(f(theta1_grid[i]))

    ax.plot(theta1_grid, J_grid, 'k')


    for j in range(1,len(Coordonnées_x)):
        ax.annotate('', xy=(Coordonnées_x[j], f(Coordonnées_x[j])), xytext=(Coordonnées_x[j-1],f(Coordonnées_x[j-1])), arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1} , va='center', ha='center')
    # Labels, titles and a legend.
    ax.scatter(Coordonnées_x, f(np.array(Coordonnées_x)), c='b', s=40, lw=1)
    ax.set_xlim(-5-abs(X),abs(X)+5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('f(x)')

    plt.tight_layout()
    plt.show()


#====================================================================================================================================================
#====================================================================================================================================================
#-----------------1d Dichotomous search method------------------------------------------------------------------------------------
def dichotomous_search(f,xs,xf,epsilon=0.01,mini_delta=0.001):
    """ + The dichotomous_search function takes 5 arguments:
    - f : a one variable & unimodal function 
    - xs : the starting point
    - xf : the finishing point
    - epsilon : which is the target precision
    - mini_delta : determinate the size of x_middle's neighborhood
            + It returns the argmin(f) & the 2D plot of the search
            ! mini_delta must be way smaller than epsilon """
    Coordonnées_x=[xs]
    X=xs
    def ds(f,xs,xf,epsilon,mini_delta):
        x_middle=(xf+xs)/2
        Coordonnées_x.append(x_middle)
        if(epsilon <= mini_delta):
            print(" mini_delta must be way smaller than epsilon !")
        elif((xf-xs)<=epsilon):
            x_min = x_middle
            return x_min
        else:
            x1= x_middle - (mini_delta/2)
            x2= x_middle + (mini_delta/2)
            f1=f(x1)
            f2=f(x2)
            if(f1<f2):
                xf=x2
            elif(f1>f2):
                xs=x1
            else:
                xs=x1
                xf=x2
            return ds(f,xs,xf,epsilon,mini_delta)
    result = ds(f,xs,xf,epsilon,mini_delta)
    print('x* = ',result)
    print('Le nombre de points parcourus avant d\'arriver au minimum est ', len(Coordonnées_x))
    print('Le plot apparaitra dans quelques secondes, merci de patienter !')
    fig, ax = plt.subplots(figsize=(10,10))
    theta1_grid = np.linspace(-10-abs(X),10+abs(X),50)
    J_grid=[]
    for i in range(len(theta1_grid)):
        J_grid.append(f(theta1_grid[i]))

    ax.plot(theta1_grid, J_grid, 'k')


    for j in range(1,len(Coordonnées_x)):
        ax.annotate('', xy=(Coordonnées_x[j], f(Coordonnées_x[j])), xytext=(Coordonnées_x[j-1],f(Coordonnées_x[j-1])), arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1} , va='center', ha='center')
    # Labels, titles and a legend.
    ax.scatter(Coordonnées_x, f(np.array(Coordonnées_x)), c='b', s=40, lw=1)
    ax.set_xlim(-5-abs(X),abs(X)+5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('f(x)')

    plt.tight_layout()
    plt.show()


#====================================================================================================================================================
#====================================================================================================================================================
    
#------------1e Interval halving method----------------------------------------------------------------------------------------------
def interval_halving(f,a,b,epsilon=0.01):
    """ + The interval_halving function takes 4 arguments:
- f : a one variable & unimodal function f
- a : the starting point
- b : the finishing point
- epsilon : which is the target precision
        + It returns the argmin(f) & the 2D plot of the search """
    Coordonnées_x=[a]
    X=a
    L=b-a
    step=L/4
    x1=a+step
    x0=a+2*step
    x2=a+3*step
    f1=f(x1)
    f2=f(x2)
    f0=f(x0)
    if(f1<f0 and f0<f2):
        b=x0
        x0=x1
    elif(f1>f0 and f0>f2): 
        a=x0
        x0=x2
    elif(f1>f0 and f0<f2):
        a=x1
        b=x2
    Coordonnées_x.append((a+b)/2)    
    L=b-a
    while(L>=2*epsilon):
        step=L/4
        x1=a+step
        x2=a+3*step
        f1=f(x1)
        f2=f(x2)
        f0=f(x0)
        if(f1<=f0 and f0<=f2):
            b=x0
            x0=x1
        elif(f1>=f0 and f0>=f2):
            a=x0
            x0=x2
        elif(f1>=f0 and f0<=f2):
            a=x1
            b=x2
        Coordonnées_x.append((a+b)/2)
        L=b-a
    x_min=(b+a)/2
    result = x_min

    print('x* = ',result)
    print('Le nombre de points parcourus avant d\'arriver au minimum est ', len(Coordonnées_x))
    print('Le plot apparaitra dans quelques secondes, merci de patienter !')
    fig, ax = plt.subplots(figsize=(10,10))
    theta1_grid = np.linspace(-10-abs(X),10+abs(X),50)
    J_grid=[]
    for i in range(len(theta1_grid)):
        J_grid.append(f(theta1_grid[i]))

    ax.plot(theta1_grid, J_grid, 'k')


    for j in range(1,len(Coordonnées_x)):
        ax.annotate('', xy=(Coordonnées_x[j], f(Coordonnées_x[j])), xytext=(Coordonnées_x[j-1],f(Coordonnées_x[j-1])), arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1} , va='center', ha='center')
    # Labels, titles and a legend.
    ax.scatter(Coordonnées_x, f(np.array(Coordonnées_x)), c='b', s=40, lw=1)
    ax.set_xlim(-5-abs(X),abs(X)+5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('f(x)')

    plt.tight_layout()
    plt.show()


#====================================================================================================================================================
#====================================================================================================================================================

#----------------1f  Fibonacci method-------------------------------------------------------------------------------------
def fibonacci_sequence(n):
    un_1=un_2=1
    if(n<=1):
        return un_1
    else:
        i=2
        while(i<=n):
            un=un_1+un_2
            un_2=un_1
            un_1=un
            i+=1
        return un

def fibonacci(f,a,b,n=15):
    """ + The fibonacci function takes 4 arguments:
- f : a one variable & unimodal function f
- a : the starting point
- b : the finishing point
- n : the number of iterations to perform 
        + It returns the argmin(f) & the 2D plot of the search """
    X0=a
    X1=b
    X=(a+b)/2
    Coordonnées_x=[(a+b)/2]
    def d(f,a,b,n):
        while(1):
            L=b-a
            L_ét=(fibonacci_sequence(n-2)/fibonacci_sequence(n))*L
            c=a+L_ét
            d=b-L_ét
            fc=f(c)
            fd=f(d)
            if(fc<fd):
                b=d
            elif(fc>fd):
                a=c
            else:
                a=c
                b=d
            Coordonnées_x.append((a+b)/2)
            if((d-c)<0.000001):
               return (a+b)/2 
            n-=1
    result =  d(f,X0,X1,n)       
    print('x* = ',result)
    print('Le nombre de points parcourus avant d\'arriver au minimum est ', len(Coordonnées_x))
    print('Le plot apparaitra dans quelques secondes, merci de patienter !')
    fig, ax = plt.subplots(figsize=(10,10))
    theta1_grid = np.linspace(-10-abs(X),10+abs(X),50)
    J_grid=[]
    for i in range(len(theta1_grid)):
        J_grid.append(f(theta1_grid[i]))

    ax.plot(theta1_grid, J_grid, 'k')


    for j in range(1,len(Coordonnées_x)):
        ax.annotate('', xy=(Coordonnées_x[j], f(Coordonnées_x[j])), xytext=(Coordonnées_x[j-1],f(Coordonnées_x[j-1])), arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1} , va='center', ha='center')
    # Labels, titles and a legend.
    ax.scatter(Coordonnées_x, f(np.array(Coordonnées_x)), c='b', s=40, lw=1)
    ax.set_xlim(-5-abs(X),abs(X)+5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('f(x)')

    plt.tight_layout()
    plt.show()


#====================================================================================================================================================
#==================================================================================================================================================== 
#------------------1g Golden section method---------------------------------------------------------------------------------------

def golden_section(f,a,b,epsilon=0.01):
    """ + The golden_section function takes 4 arguments:
- f : a one variable & unimodal function f
- a : the starting point
- b : the finishing point
- epsilon : which is the target precision
        + It returns the argmin(f) & the 2D plot of the search """
    X=(a+b)/2
    Coordonnées_x = [(a+b)/2] 
    gamma=0.618033
    while(1):
        L=b-a
        L_ét=gamma*L
        c=a+L_ét
        d=b-L_ét
        fc=f(c)
        fd=f(d)
        if(fc<fd):
            a=d
        elif(fc>fd):
            b=c
        else:
            a=d
            b=c
        Coordonnées_x.append((a+b)/2)
        if((b-a)<epsilon):
           result = (a+b)/2
           break
        
    print('x* = ',result)
    print('Le nombre de points parcourus avant d\'arriver au minimum est ', len(Coordonnées_x))
    print('Le plot apparaitra dans quelques secondes, merci de patienter !')
    fig, ax = plt.subplots(figsize=(10,10))
    theta1_grid = np.linspace(-10-abs(X),10+abs(X),50)
    J_grid=[]
    for i in range(len(theta1_grid)):
        J_grid.append(f(theta1_grid[i]))

    ax.plot(theta1_grid, J_grid, 'k')


    for j in range(1,len(Coordonnées_x)):
        ax.annotate('', xy=(Coordonnées_x[j], f(Coordonnées_x[j])), xytext=(Coordonnées_x[j-1],f(Coordonnées_x[j-1])), arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1} , va='center', ha='center')
    # Labels, titles and a legend.
    ax.scatter(Coordonnées_x, f(np.array(Coordonnées_x)), c='b', s=40, lw=1)
    ax.set_xlim(-5-abs(X),abs(X)+5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('f(x)')

    plt.tight_layout()
    plt.show()


#====================================================================================================================================================
#====================================================================================================================================================     
#------------------Armijo methods---------------------------------------------------------------------------------------

def armijo_backward(f,x0,ŋ=2,epsilon=0.01):
    """ + The armijo_backward (Backtracking_line_search) function takes 4 arguments:
- f : a one variable & unimodal function f
- x0 : the starting point
- ŋ : the coefficient by which we divide x0 at each iteration
- epsilon : which is the target precision
        + It returns the argmin(f) & the 2D plot of the search """
    X=x0
    Coordonnées_x = [x0]
    a=x0
    f_prime_de_0 = ( f(1.e-6)-f(0) ) / 1.e-6
    
    def f_chapeau(a):
            return f(0) + epsilon*a*f_prime_de_0
    i=0   
    if(f(a)<=f_chapeau(a)):
        print("Choisissez un x0 plus grand qui ne vérifie pas la condition d'Armijo pour votre f et epsilone  !")
    else:
        while(f(a)>f_chapeau(a)):
            i+=1
            a=a/ŋ
            Coordonnées_x.append(a)
        result = a
        
    if(i>0):
        print('x* = ',result)
        print('Le nombre de points parcourus avant d\'arriver au minimum est ', len(Coordonnées_x))
        print('Le plot apparaitra dans quelques secondes, merci de patienter !')
        fig, ax = plt.subplots(figsize=(10,10))
        theta1_grid = np.linspace(-10-abs(X),10+abs(X),50)
        J_grid=[]
        for i in range(len(theta1_grid)):
            J_grid.append(f(theta1_grid[i]))

        ax.plot(theta1_grid, J_grid, 'k')


        for j in range(1,len(Coordonnées_x)):
            ax.annotate('', xy=(Coordonnées_x[j], f(Coordonnées_x[j])), xytext=(Coordonnées_x[j-1],f(Coordonnées_x[j-1])), arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1} , va='center', ha='center')
        # Labels, titles and a legend.
        ax.scatter(Coordonnées_x, f(np.array(Coordonnées_x)), c='b', s=40, lw=1)
        ax.set_xlim(-5-abs(X),abs(X)+5)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title('f(x)')

        plt.tight_layout()
        plt.show()


#====================================================================================================================================================
#====================================================================================================================================================  

def armijo_forward(f,x0,ŋ=2,epsilon=0.01):
    """ + The armijo_forward function takes 4 arguments:
- f : a one variable & unimodal function f
- x0 : the starting point
- ŋ : the coefficient by which we multiply x0 at each iteration
- epsilon : which is the target precision
        + It returns the argmin(f) & the 2D plot of the search """
    Coordonnées_x = [x0]
    X=x0
    a=x0
    f_prime_de_0 = ( f(1.e-6)-f(0) ) / 1.e-6
    def f_chapeau(a):
            return f(0) + epsilon*a*f_prime_de_0
    i=0    
    if(f(a)>f_chapeau(a)):
        print("Choisissez un x0 plus petit qui vérifie la condition d'Armijo pour votre f et epsilone !")
    else:
        while(f(a)<=f_chapeau(a)):
            i+=1
            a=a*ŋ
            Coordonnées_x.append(a)
        Coordonnées_x.append(a/ŋ)
        result = a/ŋ

    if(i>0):
        print('x* = ',result)
        print('Le nombre de points parcourus avant d\'arriver au minimum est ', len(Coordonnées_x))
        print('Le plot apparaitra dans quelques secondes, merci de patienter !')
        fig, ax = plt.subplots(figsize=(10,10))
        theta1_grid = np.linspace(-10-abs(X),10+abs(X),50)
        J_grid=[]
        for i in range(len(theta1_grid)):
            J_grid.append(f(theta1_grid[i]))

        ax.plot(theta1_grid, J_grid, 'k')


        for j in range(1,len(Coordonnées_x)):
            ax.annotate('', xy=(Coordonnées_x[j], f(Coordonnées_x[j])), xytext=(Coordonnées_x[j-1],f(Coordonnées_x[j-1])), arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1} , va='center', ha='center')
        # Labels, titles and a legend.
        ax.scatter(Coordonnées_x, f(np.array(Coordonnées_x)), c='b', s=40, lw=1)
        ax.set_xlim(-5-abs(X),abs(X)+5)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title('f(x)')

        plt.tight_layout()
        plt.show()


#====================================================================================================================================================
#====================================================================================================================================================

def compare_all_time(f,xs,xf,epsilon,mini_delta_dichotomous,n_fibo,ŋ_armijo,xs_armijo_forward,xs_armijo_backward):
    T=[0 for i in range(10)]
    P=[0 for i in range(9)]
    T[0]=perf_counter()
    P[0]=_fs_(f,xs,epsilon)
    
    T[1]=perf_counter()
    P[1]=_as_(f,xs,epsilon)
    
    T[2]=perf_counter()
    P[2]=_es_(f,xs,xf,epsilon)
    
    T[3]=perf_counter()
    P[3]=_ds_(f,xs,xf,epsilon,mini_delta_dichotomous)
    
    T[4]=perf_counter()
    P[4]=_ih_(f,xs,xf,epsilon)
    
    T[5]=perf_counter()
    P[5]=_fi_(f,xs,xf,n_fibo)
    
    T[6]=perf_counter()
    P[6]=_gs_(f,xs,xf,epsilon)
    
    T[7]=perf_counter()
    P[7]=_ab_(f,xs_armijo_backward,ŋ_armijo,epsilon)
    
    T[8]=perf_counter()
    P[8]=_af_(f,xs_armijo_forward,ŋ_armijo,epsilon)
    
    T[9]=perf_counter()
    
    Optimization_Methods=["fixed_step","accelerated_step","exhaustive_search","dichotomous_search","interval_halving","fibonacci","golden_section","armijo_backward","armijo_forward"]
    Time=[T[1]-T[0], T[2]-T[1], T[3]-T[2], T[4]-T[3], T[5]-T[4], T[6]-T[5], T[7]-T[6],T[8]-T[7], T[9]-T[8]]
    plt.style.use('ggplot')
    plt.figure(figsize=(12,7))
    plt.barh(Optimization_Methods, Time)
    plt.title('Comparaison Temporelle')
    plt.ylabel("Les Méthodes d'Optimisations à une Variable et Sans Contrainte" )
    plt.xlabel('Temps en (s)')
    plt.show()

#====================================================================================================================================================
#====================================================================================================================================================
import scipy.optimize as spo
def compare_all_precision(f,xs,xf,epsilon,mini_delta_dichotomous, n_fibo, ŋ_armijo ,xs_armijo_forward,xs_armijo_backward):

    P=[0 for i in range(9)]
    P[0]=_fs_(f,xs,epsilon)
    P[1]=_as_(f,xs,epsilon)
    P[2]=_es_(f,xs,xf,epsilon)
    P[3]=_ds_(f,xs,xf,epsilon,mini_delta_dichotomous)
    P[4]=_ih_(f,xs,xf,epsilon)
    P[5]=_fi_(f,xs,xf,n_fibo)
    P[6]=_gs_(f,xs,xf,epsilon)
    P[7]=_ab_(f,xs_armijo_backward,ŋ_armijo,epsilon)
    P[8]=_af_(f,xs_armijo_forward,ŋ_armijo,epsilon)
    Xopt=spo.minimize(f,xs)

    Optimization_Methods=["fixed_step","accelerated_step","exhaustive_search","dichotomous_search","interval_halving","fibonacci","golden_section","armijo_backward","armijo_forward"]
    Precision_error=[abs((P[i]-Xopt.x)[0]) for i in range(9)]
    plt.style.use('ggplot')
    plt.figure(figsize=(12,7))
    plt.barh(Optimization_Methods, Precision_error)
    plt.title("Comparaison des erreurs")
    plt.ylabel("Les Méthodes d'Optimisations à une Variables et Sans Contraintes" )
    plt.xlabel("les écarts entre la valeur recherché et les valeurs trouvés")
    plt.show()












#----------------1a Unrestricted_fixed_step_search----------------------------------------------------------------------------------------------

def _fs_(function,x0,epsilon):
    step=epsilon
    x1=x0+step
    x_1=x0-step
    f1=function(x1)
    f0=function(x0)
    f_1=function(x_1)
    if(f1<f0):
        while(f1<f0):
            x0=x1
            x1+=step
            f1=function(x1)
            f0=function(x0)
        result = (x0 + x1)/2
    
    elif(f_1<f0):
        while(f_1<f0):    
            x0=x_1
            x_1=x0-step
            f0=function(x0)
            f_1=function(x_1)
        result = (x0 + x_1)/2 
    
    else:
        result = (x0 + x1)/2
    w=epsilon
    p=0
    while(w<1):
        w*=10
        p+=1
    
    return round(result,p)
#--------------1b Unrestricted_accelerated_step_search------------------------------------------------------------------------------------------
def _as_(function,x0,epsilon):
    init_step=epsilon
    step=init_step
    x1=x0+step
    x_1=x0-step
    f1=function(x1)
    f0=function(x0)
    f_1=function(x_1)
    if(f1<f0):
        while(1):
            while(f1<f0):
                x0=x1
                step*=2
                x1=x0+step
                f1=function(x1)
                f0=function(x0)
            if(step==epsilon):
                result = (x0 + x1)/2  #here we found xk==x0 et xk+1==x1 such as |xk - xk+1|==epsilon
                break
            #so we can stop searching and choose any x in between xk and xk+1, I chosed the x in middle.
            step=init_step #we reduce the step length after bracketing the minimum
            x1=x0+step
            f1=function(x1)
    elif(f1>f0):
        while(1):
            while(f_1<f0):
                x0=x_1
                step*=2
                x_1=x0-step
                f0=function(x0)
                f_1=function(x_1)
            if(step==epsilon):
                result = (x0 + (x_1))/2 #here we found xk==x0 et xk+1==x1 such as |xk - xk+1|==epsilon
                break
            #so we can stop searching and choose any x in between xk and xk+1, I chosed the x in middle.
            step = init_step #we reduce the step length after bracketing the minimum
            x_1 = x0-step
    else:
        result = (x0 + x1)/2   #here xk==x0 et xk+1==x1 and |xk - xk+1|==init_step==epsilon
    #so we can stop searching and choose any x in between xk and xk+1, I chosed the x in middle.
    #-----------Calcule de la precision--------
    w=epsilon
    p=0
    while(w<1):
        w*=10
        p+=1
    #------------------------------------------
    return round(result,p)

#----------------1c Exhaustive search method--------------------------------------------------------------------------------------
def _es_(function,xs,xf,epsilon):
    step=epsilon
    num_of_equ_spaced_points=((xf-xs)/step)+1
    num_of_equ_spaced_points=int(num_of_equ_spaced_points)
    list_of_evaluations=[]
    xi=0
    if(function(xs)>function(xs+step)):
        for i in range(0,num_of_equ_spaced_points):
            x=xs+i*step
            list_of_evaluations.append(function(x))
        min=list_of_evaluations[0]
        p=0
        for i in range(0,num_of_equ_spaced_points):
            if(list_of_evaluations[i]<=min):
                min=list_of_evaluations[i]
                p=i
        xi=xs+p*step
        xi_1=xs+(p-1)*step
        xi1=xs+(p+1)*step
        #the final interval of uncertitude is [xi-1,xi+1] of length 2*step which is epsilon so i chosed the middle of this interval for instance xi as a minimum  
    elif(function(xs)==function(xs+step)):
        xi=(xs+xf)/2
    result = xi
    #-----------Calcule de la precision--------
    w=epsilon
    p=0
    while(w<1):
        w*=10
        p+=1
    #------------------------------------------
    return round(result,p)
#-----------------1d Dichotomous search method------------------------------------------------------------------------------------
def _ds_(function,xs,xf,epsilon,mini_delta):
    x_middle=(xf+xs)/2
    if(epsilon<= mini_delta):
        return " mini_delta must be way smaller than epsilon !"
    elif((xf-xs)<=epsilon):
        x_min=x_middle
        return x_min
    else:
        x1= x_middle - (mini_delta/2)
        x2= x_middle + (mini_delta/2)
        f1=function(x1)
        f2=function(x2)
        if(f1<f2):
            xf=x2
        elif(f1>f2):
            xs=x1
        else:
            xs=x1
            xf=x2
        return _ds_(function,xs,xf,epsilon,mini_delta)
    
#------------1e Interval halving method----------------------------------------------------------------------------------------------
def _ih_(function,a,b,epsilon):

    L=b-a
    step=L/4
    x1=a+step
    x0=a+2*step
    x2=a+3*step
    f1=function(x1)
    f2=function(x2)
    f0=function(x0)
    if(f1<=f0 and f0<=f2):
        b=x0
        x0=x1
    elif(f1>=f0 and f0>=f2):
        a=x0
        x0=x2
    elif(f1>=f0 and f0<=f2):
        a=x1
        b=x2
    L=b-a
    while(L>=epsilon):
        step=L/4
        x1=a+step
        x2=a+3*step
        f1=function(x1)
        f2=function(x2)
        f0=function(x0)
        if(f1<f0 and f0<f2):
            b=x0
            x0=x1
        elif(f1>f0 and f0>f2):
            a=x0
            x0=x2
        elif(f1>f0 and f0<f2):
            a=x1
            b=x2
        L=b-a
        #print("L = ",L,"\n")
    x_min=(b+a)/2
    result = x_min
    #-----------Calcule de la precision--------
    w=epsilon
    p=0
    while(w<1):
        w*=10
        p+=1
    #------------------------------------------
    return round(result,p)
#----------------1f  Fibonacci method-------------------------------------------------------------------------------------
def _fse_(n):
    un_1=un_2=1
    if(n<=1):
        return un_1
    else:
        i=2
        while(i<=n):
            un=un_1+un_2
            un_2=un_1
            un_1=un
            i+=1
        return un

def _fi_(f,a,b,n):


    precision=(b-a)/fibonacci_sequence(n)
    while(1):
        L=b-a
        L_ét=(_fse_(n-2)/_fse_(n))*L
        c=a+L_ét
        d=b-L_ét
        fc=f(c)
        fd=f(d)
        if(fc<fd):
            b=d
        elif(fc>fd):
            a=c
        else:
            a=c
            b=d
        if((d-c)<precision):
           result = (a+b)/2
           break
        n-=1
    #-----------Calcule de la precision--------
    w=precision
    p=0
    while(w<1):
        w*=10
        p+=1
    #------------------------------------------
    return round(result,p)

#------------------1g Golden section method---------------------------------------------------------------------------------------

def _gs_(f,a,b,epsilon):

    gamma=0.618033
    while(1):
        L=b-a
        L_ét=gamma*L
        c=a+L_ét
        d=b-L_ét
        fc=f(c)
        fd=f(d)
        if(fc<fd):
            a=d
        elif(fc>fd):
            b=c
        else:
            a=d
            b=c
        if((b-a)<epsilon):
           result = (a+b)/2
           break
    #-----------Calcule de la precision--------
    w=epsilon
    p=0
    while(w<1):
        w*=10
        p+=1
    #------------------------------------------
    return round(result,p)
    
#------------------Armijo methods---------------------------------------------------------------------------------------

def _ab_(f,x0,ŋ,epsilon):

    a=x0
    f_prime_de_0 = ( f(1.e-6)-f(0) ) / 1.e-6
    
    def f_chapeau(a):
            return f(0) + epsilon*a*f_prime_de_0
        
    if(f(a)<=f_chapeau(a)): return "Choisissez un x0 plus grand qui ne vérifie pas la condition d'Armijo pour votre f et epsilone  !"
    
    else:
        while(f(a)>f_chapeau(a)):
            a=a/ŋ
        result = a
        #-----------Calcule de la precision--------
        w=epsilon
        p=0
        while(w<1):
            w*=10
            p+=1
        #------------------------------------------
        return round(result,p)    


def _af_(f,x0,ŋ,epsilon):

    a=x0
    f_prime_de_0 = ( f(1.e-6)-f(0) ) / 1.e-6
    def f_chapeau(a):
            return f(0) + epsilon*a*f_prime_de_0
        
    if(f(a)>f_chapeau(a)): return "Choisissez un x0 plus petit qui vérifie la condition d'Armijo pour votre f et epsilone !"
    
    else:
        while(f(a)<=f_chapeau(a)):
            a=a*ŋ
        result = a
        #-----------Calcule de la precision--------
        w=epsilon
        p=0
        while(w<1):
            w*=10
            p+=1
        #------------------------------------------
        return round(result,p)  
    
    
                                                      
