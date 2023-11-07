#----------------1a Unrestricted_fixed_step_search----------------------------------------------------------------------------------------------

def fixed_step(function,x0,epsilon=0.01):
    """+ fixed_step function takes 3 arguments:
- f : a one variable & unimodal function 
- x0 : the initial starting point
- epsilon : which is the target precision
       + It returns the argmin(f)
       * the step size is fixed at epsilon """
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
        result = (x0 + x1)/2  #here xk==x0 et xk+1==x1 and |xk - xk+1|==step==epsilon
    #so we can stop searching and choose any x in between xk and xk+1, I chosed the x in middle.
    
    elif(f_1<f0):
        while(f_1<f0):    
            x0=x_1
            x_1=x0-step
            f0=function(x0)
            f_1=function(x_1)
        result = (x0 + x_1)/2 #here xk==x0 et xk+1==x1 and |xk - xk+1|==step==epsilon
    #so we can stop searching and choose any x in between xk and xk+1, I chosed the x in middle.
    
    else:
        result = (x0 + x1)/2   #here xk==x0 et xk+1==x1 and |xk - xk+1|==step==epsilon
    #so we can directly choose any x in between xk and xk+1, I chosed the x in middle.    
    #-----------Calcule de la precision--------
    w=epsilon
    p=0
    while(w<1):
        w*=10
        p+=1
    #------------------------------------------
    return round(result,p)
#--------------1b Unrestricted_accelerated_step_search------------------------------------------------------------------------------------------
def accelerated_step(function,x0,epsilon=0.01):
    """+ accelerated_step function takes 3 arguments:
- f : a one variable & unimodal function 
- x0 : the initial starting point
- epsilon : which is the target precision
       + It returns the argmin(f)
       * the step size is initialized at epsilon """
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
def exhaustive_search(function,xs,xf,epsilon=0.01):
    """+ The exhaustive_search function takes 4 arguments:
- f : a one variable & unimodal function 
- xs : the starting point
- xf : the finishing point
- epsilon : which is the target precision
       + It returns the argmin(f)
       * the step size is fixed at epsilon """
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
def dichotomous_search(function,xs,xf,epsilon=0.01,mini_delta=0.001):
    """ + The dichotomous_search function takes 5 arguments:
- f : a one variable & unimodal function 
- xs : the starting point
- xf : the finishing point
- epsilon : which is the target precision
- mini_delta : determinate the size of x_middle's neighborhood
        + It returns the argmin(f)
        ! mini_delta must be way smaller than epsilon """
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
        return dichotomous_search(function,xs,xf,epsilon,mini_delta)
    
#------------1e Interval halving method----------------------------------------------------------------------------------------------
def interval_halving(function,a,b,epsilon=0.01):
    """ + The interval_halving function takes 4 arguments:
- f : a one variable & unimodal function f
- a : the starting point
- b : the finishing point
- epsilon : which is the target precision
        + It returns the argmin(f) """
    L=b-a
    step=L/4
    x1=a+step
    x0=a+2*step
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
        + It returns the argmin(f) """

    precision=(b-a)/fibonacci_sequence(n)
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

def golden_section(f,a,b,epsilon=0.01):
    """ + The golden_section function takes 4 arguments:
- f : a one variable & unimodal function f
- a : the starting point
- b : the finishing point
- epsilon : which is the target precision
        + It returns the argmin(f) """
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

def armijo_backward(f,x0,ŋ=2,epsilon=0.01):
    """ + The armijo_backward (Backtracking_line_search) function takes 4 arguments:
- f : a one variable & unimodal function f
- x0 : the starting point
- ŋ : the coefficient by which we divide x0 at each iteration
- epsilon : which is the target precision
        + It returns the argmin(f) """
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


def armijo_forward(f,x0,ŋ=2,epsilon=0.01):
    """ + The armijo_forward function takes 4 arguments:
- f : a one variable & unimodal function f
- x0 : the starting point
- ŋ : the coefficient by which we multiply x0 at each iteration
- epsilon : which is the target precision
        + It returns the argmin(f) """
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
    
    
