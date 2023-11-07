import math as m
import numpy as np

def gaussjordan(A):
    """
gaussjordan(A) takes a square matrix A as an argument,

    if A is invertable, it returns it's inverse using Gauss-Jordan method
    otherwise it returns an error message.
    
""" 
    n=len(A)
    I=np.identity(n)
    for i in range(n-1):
        maxx=abs(A[i,i])
        for j in range(i,n):
            if(abs(A[j,i])>maxx): maxx=abs(A[j,i])       
        
        if(maxx!=0):
            j=i
            while(abs(A[j,i])!=maxx):
                j+=1
            for k in range(0,n):
                save=A[i,k]
                A[i,k]=A[j,k]
                A[j,k]=save
                #same changes for In
                save=I[i,k]
                I[i,k]=I[j,k]
                I[j,k]=save
        else:
            return "This matrix is not invertable"
        
        for j in range(i+1,n):
            if(A[j,i]==0): continue
            c=A[j,i]/A[i,i]
            c=c.round(1)
            for k in range(i,n):
                A[j,k]=A[j,k]- c*A[i,k]
                #same changes for In
                I[j,k]=I[j,k]- c*I[i,k]
            A=np.around(A,2)    

    
    #going backwards

    for i in range(n-1,0,-1):
        for j in range(i-1,-1,-1):
            if(A[j,i]==0): continue            
            c=A[j,i]/A[i,i]
            c=c.round(1)
            for k in range(n-1,-1,-1):
                A[j,k]=A[j,k] - c*A[i,k]
                #same changes for In
                I[j,k]=I[j,k] - c*I[i,k]
    #normalizing the diagonal
    for i in range(n):
        Aii=A[i,i]
        for k in range(n):
            I[i,k]=I[i,k]/Aii
            A[i,k]=A[i,k]/Aii
    return I
