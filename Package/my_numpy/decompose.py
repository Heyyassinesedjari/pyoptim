import math as m
import numpy as np

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

def LU(A):
    """
LU(A) takes a square matrix A as an argument and returns a tuple of 3 square matrices (L,U,P) such that:

    A = P@L@U , where:
                        P is a permutation matrix
                        L is a lower triangular matrix
                        U is an upper tiangular matrix
    
"""
    n=len(A)
    I=np.identity(n)
    P=np.identity(n)
    A=A.astype(float)
    
    for i in range(n-1):
        maxx=abs(A[i,i])
        for j in range(i,n):
            if(abs(A[j,i])>maxx): maxx=abs(A[j,i])
                
        
        if(maxx!=0):
            j=i
            while(j<n and abs(A[j,i])!=maxx):
                j+=1
            for k in range(0,n):
                save=A[i,k]
                A[i,k]=A[j,k]
                A[j,k]=save
                #same changes for I
                save=I[i,k]
                I[i,k]=I[j,k]
                I[j,k]=save
                #same changes for P
                save=P[i,k]
                P[i,k]=P[j,k]
                P[j,k]=save
                
            
        else:
            return "This matrix is not inversable"
        
        for j in range(i+1,n):
            if(A[j,i]==0): continue
            c=A[j,i]/A[i,i]
            for k in range(0,n):
                A[j,k]=A[j,k]- c*A[i,k]
                #same changes for In
                I[j,k]=I[j,k]- c*I[i,k]
    U=A
    L=P@__inv__(I)
    P=__inv__(P)
    return (L,U,P)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

def is_pos_def(A):
    return np.all(np.linalg.eigvals(A) > 0)

def choleski(A):
    """
choleski(A) takes a square matrix A as an argument and

    if A is positive definite it returns a square matrix L such that: A = L@(L.T) , where L is a lower triangular matrix.

    otherwise it returns an error message
                        
    
"""
    
    #if A is not positive definite-----------------------------------------------------------------------------

    if(not is_pos_def(A)):
        print("A must be positive definite !")
        
    #if A is positive definite we proceed to Choleski decomposition -------------------------------------------
    else:
        n=len(A)
        L=np.zeros((n,n))
        L00=m.sqrt(A[0,0])
        L[0,0]=L00
        
        for j in range(0,n):
            for i in range(j,n):
                
                if(j==0):
                    L[i,j]=A[i,j]/L00
                    
                elif(i==j):
                    
                    summ=0
                    for k in range(0,j):
                        summ+=L[j,k]**2
                        
                    L[i,j]=m.sqrt(A[i,j]-summ)
                
                else:
                    
                    Ljj=L[j,j]
                    summ=0
                    for k in range(0,j):
                        summ+=L[i,k]*L[j,k]
                        
                    L[i,j]=(A[i,j]-summ)/Ljj
        return L


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

def __inv__(A):
    n=len(A)
    I=np.identity(n)
    A=A.astype(float)
    
    for i in range(n-1):
        maxx=abs(A[i,i])
        for j in range(i,n):
            if(abs(A[j,i])>maxx): maxx=abs(A[j,i])
                
        
        if(maxx!=0):
            j=i
            while(j<n and abs(A[j,i])!=maxx):
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
            for k in range(0,n):
                A[j,k]=A[j,k]- c*A[i,k]
                #same changes for In
                I[j,k]=I[j,k]- c*I[i,k]
    
    
    #going backwards
    for i in range(n-1,0,-1):
        for j in range(i-1,-1,-1):
            if(A[j,i]==0): continue
            c=A[j,i]/A[i,i]
            for k in range(i,-1,-1):
                A[j,k]=A[j,k] - c*A[i,k]
                #same changes for In
                I[j,k]=I[j,k] - c*I[i,k]
    #normalizing the diagonal
    for i in range(n):
        Aii=A[i,i]
        for k in range(n):
            I[i,k]=I[i,k]/Aii
    
    I[I==0.]=0. #eliminate negative zero
    return I

