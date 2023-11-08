import math as m
import numpy as np




def gaussjordan(A,Y):
    """gaussjordan(A,Y) takes a square matrix A and a column vector Y of the same length as A and
    if A is invertible it solves the system A@X=Y of linear equations using Gauss-Jordan method and returns the column vector solution X:
    otherwise it returns an error message."""
    n=len(A)
    A=A.astype(float)
    Y=Y.astype(float)
    
    #looking for the highest pivot value
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
            save=Y[i,0]
            Y[i,0]=Y[j,0]
            Y[j,0]=save
            
        else:
            return "This matrix is not invertable"
        
        for j in range(i+1,n):
            if(A[j,i]==0): continue
            c=A[j,i]/A[i,i]
            for k in range(0,n):
                A[j,k]=A[j,k]- c*A[i,k]
                #same changes for In
            Y[j,0]=Y[j,0]- c*Y[i,0]
    
    
    #going backwards
    for i in range(n-1,0,-1):
        for j in range(i-1,-1,-1):
            if(A[j,i]==0): continue
            c=A[j,i]/A[i,i]
            for k in range(i,-1,-1):
                A[j,k]=A[j,k] - c*A[i,k]
                #same changes for Y
            Y[j,0]=Y[j,0] - c*Y[i,0]
    #normalizing
    for i in range(n):
        Y[i,0]=Y[i,0]/A[i,i]
    
    Y[Y==0.]=0. #eliminate negative zero
    return Y

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

def LU(A,Y):
    """LU(A,Y) takes a square matrix A and a column vector Y of the same length as A and

    if A is invertible it solves the system A@X=Y of linear equations using LU decomposition and returns the column vector solution X:
    otherwise it returns an error message.""" 
    B=_lu_(A)
    L=B[0]
    U=B[1]
    P=B[2]
    n=len(A)
    XXX=XX=np.zeros(n) #XXX=L*U*X changement de variable
    XX=np.zeros(n) #XX=U*X changement de variable
    X=np.zeros((n,1))
    for i in range(n):
        for j in range(n):
            if(P[i,j]==1):
                XXX[j]=Y[i]
    
    for i in range(0,n):
        summ=0
        for k in range(0,i):
            summ+=L[i,k]*XX[k]
        XX[i]=(XXX[i]-summ)/L[i,i]
        
    for i in range(n-1,-1,-1):
        summ=0
        for k in range(i+1,n):
            summ+=U[i,k]*X[k]
        X[i]=(XX[i]-summ)/U[i,i] 
    return X


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

def choleski(A,Y):
    """choleski(A,Y) takes a square matrix A and a column vector Y of the same length as A and

    if A is positive definite it solves the system A@X=Y of linear equations using choleski decomposition and returns the column vector solution X:
    otherwise it returns an error message."""
    if(_is_cho_(A)):
        L=_cho_(A)
        LT=L.T
        n=len(A)
        XX=np.zeros(n) #XX=LT*X changement de variable
        X=np.zeros((n,1))
        for i in range(0,n):
            summ=0
            for k in range(0,i):
                summ+=L[i,k]*XX[k]
            XX[i]=(Y[i]-summ)/L[i,i]
            
        for i in range(n-1,-1,-1):
            summ=0
            for k in range(i+1,n):
                summ+=LT[i,k]*X[k]
            X[i]=(XX[i]-summ)/LT[i,i] 

        return X
    else:
        print('!! A must be positive definite !!')
        


def _lu_(A):
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
    L=P@_inv_(I)
    P=_inv_(P)
    return (L,U,P)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

def _is_cho_(A):
    return np.all(np.linalg.eigvals(A) > 0)

def _cho_(A):
    
    #if A is not positive definite-----------------------------------------------------------------------------

    if(not _is_cho_(A)):
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
                    
                    sum=0
                    for k in range(0,j):
                        sum+=L[j,k]**2
                        
                    L[i,j]=m.sqrt(A[i,j]-sum)
                
                else:
                    
                    Ljj=L[j,j]
                    sum=0
                    for k in range(0,j):
                        sum+=L[i,k]*L[j,k]
                        
                    L[i,j]=(A[i,j]-sum)/Ljj
        return L


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _inv_(A):
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
