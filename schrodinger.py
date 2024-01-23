import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import eig

def V_harmonic(X):    
    k0=0
    k2=1.4494
    V_harm=[]
    for x in X:
        v_harm = k0 + k2*(x**2)
        V_harm.append(v_harm)
    return V_harm

def Schrodinger_harm(X,V):
    dx = (X[-1]-X[0])/(len(X)-1)     #Angstrom 
    m  = 266.1*931.49*1.0e6          #Non dimensionalizing with eV/c^2
    h  = 1973.2698                   #eV/c
    a  = (h**2)/(2*m*(dx**2))        
    
    A=np.zeros([n,n])
    for i in range(1,n-1):
        A[i][i-1] = -a
        A[i][i]   = 2*a+V[i]
        A[i][i+1] = -a
    A[0][0] = 2*a+V[0]
    A[0][1] = -a
    A[-1][-2]=-a
    A[-1][-1]=2*a+V[-1]
    
    eigvals,eigvecs=eig(A)
    eigvecs = np.transpose(eigvecs)
    Eigens = np.column_stack((eigvals,eigvecs))
    Eigens=Eigens[Eigens[:, 0].argsort()]
    
    Eig = Eigens[0:6,0]
    Eigvecs= Eigens[0:6,1:]

    return Eig,Eigvecs


n=1000
X=np.linspace(-1,1,n)
V_harm=V_harmonic(X)

Eig_harm,Eigvecs_harm=Schrodinger_harm(X,V_harm)

plt.figure(figsize=(8,10))
plt.plot(X, V_harm,'k')

for i in range(len(Eig_harm)):
    plt.axhline(Eig_harm[i],linestyle='--')

plt.ylim(0, 1.05*Eig_harm[5])  # Set appropriate y-axis limits
plt.yticks(Eig_harm, ['E0', 'E1', 'E2', 'E3', 'E4', 'E5'])
plt.ylabel("Energy h$\omega$ -->",fontsize=14)
plt.xlim([-0.175,0.175])
plt.title("Energy levels with harmonic potential",fontsize=18)
plt.xticks([])
plt.show()


plt.figure(figsize=(8,10))
plt.plot(X,np.array(Eigvecs_harm[0])/100+Eig_harm[0],'r')
plt.plot(X,np.array(Eigvecs_harm[1])/100+Eig_harm[1],'b')
plt.plot(X,np.array(Eigvecs_harm[2])/100+Eig_harm[2],'orange')
plt.plot(X,np.array(Eigvecs_harm[3])/100+Eig_harm[3],'g')
plt.plot(X,np.array(Eigvecs_harm[4])/100+Eig_harm[4],'m')

plt.plot(X,V_harm,'k')
plt.xticks([])
plt.yticks(Eig_harm[0:5], ['$\Psi_0$', '$\Psi_1$', '$\Psi_2$', '$\Psi_3$', '$\Psi_4$'])
plt.xlim([-0.4,0.4])
plt.ylim([0, 1.2*Eig_harm[4]])
plt.title("Eigen vectors with harmonic potential",fontsize=18)
plt.savefig('2.jpg',bbox_inches='tight', dpi=150)
plt.show()
