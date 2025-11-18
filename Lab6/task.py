import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve


lambda_12 = 5.0
lambda_14 = 4.0
lambda_15 = 1.0
lambda_23 = 2.0
lambda_24 = 8.0
lambda_25 = 2.0
lambda_31 = 3.0
lambda_34 = 7.0
lambda_42 = 2.0
lambda_45 = 3.0
lambda_52 = 4.0

# Сборка Lambda
Lambda = np.zeros((5,5))
def add(i,j,lam):
    Lambda[j,i] += lam
    Lambda[i,i] -= lam

add(0,1,lambda_12)
add(0,3,lambda_14)
add(0,4,lambda_15)
add(1,2,lambda_23)
add(1,3,lambda_24)
add(1,4,lambda_25)
add(2,0,lambda_31)
add(2,3,lambda_34)
add(3,1,lambda_42)
add(3,4,lambda_45)
add(4,1,lambda_52)

print("Lambda =\n", Lambda)

# Аналитическое P_lim
M = Lambda.copy()
b = np.zeros(5)
M[4,:] = 1.0
b[4] = 1.0
P_lim = solve(M,b)
print("P_lim =", P_lim)

# Эйлер
P0 = np.array([1.,0.,0.,0.,0.])
T = 20.0; h = 0.01
I = np.eye(5)
A = I + h*Lambda
N = int(T/h)
P = P0.copy()
times = np.linspace(0,T,N+1)
traj = np.zeros((N+1,5))
traj[0,:] = P0
for k in range(N):
    P = A @ P
    s = P.sum()
    if s>0:
        P = P / s
    traj[k+1,:] = P

# График
plt.figure(figsize=(10,6))
for i in range(5):
    plt.plot(times, traj[:,i], label=f"p_{i+1}(t)")
plt.xlabel("t"); plt.ylabel("p_i(t)")
plt.legend(); plt.grid(True)
plt.show()