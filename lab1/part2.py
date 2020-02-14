import numpy as np

A = np.zeros((9,6), dtype = np.int16) #Init A

A[0:2,1:5] = 1 #Fill in 1s to make I
A[2:7,2:4] = 1
A[7:,1:5] = 1
#print(A)

B = np.insert(A, 9, [0,0,0,0,0,0], 0) #init B
B = np.insert(B, 0, [0,0,0,0,0,0], 0)
#print(B)

C = np.arange(1,67).reshape(11,6) #init C
#print(C)

D = np.multiply(B, C) #init D
#print(D)

E = D.flatten() #init E
E = E[E != 0]
#print(E)

max, min = E.max(), E.min() #init F
F = (E - min) / (max - min)
#print(F)

absar = np.abs(F - 0.25) #array of differences
#print(F[absar.argmin()])