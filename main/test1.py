import math


def __delta(index):
    if index == 0:
        return 1
    else:
        return 1/math.sqrt(2)

def __alpha(index):
    

def __res (val):
    for i in range(val):
        res += [0]*val
    return res


def __Cont(val):
    ans= [[1]*val for k in range(val)]
    for line in range(val):
        for collumn in range (val):
            ans[line][collumn] = __delta(line)*math.sqrt(2/val)*math.cos((math.pi/val)*(collumn + 1/2)*line)
    return ans


def __aux(l):
    res = [[] for i in range(l)]
    return res

def __trans(M):
    l = len(M[0])
    h = len(M)
    aux = __aux(l)
    for line in range(h):
        for column in range(len(M[line])):
            aux[column].append(M[line][column])
    return aux

def DCT(v):
#Write a function DCT(v) that computes and return the DCT-II of the vector v.
    ans = []
    trans = __trans(__Cont(len(v)))
    for i in range (0, len(v)):
        somme = sum(v[j]*trans[j][i] for j in range(len(v)))
        rounded = round (somme, 2)
        ans.append(rounded)
    return ans

print(DCT([8, 16, 24, 32, 40, 48, 56, 64]))
    
def IDCT(v):
# computes and return the DCT-II of the vector v (opposote of DCT)
    ans = []
    cont = __Cont(len(v))
    for i in range(len(v)):
        somme = sum(v[j]*cont[j][i] for j in range(len(v)))
        rounded = round (somme, 2)
        ans.append(rounded)
    return ans

def __matrix_mult(A, B):
    # computes the matrix product of A and B
    ans = []
    for i in range(len(A)):
        ans.append([])
        for j in range(len(B[0])):
            somme = sum(A[i][k]*B[k][j] for k in range(len(A)))
            rounded = round (somme, 2)
            ans[i].append(rounded)
    return ans

 
def DCT2(m, n, A):
    #that computed the 2D DCT-II of the matrix A.
    #The matrix A is of size m x n.
    #The matrix A is returned.
    return __matrix_mult(__Cont(m), __matrix_mult(A, __trans(__Cont(n))))

    
def IDCT2(m, n, A):
    #opposite of DCT2
    return __matrix_mult(__trans(__Cont(m), __matrix_mult(A, __Cont(n))))
