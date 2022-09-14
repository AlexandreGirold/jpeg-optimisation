#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:26:34 2022

@author: dieucrevette
"""
#Part 1


#Ex1: The PPM input format

import math


def ppm_tokenize(stream):
    for token in stream:
        if '#' in token:
            a = token.split('#')
            token = a[0]
        token = token.split()
        for i in range(len(token)):
            yield token[i]

# with open('file.ppm') as stream:
#     for token in ppm_tokenize(stream):
#         print(token)
        
def ppm_load(stream):
    l = [i for i in ppm_tokenize(stream)]
    l2 = []
    w = l[1]
    h = l[2]
    for i in range(4, len(l), 3):
        l2.append([l[i], l[i+1], l[i+2]])
    return (w, h, l2)

def ppm_save(w, h, img, output):
    with open(output, 'w') as output:
        output.write('P3')
        output.write('\n')
        output.write(f'{w}'+" "+f'{h}')
        output.write('\n')
        output.write('255')
        output.write('\n')
        for k in range(len(img)):
            output.write('\n') 
            for j in range(3):
                line = img[k][j]
                output.write(f'{line}')
                output.write(' ')  

with open('file.ppm') as stream:    
    w,h,img = ppm_load(stream)
    ppm_save(w, h , img , 'outfile.txt')
    
    
#Ex2: RGB to YCbCr conversion & channel separation

def RGB2YCbCr(r, g, b):
    l = []
    Y = round(0.299*r + 0.587*g + 0.114*b)
    Cb = round(128 - 0.168736*r - 0.331264*g + 0.5*b)
    Cr = round(128 + 0.5*r - 0.418688*g - 0.081312*b)
    for i in (Y, Cb, Cr):
        if i > 255:
            i = 255
        if i < 0:
            i = 0
        l.append(i)
    return (tuple(l))

def YCbCr2RGB(Y, Cb, Cr):
    l = []
    r = round(Y + 1.402*(Cr-128))
    g = round(Y - 0.344136*(Cb-128) - 0.714136*(Cr-128))
    b = round(Y + 1.772*(Cb-128))
    for i in (r, g, b):
        if i > 255:
            i = 255
        if i < 0:
            i = 0
        l.append(i)
    return (tuple(l))

def img_RGB2YCbCr(img):
    Y = []
    Cb = []
    Cr = []
    for i in range(len(img)):
        Y.append([])
        Cb.append([])
        Cr.append([])
        for j in range(len(img[i])):
            r, g, b = img[i][j]
            color = RGB2YCbCr(r, g, b)
            Y[i].append(color[0])
            Cb[i].append(color[1])
            Cr[i].append(color[2])
    return (Y, Cb, Cr)
    
def img_YCbCr2RGB(Y, Cb, Cr):
    R = []
    G = []
    B = []
    for i in range(len(Y)):
        R.append([])
        G.append([])
        B.append([])
        for j in range(len(Y[i])):
            y, cb, cr = Y[i][j], Cb[i][j], Cr[i][j]
            color = YCbCr2RGB(y, cb, cr)
            R[i].append(color[0])
            G[i].append(color[1])
            B[i].append(color[2])
    return (R, G, B)
            
    
#Ex3: Subsampling

def subsampling(w, h, C, a, b):
    Csub = []
    sums = 0
    step = 0
    step2 = 0
    for n in range(h//b):
        Csub.append([])
        for _ in range(w//a):
            for i in range(step2, step2 + b):
                for k in range(step, step + a):
                    sums += C[i][k]
            new = round(sums/(a*b))
            Csub[n].append(new)
            sums = 0
            if step+a < w:
                step += a
        if step2+b < h:
            step2 += b
    return Csub

def extrapolate(w, h, C, a, b):
   return [[C[i//a][j//b] for j in range(w)] for i in range(h)]
    

# #Ex4: Block splitting

def block_splitting(w, h, C):
    Q = []
    Q0 = [[C[i][j] for j in range(8)] for i in range(8)]
    Q.append(Q0)
    
    if w > 8:
        Q1 = []
        for i in range(8):
              Q1.append([])
              for j in range(8, w):
                  Q1[i].append(C[i][j])
    if w-8 < 8:
        for k in range(8-(w-8)):
            for i in range(8):
                Q1[i].append(Q1[i][-1])
    Q.append(Q1)        
    
    if h > 8:
        Q2 = []
        for i in range(8, h):
            Q2.append([])
            for j in range(8):
                Q2[i-8].append(C[i][j])
    if h-8 < 8:
        l = Q2[-1]
        for i in range(8-(h-8)):
            Q2.append(l)
    Q.append(Q2)

    if h > 8 and w > 8:
        Q3 = []
        for i in range(8, h):
            Q3.append([])
            for j in range(8, w):
                Q3[i-8].append(C[i][j])
    if w-8 < 8:
        for i in range(len(Q3)):
            for _ in range(8-(w-8)):
                Q3[i].append(Q3[i][-1])
    if h-8 < 8:
        l = Q3[-1]
        for i in range(8-(h-8)):
            Q3.append(l)
    Q.append(Q3) 
               
    for i in range(len(Q)):
        yield Q[i]
            
#test
#     for Q in block_splitting(10, 9, [
#     [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
#     [ 2,  3,  4,  5,  6,  7,  8,  9, 10,  1],
#     [ 3,  4,  5,  6,  7,  8,  9, 10,  1,  2],
#     [ 4,  5,  6,  7,  8,  9, 10,  1,  2,  3],
#     [ 5,  6,  7,  8,  9, 10,  1,  2,  3,  4],
#     [ 6,  7,  8,  9, 10,  1,  2,  3,  4,  5],
#     [ 7,  8,  9, 10,  1,  2,  3,  4,  5,  6],
#     [ 8,  9, 10,  1,  2,  3,  4,  5,  6,  7],
#     [ 9, 10,  1,  2,  3,  4,  5,  6,  7,  8],
# ]):
#         print(Q)    
        
    
#Part 2


#Ex1: Discrete Cosine Transform, the general case

#All the auxiliary functions are written as follows __namefunction():


def __delta(index):
    if index == 0:
        return 1
    else:
        return 1/math.sqrt(2)

def __res (val):
    for i in range(val):
        res += [1]*val
    return res


def __Cont(val):
    ans= [[0]*val for k in range(val)]
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


def DCT(v):
#Write a function DCT(v) that computes and return the DCT-II of the vector v.
    ans = []
    trans = __trans(__Cont(len(v)))
    for i in range (0, len(v)):
        somme = sum(v[j]*trans[j][i] for j in range(len(v)))
        rounded = round (somme, 2)
        ans.append(rounded)
    return ans

    
def IDCT(v):
# computes and return the DCT-II of the vector v (opposote of DCT)
    ans = []
    cont = __Cont(len(v))
    for i in range(len(v)):
        somme = sum(v[j]*cont[j][i] for j in range(len(v)))
        rounded = round (somme, 2)
        ans.append(rounded)
    return ans


def DCT2(m, n, A):
    #that computed the 2D DCT-II of the matrix A.
    #The matrix A is of size m x n.
    #The matrix A is returned.
    return __matrix_mult(__Cont(m), __matrix_mult(A, __trans(__Cont(n))))

    
def IDCT2(m, n, A):
    #opposite of DCT2
    return __matrix_mult(__trans(__Cont(m), __matrix_mult(A, __Cont(n))))

    
#Ex2: The 8x8 DCT-II Transform & Chen’s Algorithm


def redalpha(i):
#Write a function redalpha(i) that takes a non-negative integer i and that returns a pair (s, k)

#TODO: Could be done, my brain small

    return None


def ncoeff8(i, j):
#Write a function ncoeff8(i, j) that takes two integers i & j in the range {0..8}
    if i == 0:
        return (1,4)
    else :
        return redalpha(i*(2*j+1))
    

def DCT_Chen(A):
    #The function DCT_Chen takes as input a matrix A
    #and returns the 2D DCT-II transform of A thanks to the Chen algorithm'''
    m = len(A)
    n = len(A[0])
    ans = [[] for i in range(m)]
    for i in range(m):
        for j in range(n):
            s, k = ncoeff8(i, j)
            somme = sum(A[i][j]*ncoeff8(i, j)[0]*ncoeff8(i, j)[1] for i in range(m))
            rounded = round (somme, 2)
            ans[i].append(rounded)
    return ans  

#Ex3: The inverse 8x8 Transform

def IDCT_Chen(A):
#Inverse of DCT_Chen
#TODO : My brain small.
    return None   
    
#Ex4: Quantization

def quantization(A, Q):
    #The function quantization takes as input a matrix A and a quantization matrix Q
    #and returns the quantized matrix Aq
    ans = [[] for i in range(8)]
    for i in range(8):
        for j in range(8):
            ans[i].append(round(A[i][j]/Q[i][j], 2))
    return ans


def quantizationI(A, Q):
    #The function quantizationI takes as input a matrix A and a quantization matrix Q
    #and returns the inverse quantized matrix Aq
    ans = [[] for i in range(8)]
    for i in range(8):
        for j in range(8):
            ans[i].append(round(A[i][j]*Q[i][j], 2))
    return ans


#Part 3, Zig-Zag walk & RLE Encoding:

def zigzag(A):
    """The function zigzag takes as input an 8x8 matrix A
    and yields the values corresponding to the zig zag
    """
    solution=[[] for i in range(15)]
    print("zigzag:")
    for i in range(8):
        for j in range(8):
            sum=i+j
            if(sum%2 ==0):
                solution[sum].insert(0,A[i][j])
            else:
                solution[sum].append(A[i][j])
    for i in range(15):
        for j in range(len(solution[i])):
            yield (solution[i][j])
