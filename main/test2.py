import math



def mat_mul(A, B): #matrix multiplication of two matrices: A*B
    res = [[] for _ in range(len(A))]
    if len(A[0]) != len(B):
        return "The two matrices cannot be multiplied, A doesn't have as many columns as B has rows."
    for line_ind in range(len(A)):
        for col_ind in range(len(B[0])):
            res[line_ind].append(round(sum(A[line_ind][k] * B[k][col_ind] for k in range(len(B))), 3))
    return res

def ncoeff8(i, j):
#Write a function ncoeff8(i, j) that takes two integers i & j in the range {0..8}
    if i == 0:
        return (1,4)
    else :
        return redalpha(i*(2*j+1))


def __matrix_mult(A, B):
    # computes the matrix product of A and B
    ans = [] 
    for i in range(len(A)):
        if len(A[0]) != len(B):
            raise ValueError("A doesn't have as many columns as B has rows.")
        ans.append([])
        for j in range(len(B[0])):
            somme = sum(A[i][k]*B[k][j] for k in range(len(B)))
            rounded = round (somme, 3)
            ans[i].append(rounded)
    return ans

print(__matrix_mult([[1,2,3],[4,5,6]],[[7,8],[9,-1],[-2,-3]]))

def DCT_Chen_line(v):
    aout = []
    for i in range(len(v)):
        aout.append(DCT_Chen(v[i]))
    return aout

def block_splitting(w, h, C):
   C1 = C
   l = len(h)
   if w % 8 != 0:
       for i in range(h):
           C1[i].extend([C[i][-1]] * (8 * (w//8 + 1)-w))
   if h % 8 != 0:

       C1.extend([C[-1]] * (8*(h//8 + 1) - h))

   for i in range(len(C1)//8):

           for j in range(len(C1[0])//8):

               block = []

               for l in range(i*8, (i+1)*8):

                   block.append(C1[l][j*8: (j+1)*8])

               yield block
     
def block_splitting(w, h, C):
    """Block splitting"""
    l = len(h)
    l1 = len(mem[0])//8
    l2 = len(mem)//8
    mem = C
    i=0
    j=0
    if w % 8 != 0:
        while i < h:
            mem[i].extend([C[i][-1]] * (8 * (w//8 + 1)-w))
            i += 1
    if h % 8 != 0:
        mem.extend([C[-1]] * (8*(h//8 + 1) - h))
    while j < l2:
        for k in range(l1):
            ans = []
            for l in range(k*8, (k+1)*8):
                ans.append(mem[l][k*8: (k+1)*8])
            yield ans

        

    


print(DCT_Chen([
 [ 140,  144,  147,  140,  140,  155,  179,  175],
 [ 144,  152,  140,  147,  140,  148,  167,  179],
 [ 152,  155,  136,  167,  163,  162,  152,  172],
 [ 168,  145,  156,  160,  152,  155,  136,  160],
 [ 162,  148,  156,  148,  140,  136,  147,  162],
 [ 147,  167,  140,  155,  155,  140,  136,  162],
 [ 136,  156,  123,  167,  162,  144,  140,  147],
 [ 148,  155,  136,  155,  152,  147,  147,  136],
]))
