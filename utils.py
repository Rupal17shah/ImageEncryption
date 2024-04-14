import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
import hashlib
import math
import cv2
import base64
import scipy

global M, N
M = 0
N = 0
class Arnold:
    def __init__(self, a:int, b:int, rounds:int):
        # Parameters
        self.__a = a
        self.__b = b
        self.__rounds = rounds

    def mapping(self, s:np.shape):
        x, y = np.meshgrid(range(s[0]), range(s[0]), indexing="ij")
        xmap = (self.__a*self.__b*x + x + self.__a*y) % s[0]
        ymap = (self.__b*x + y) % s[0]
        return xmap, ymap

    def inverseMapping(self, s:np.shape):
        x, y = np.meshgrid(range(s[0]), range(s[0]), indexing="ij")
        xmap = (x - self.__a*y) % s[0]
        ymap = (-self.__b*x + self.__a*self.__b*y + y) % s[0]
        return xmap, ymap

    def applyTransformTo(self, image:np.ndarray):
        xm, ym = self.mapping(image.shape)
        img = image
        for r in range(self.__rounds):
            img = img[xm, ym]
        return img

    def applyInverseTransformTo(self, image:np.ndarray):
        xm, ym = self.inverseMapping(image.shape)
        img = image
        for r in range(self.__rounds):
            img = img[xm, ym]
        return img

# Chirikov standard map
def generateChirikovMap(xo, yo, iter, delta, to):
    x = []
    y = []
    for i in range(iter+to):
        x1 = (xo+yo)%(2*math.pi)
        y1 = (xo+delta*math.sin(xo+yo))%(2*math.pi)
        xo = x1
        yo = y1
        if i>=to:
            x.append(x1)
            y.append(y1)
    return x,y

def padding(image):
    M,N = image.shape
    if (M<N):
        pad = np.zeros((N,N))
        pad[0:M,:] = image
    else:
        pad = np.zeros((M,M))
        pad[:,0:N] = image
    return pad

def unpadding(M, N, image):
    return image[0:M,0:N]

def generateKey(password):
    hash = hashlib.sha512(password.encode())
    hash_str = hash.hexdigest()
    k = []
    prev = 0
    for i in range(2,129,2):
        kth = hash_str[prev:i]
        prev = i
        k.append(int(kth, 16))
    a1 = reduce(lambda x, y: x ^ y, k[:16])
    a2 = reduce(lambda x, y: x ^ y, k[16:32])
    a3 = reduce(lambda x, y: x ^ y, k[32:48])
    a4 = reduce(lambda x, y: x ^ y, k[48:64])
    t1 = 1
    t2 = 2
    t3 = 3
    t4 = 4
    # Four parameters
    h1 = t1 + (a1+a2)/256
    h2 = t2 * (a3+a4)/256
    h3 = t3 + (a1+a2)/256
    h4 = t4 * 256 * sum(k[16:32])/max(k[16:32])
    # parameters u1, u2, x1, x2 are used as initial values of the 4-D hyperchaotic system
    xo = (math.floor((h1 + h2 + h3) * 1e5) % 512)/512
    yo = (math.floor((h2 + h3 + h4) * 1e5) % 512)/512
    xo_ = (math.floor((h1 + h2 + h3 + h4) * 1e5) % 512)/512
    yo_ = (math.floor((h1 + h4) * 1e5) % 512)/512

    return xo, yo, xo_, yo_

def encrypt(password, image):

    xo, yo, xo_, yo_ = generateKey(password)

    I1 = image[:,:,0]
    I2 = image[:,:,1]
    I3 = image[:,:,2]

    M = I1.shape[0]
    N = I1.shape[1]

    # Generate chaotic chirikov map for vertical shift
    x3m,y1 = generateChirikovMap(xo, yo, 3*M, 0.1, 1000)
    # plot1 = plt.plot(x,y1)
    x3m = np.array(x3m)
    print(len(x3m))
    x1,yn = generateChirikovMap(xo, yo, N, 0.1, 1000)
    # plot2 = plt.plot(x1,y)
    yn = np.array(yn)
    print(len(yn))

# Generated chaotic sequence for encryption
    Xv = np.floor(x3m*1e14)%N
    Yv = np.floor(yn*1e14)%(3*M)

    # Combining the three indexed images
    Fv = np.concatenate((I1, I2, I3))
    actual_concat = np.copy(Fv)

    # Performing right shift in dimension N
    for i in range(len(Xv)):
        row = Fv[i,:]
        Fv[i,:] = np.roll(row, int(Xv[i]))

    # Performing bottom shift in dimension 3M
    for j in range(len(Yv)):
        column = Fv[:,j]
        Fv[:,j] = np.roll(column, int(Yv[j]))
    
    # Decomposing the scrambled image into three channels
    F1 = Fv[0:M, :]
    F2 = Fv[M:2*M, :]
    F3 = Fv[2*M:3*M, :]

    # Horizontal concatenation of the three channels
    Fh = np.concatenate((F1, F2, F3), axis=1)

    F_checkh = np.copy(Fh)

# Generate chaotic chirikov map for vertical shift
    xh,_ = generateChirikovMap(xo_, yo_, M, 0.1, 1000)
    xh = np.array(xh)
    print(len(xh))
    _,yh = generateChirikovMap(xo_, yo_, 3*N, 0.1, 1000)
    yh = np.array(yh)
    print(len(yh))

    Xh = np.floor(xh*1e14)%(3*N)
    Yh = np.floor(yh*1e14)%M

    # Performing left shift in dimension N
    for i in range(len(Xh)):
        row = Fh[i,:]
        Fh[i,:] = np.roll(row, int(Xh[i]))

    # Performing bottom shift in dimension M
    for j in range(len(Yh)):
        column = Fh[:,j]
        Fh[:,j] = np.roll(column, int(Yh[j]))

    C1 = Fh[:, 0:N]
    C2 = Fh[:, N:2*N]
    C3 = Fh[:, 2*N:3*N]

    # Encrypted image
    E = np.zeros((M, N, 3), dtype=np.uint8)
    E[:,:,0] = C1
    E[:,:,1] = C2
    E[:,:,2] = C3
    C1_dct = scipy.fftpack.dct(C1, axis=0, norm='ortho', type=2)
    C2_dct = scipy.fftpack.dct(C2, axis=0, norm='ortho', type=2)
    C3_dct = scipy.fftpack.dct(C3, axis=0, norm='ortho', type=2)

    # padding
    C1_pad = padding(C1_dct)
    C2_pad = padding(C2_dct)
    C3_pad = padding(C3_dct)

    arnold = Arnold(1, 2, 20)

    C1_AT = arnold.applyTransformTo(C1_pad)
    C2_AT = arnold.applyTransformTo(C2_pad)
    C3_AT = arnold.applyTransformTo(C3_pad)

    encrypted_img = np.zeros((C1_AT.shape[0], C1_AT.shape[1], 3))
    encrypted_img[:,:,0] = C1_AT
    encrypted_img[:,:,1] = C2_AT
    encrypted_img[:,:,2] = C3_AT

    return encrypted_img

def decrypt(password, encrypted_img):

    xo, yo, xo_, yo_ = generateKey(password)
    iC1_AT = encrypted_img[:,:,0]
    iC2_AT = encrypted_img[:,:,1]
    iC3_AT = encrypted_img[:,:,2]
    arnold = Arnold(1, 2, 20)

    IC1_AT = arnold.applyInverseTransformTo(iC1_AT)
    IC2_AT = arnold.applyInverseTransformTo(iC2_AT)
    IC3_AT = arnold.applyInverseTransformTo(iC3_AT)

    # unpadding 
    IC1_unpad = unpadding(M, N, IC1_AT)
    IC2_unpad = unpadding(M, N, IC2_AT)
    IC3_unpad = unpadding(M, N, IC3_AT)

    iC1 = scipy.fftpack.idct(IC1_unpad, axis=0, norm='ortho', type=2)
    iC2 = scipy.fftpack.idct(IC2_unpad, axis=0, norm='ortho', type=2)
    iC3 = scipy.fftpack.idct(IC3_unpad, axis=0, norm='ortho', type=2)

    iC1 = iC1.round().astype(np.uint8)
    iC2 = iC2.round().astype(np.uint8)
    iC3 = iC3.round().astype(np.uint8)

    iFh = np.concatenate([iC1,iC2,iC3], axis=1)
    print(iFh.shape)

    # Generate chaotic chirikov map for vertical shift
    ixm,y1 = generateChirikovMap(xo_, yo_, M, 0.1, 1000)
    # plot1 = plt.plot(x,y1)
    ixm = np.array(ixm)
    print(len(ixm))
    x1,iy3n = generateChirikovMap(xo_, yo_, 3*N, 0.1, 1000)
    # plot2 = plt.plot(x1,y)
    iy3n = np.array(iy3n)
    print(len(iy3n))

    iXh = np.floor(ixm*1e14)%(3*N)
    iYh = np.floor(iy3n*1e14)%M

    # Performing bottom shift in dimension M
    for j in range(len(iYh)):
        column = iFh[:,j]
        iFh[:,j] = np.roll(column, -int(iYh[j]))
    

    # Performing bottom shift in dimension N
    for i in range(len(iXh)):
        row = iFh[i,:]
        iFh[i,:] = np.roll(row, -int(iXh[i]))

    F1 = iFh[:,:N]
    F2 = iFh[:,N:2*N]
    F3 = iFh[:,2*N:3*N]

    Fv = np.concatenate([F1,F2,F3])

    # Generate chaotic chirikov map for vertical shift
    ix3m,y1 = generateChirikovMap(xo, yo, 3*M, 0.1, 1000)
    # plot1 = plt.plot(x,y1)
    ix3m = np.array(ix3m)
    print(len(ix3m))
    x1,iyn = generateChirikovMap(xo, yo, N, 0.1, 1000)
    # plot2 = plt.plot(x1,y)
    iyn = np.array(iyn)
    print(len(iyn))

    # Generated chaotic sequence for encryption
    iXv = np.floor(ix3m*1e14)%N
    iYv = np.floor(iyn*1e14)%(3*M)
    iFv = np.copy(Fv)
    # Performing up shift in dimension 3M
    for j in range(len(iYv)):
        column = iFv[:,j]
        iFv[:,j] = np.roll(column, -int(iYv[j]))

    # Performing left shift in dimension N
    for i in range(len(iXv)):
        row = iFv[i,:]
        iFv[i,:] = np.roll(row, -int(iXv[i]))

    ch1 = iFv[:M,:]
    ch2 = iFv[M:2*M,:]
    ch3 = iFv[2*M:3*M,:]

    chirikov_decrypted = np.zeros((M, N, 3), dtype=np.uint8)
    chirikov_decrypted[:,:,0] = ch1
    chirikov_decrypted[:,:,1] = ch2
    chirikov_decrypted[:,:,2] = ch3

    return chirikov_decrypted