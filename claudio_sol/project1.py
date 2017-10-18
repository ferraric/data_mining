#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:15:29 2017

@author: claudioferrari
"""
import re
import numpy as np
import itertools

# number of rows of the shingle matrix
N = 8192
# number of row of the signature matrix
k = 512
# those hold the parameters for the hashfunctions used in the min hashing step
a_array = np.zeros(k)
b_array = np.zeros(k)
for i in range(k):
    a_array[i] = int(np.random.uniform(1,N))
    b_array[i] = int(np.random.uniform(0,N))
def hash(x,a,b,n, p):
    return  ((a*x + b) % p) % n
# our prime used for min-hashing
p1 = 9241
# our prime used for lhs
p2  = 9241

# threshold for similarity
t = 0.85

# single matrix
shMatrix = np.zeros(shape=(700,N+1))

#compare shingle matrix if candidate pair
def compare_sim(v1,v2):
    if v1 == 2:
        print ("comparing")
        print(v1,v2)
    countSame = 0
    countDiff = 0
    doc1 = shMatrix[v1]
    doc2 = shMatrix[v2]
    print(shMatrix)
    print(np.sum(np.sum(shMatrix)))
    for i in range(N):
        if doc1[i] == 1 and doc2[i] == 1:
            countSame = countSame + 1
            countDiff = countDiff + 1
        elif (doc1[i] == 1 and doc2[i] == 0) or (doc1[i] == 0 and doc2[i] == 1):
            countDiff = countDiff + 1
    if countDiff == 0:
        print("BAAAAD")
        return 0
    else:
        print(countDiff)
        print(countSame)
        print(countSame/countDiff)
        return float(countSame) / countDiff


# partition signature matrix into b bands of r rows each
b = 32
r = 16

# number of buckets
m = 9999

# hash functions for lhs hashing
a_array2 = np.zeros(b*r)
b_array2 = np.zeros(b*r)
for i in range(b):
   for j in range(r):
       a_array2[i*r+j] = int(np.random.uniform(1,m))
       b_array2[i*r+j] = int(np.random.uniform(0,m))

def mapper(key, value):
    # key: None
    # value: one line of input file

    # extract the page number
    values = value.split(' ')
    pageNrStr = values.pop(0)
    pageNr = re.findall('\d+', pageNrStr)
    pageNr = int(''.join(pageNr))


    # build our shingle matrix, or rather one column of it
    ShM = np.zeros(N+1)

    for shingle in values:
        # when observing a shingle, set correspondig matrix entry to 1
        i = int(shingle)
        ShM[i]= 1

    global shMatrix
    shMatrix[pageNr] = ShM

    # column of signature Matrix, initialized to inf
    SigM = np.full(k,np.inf)
    # do min hashing with k hash funtions, see "implementing min-hashing" slide
    for i in range(N+1):
        if ShM[i] == 1:
            for j in range(k):
                SigM[j] = np.minimum(SigM[j], hash(i,a_array[j],b_array[j],N,p1))


    # for each band, hash the r values with the corresponding hash functions
    # and then aggregat in h
    for i in range(b):
        h = 0
        for j in range(r):
            h += hash(SigM[i*r+j],a_array2[r*i+j],b_array2[r*i+j],m,p2)
        h = h % m
        # we take b*10000+h as a key, the first digit will signal the band and
        # the rest will tell what bucket it got hashed into
        # as a value we emit the pageNr
        # example: emitting (10324,3) would mean, band number 1 of document 3
        # got hashed into bucket 324
        # when a group by on the key is done, we know that the 2 values
        # got one of their bands mapped to the same bucket, which makes them
        # similar by definition, therefore we can emit the pair
        yield (10000*i+h), pageNr



def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key

    # as described above, we report all pairs in values
    for v1 in values:
        for v2 in values:
            if v1 < v2 and compare_sim(v1,v2):
                yield v1, v2
