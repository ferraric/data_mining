#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:15:29 2017

@author: claudioferrari
"""
import re
import numpy as np


N = 8192
k = 128
# those hold the parameters for the hashfunctions
a_array = np.zeros(k) 
b_array = np.zeros(k) 
for i in range(k):
    a_array[i] = int(np.random.uniform(0,8192))
    b_array[i] = int(np.random.uniform(0,8192))
def hash(x,a,b,n):
    return  ((a*x + b) % p) % n
# our prime used for hashing
p = 9241
   

# partition signature matrix into b bands of r rows each           
b = 8
r = 16

# number of buckets
m = 10

# hash functions for lhs hashing
for i in range(b):
   a_array2 = np.zeros(b*r)
   b_array2 = np.zeros(b*r)
   for j in range(r):
       a_array2[i*r+j] = int(np.random.uniform(0,k))
       b_array2[i*r+j] = int(np.random.uniform(0,k))

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
    
    counter = 0
    for shingle in values:
        i = int(shingle)
        ShM[i]= 1
    
    
    # column of signature Matrix
    SigM = np.full(k,np.inf)
    # do min hashing with k hash funtions
    for i in range(N+1):
        if ShM[i] == 1:
            for j in range(k):
                SigM[j] = np.minimum(SigM[j], hash(i,a_array[j],b_array[j],N))
                
    
    
    if pageNr == 0:
        print(ShM[0:100])
        print(np.shape(ShM))
        print(np.shape(SigM))
        print(SigM)
    # for each band, choose r hash functions independently and hash the r
    # values to m buckets
    for i in range(b):
        h = 0
        for j in range(r):
            h += hash(SigM[i*r+j],a_array2[r*i+j],b_array2[r*i+j],m)
        h = h % m
        # we take b*100+h as a key, the first digit will signal the band and
        # the second and third will tell what bucket
        # as a value we emit the pageNr
        #print((100*b+h), pageNr)
        yield (100*b+h), pageNr
    
     
    
    
    if False:
        yield "key", "value"  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    
    #possibly the key could be one hash bucket, and values are all documents
    #that go into that buckets, so we would emit all pairs of documents in values
    
    #get a signature matrix or column of it and output two columns if they are similar enough
    #output is the column numbers
    for v1 in values:
        for v2 in values:
            if v1 < v2:
                yield v1, v2
    
#    if False:
#        yield "key", "value"  # this is how you yield a key, value pair