#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:15:29 2017

@author: claudioferrari
"""

def mapper(key, value):
    # key: None
    # value: one line of input file
    
    # assumed that input is a signature matrix
    
    # we split it into b band of r values each, with r*b<=1024
    #for each band we choose r hashfunctions and put the r values through them
    # aggregating the results we should get a hash bucket number
    # we emit the bucket number as key and the document number as value
    
    #nr of rows in signature matrix
    n = len(value)
    
    #get a document with shingle values, split it up into bands
    #hash the bands, pass it on
    if False:
        yield "key", "value"  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    
    #possibly the key could be one hash bucket, and values are all documents
    #that go into that buckets, so we would emit all pairs of documents in values
    
    #get a signature matrix or column of it and output two columns if they are similar enough
    #output is the column numbers
    if False:
        yield "key", "value"  # this is how you yield a key, value pair