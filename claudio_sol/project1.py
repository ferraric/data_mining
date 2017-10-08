#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:15:29 2017

@author: claudioferrari
"""

def mapper(key, value):
    # key: None
    # value: one line of input file
    if False:
        yield "key", "value"  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    if False:
        yield "key", "value"  # this is how you yield a key, value pair