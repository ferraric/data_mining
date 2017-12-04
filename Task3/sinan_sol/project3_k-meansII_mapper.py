import numpy as np
import time

nr_centers_per_round = 200
nr_total_centers = 200
feature_dimension = 250

def mapper(key, value):
    # key: None
    # value: one line of input file
    #start = time.time()
    np.random.shuffle(value)
    # number of images
    k = value.shape[0]
    # array containing 200 centers for the result
    centers = np.zeros((nr_centers_per_round,feature_dimension))
    # pick the first center randomly (shuffled)
    centers[0,:] = value[0,:]
    # D holds the distances from datapoints to closest center
    D = np.zeros(k)
    # psi is the sum of distances from the centers (sum of values in D)
    psi = 0
    # for i in range(k):
    #     D[i] = np.linalg.norm(centers[0,:] - value[i,:])**2
    # psi = np.sum(D)
    # counting acquired centers
    r = 1
    # oversampling factor and nr of iterations (we get about l*n centers)
    n = 21
    l = 5
    # start k-means|| (initialization)
    for i in range(n):
        # print("start " + str(i))
        for j in range(k):
            # go through the dataset
            c = np.inf
            # find distance of closest center
            for m in range(r):
                dist = np.linalg.norm(value[j,:] - centers[m,:])
                if dist < c:
                    c = dist
            # store closest distance squared
            D[i] = c**2
        psi = np.sum(D)
        # go through the dataset again to sample new centers
        for i in range(k):
            # random value between 0 and 1
            ind = np.random.random_sample()
            # if probability (l*D[i]/psi) is high enough, sample as center
            if ind <= l*D[i]/psi:
                centers[r,:] = value[i,:]
                r += 1
    #end = time.time()
    #print("Mapper time: " + str((end-start)/60.0))
    yield "key", (centers,value)



def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    #start = time.time()
    # number of mapper rounds
    k = values.shape[0]
    # number of images per mapper
    l = values[0][1].shape[0]
    center_list = np.zeros((k*nr_centers_per_round,feature_dimension))
    dataset = np.zeros((k*l,feature_dimension))
    #
    result = np.zeros((nr_total_centers,feature_dimension))
    # assemble the centers and the whole dataset from the mapper
    for i in range(k-1):
        center_list[i*nr_centers_per_round:(i+1)*nr_centers_per_round,:] = values[i][0]
        dataset[i*l:(i+1)*l,:] = values[i][1]
    # array containing 200 centers for the result
    z = center_list.shape[0]
    np.random.shuffle(center_list)
    result = center_list[:200,:]


    # begin online k-means
    t = 1.0
    # loop over all images and do online k-means
    for i in range(k*l):
        # get the next image
        temp = dataset[i,:]
        # initialize parameters (min distance and that index)
        c = np.inf
        min_index = 0
        # find index of center which is closest to image
        for j in range(nr_total_centers):
            dist = np.linalg.norm(result[j,:] - temp)
            if dist < c:
                c = dist
                min_index = j
        # print(c)
        # weigh higher distances more than lower ones
        if c > 10 and i < k*l-6000:
            stepsize = 1.0
        elif c == 0 or c > 10:
            continue
        else:
            stepsize = 1.0 / t
        result[min_index,:] += stepsize*(temp - result[min_index,:])
        t += 0.003
    #end = time.time()
    #print("Reducer time: " + str((end-start)/60.0))
    yield result
