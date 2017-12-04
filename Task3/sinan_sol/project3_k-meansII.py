import numpy as np
import time

nr_total_centers = 200
feature_dimension = 250

def mapper(key, value):
    # key: None
    # value: one line of input file
    np.random.shuffle(value)
    yield "key", value


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    start = time.time()
    #np.random.shuffle(values)
    # number of images
    k = values.shape[0]
    # array containing 200 centers for initialization of the result (random)
    result = np.zeros((nr_total_centers,feature_dimension))
    # pick the first center randomly (shuffled)
    result[0,:] = values[np.random.choice(range(k)),:]
    # D holds the distances from datapoints to closest center
    D = np.zeros(k)
    # psi is the sum of distances from the centers (sum of values in D)
    psi = 0
    # counting acquired centers
    r = 1
    # oversampling factor and nr of iterations (we get about l*n centers)
    n = 7
    l = 30
    # start k-means|| (initialization)
    start_barbar = time.time()
    for i in range(n):
        for j in range(k):
            # go through the dataset
            c = np.inf
            # find distance of closest center
            for m in range(r):
                dist = np.linalg.norm(result[m,:] - values[j,:])
                if dist < c:
                    c = dist
            # store closest distance squared
            D[j] = c**2
        psi = np.sum(D)
        # go through the dataset again to sample new centers
        for p in range(k):
            # random value between 0 and 1
            ind = np.random.random_sample()
            # if probability (l*D[i]/psi) is high enough, sample as center
            if ind <= l*D[p]/psi:
                result[r,:] = values[p,:]
                r += 1
                if r == nr_total_centers:
                    break
        if r == nr_total_centers:
            break
    end_barbar = time.time()
    print(r)
    print("Initialization done. Time: " + str((end_barbar-start_barbar)/60.0))
    # begin online k-means
    t = 1.0
    # loop over all images and do online k-means
    for i in range(k):
        # get the next image
        temp = values[i,:]
        # initialize parameters (min distance and that index)
        c = np.inf
        min_index = 0
        # find index of center which is closest to image
        for j in range(nr_total_centers):
            dist = np.linalg.norm(result[j,:] - temp)
            if dist < c:
                c = dist
                min_index = j
        # weigh higher distances more than lower ones
        if c > 10 and i < k*l-5000:
            stepsize = 1.0
        elif c == 0:
            continue
        else:
            stepsize = 1.0 / t
        result[min_index,:] += stepsize*(temp - result[min_index,:])
        t += 0.003
    end = time.time()
    print("Reducer time: " + str((end-start)/60.0))
    yield result
