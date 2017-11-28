import numpy as np
import time

nr_centers_per_round = 500
nr_total_centers = 200
feature_dimension = 250

distance_threshold = 20

def mapper(key, value):
    # key: None
    # value: one line of input file
    np.random.shuffle(value)
    # start = time.time()
    # # initialize the centers randomly from normal distribution
    # centers = np.random.randn(nr_centers_per_round,feature_dimension)
    # # get number of images provided to mapper
    # nr_images = value.shape[0]
    # stepsize = 0.5
    # #do online k-means
    # for i in range(nr_images):
    #     im = value[i,:]
    #     # check which center is closest
    #     c = np.inf
    #     min_dist = 0
    #     min_index = 0
    #     for j in range(nr_centers_per_round):
    #         dist = np.linalg.norm(centers[j,:]-im)
    #         if dist < c:
    #             c = dist
    #             min_index = j
    #     centers[min_index,:] += stepsize*(im - centers[min_index,:])
    # end = time.time()
    # print("Mapper time: " + str(end-start))
    # yield "key", centers
    yield "key", value



def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    start = time.time()
    np.random.shuffle(values)
    # number of images
    k = values.shape[0]
    t = 1.0
    # array containing 200 centers for initialization of the result
    result = np.random.randn(nr_total_centers,feature_dimension)
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
        print(c)
        # weigh higher distances more than lower ones
        if c > 10:
            stepsize = 1.0
        # elif c > 10:
        #     stepsize = 0.5
        # elif c > 7:
        #     stepsize = 0.25
        # elif c > 6:
        #     stepsize = 0.2
        # elif c > 5.5:
        #     stepsize = 0.15
        # elif c > 5:
        #     stepsize = 0.1
        # elif c > 4.5:
        #     stepsize = 0.05
        # elif c > 4:
        #     stepsize = 0.01
        else:
            stepsize = 1.0 / t
        result[min_index,:] += stepsize*(temp - result[min_index,:])
        t += 0.005
    end = time.time()
    print("Reducer time: " + str(end-start))
    yield result
