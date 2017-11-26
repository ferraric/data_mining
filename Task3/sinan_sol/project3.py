import numpy as np
import time

nr_centers = 200
feature_dimension = 250

def mapper(key, value):
    # key: None
    # value: one line of input file
    start = time.time()
    # initialize the centers randomly from normal distribution
    # centers = np.random.multivariate_normal(np.zeros(feature_dimension),np.ones(feature_dimension,feature_dimension),nr_centers)
    centers = np.random.randn(200,250)
    # get number of images provided to mapper
    nr_images = value.shape[0]
    #do online k-means
    for i in range(nr_images):
        im = value[i,:]
        # check which center is closest
        c = np.inf
        min_dist = 0
        for j in range(nr_centers):
            dist = np.linalg.norm(centers[j,:]-im)
            if dist < c:
                min_dist = dist
                min_index = j
        stepsize = min_dist / (i+1)
        centers[min_index,:] += stepsize*(im - centers[min_index,:])

    end = time.time()
    print("Mapper time: " + str(end-start))
    yield "key", centers  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.

    # k = values.shape[0]
    # result = np.zeros((200,250))
    # for centers in values:
    #     result += centers
    # result /= k
    # yield result
    yield values
