import numpy as np
import time

nr_centers_per_round = 200
nr_total_centers = 200
feature_dimension = 250

distance_threshold = 20

def mapper(key, value):
    # key: None
    # value: one line of input file

    start = time.time()

    np.random.shuffle(value)

    k = value.shape[0]
    t = 1.0

    result = np.random.randn(nr_centers_per_round,feature_dimension)

    for i in range(k):
        if i >= k:
            if i == k:
                t = 1
            l = i - k
        else:
            l = i
        # get the next image
        temp = value[l,:]
        # initialize parameters (min distance and that index)
        c = np.inf
        min_index = 0
        # find index of center which is closest to image
        for j in range(nr_centers_per_round):
            dist = np.linalg.norm(result[j,:] - temp)
            if dist < c:
                c = dist
                min_index = j
        #print(c)
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

    # attempt standard k-means
    # store cluster assignments
    # z = np.full(k,-1,dtype=np.int32)
    # for m in range(5):
    #     print(m)
    #     # find closest centers
    #     changes = 0
    #     for i in range(k):
    #         min_index = 0
    #         # get the next image
    #         temp = value[i,:]
    #         # initialize parameters (min distance and that index)
    #         c = np.inf
    #         # find index of center which is closest to image
    #         for j in range(nr_total_centers):
    #             dist = np.linalg.norm(result[j,:] - temp)
    #             if dist < c:
    #                 c = dist
    #                 min_index = j
    #         if min_index != z[i]:
    #             changes += 1
    #         z[i] = min_index
    #     print (z[1:100])
    #     print("changes:")
    #     print(changes)
    #     # update center as mean of assigned data points
    #
    #     #array that holds how many data points assigned to which center
    #     center_count = np.zeros(nr_total_centers)
    #     # matrix that holds new centers
    #     new_centers = np.zeros((nr_total_centers,feature_dimension))
    #     for j in range(k):
    #         center_count[z[j]] += 1
    #         new_centers[z[j], :] += value[j,:]
    #     for l in range(nr_total_centers):
    #         if center_count[l] == 0:
    #             pass
    #             #print(new_centers[l,:])
    #         else:
    #             new_centers[l,:] = new_centers[l,:]/center_count[l]
    #     result = new_centers


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
    end = time.time()
    print("Mapper time: " + str(end-start))
    yield "key", result



def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    start = time.time()
    np.random.shuffle(values)
    # number of images
    k = values.shape[0]
    t = 1.0
    print(k)
    # attempt at farthest points heuristic

    # result = np.zeros((nr_total_centers,feature_dimension))
    # # choose 200 points with farthest point heuristic
    # # first point chosen at random
    # init_index = np.random.choice(k, 1, replace=False)
    # # prevents indexed from being used twice
    # index_set = np.array([init_index])
    # curr_initial_center = values[init_index,:]
    # for i in range(nr_total_centers):
    #     result[i,:] = curr_initial_center
    #     max_dist = 0
    #     farthest_index = np.inf
    #     # find farthes point from current to initialize next cluster
    #     for j in range(k):
    #         dist = np.linalg.norm(curr_initial_center - values[j,:])
    #         if dist > max_dist and j not in index_set:
    #             max_dist = dist
    #             farthest_index = j
    #             np.append(index_set,j)
    #     curr_initial_center = values[farthest_index,:]

    # array containing 200 centers for initialization of the result
    result = np.random.randn(nr_total_centers,feature_dimension)

    # attempt standard k-means
    # store cluster assignments
    z = np.full(k,-1,dtype=np.int32)
    changes = -1
    for m in range(100):
        if changes == 0:
            break
        print(m)
        # find closest centers
        changes = 0
        for i in range(k):
            min_index = 0
            # get the next image
            temp = values[i,:]
            # initialize parameters (min distance and that index)
            c = np.inf
            # find index of center which is closest to image
            for j in range(nr_total_centers):
                dist = np.linalg.norm(result[j,:] - temp)
                if dist < c:
                    c = dist
                    min_index = j
            if min_index != z[i]:
                changes += 1
            z[i] = min_index
        print (z[1:100])
        print("changes:")
        print(changes)
        # update center as mean of assigned data points

        # #array that holds how many data points assigned to which center
        # center_count = np.zeros(nr_total_centers)
        # # matrix that holds new centers
        # new_centers = np.zeros((nr_total_centers,feature_dimension))
        # for j in range(k):
        #     center_count[z[j]] += 1
        #     new_centers[z[j], :] += values[j,:]
        # for l in range(nr_total_centers):
        #     if center_count[l] == 0:
        #         pass
        #         #print(new_centers[l,:])
        #     else:
        #         new_centers[l,:] = new_centers[l,:]/center_count[l]
        # result = new_centers

        for l in range(nr_total_centers):
            new_center = np.zeros(feature_dimension)
            count = 0
            for j in range(k):
                if z[j] == l:
                    count += 1
                    new_center += values[j,:]
            if count == 0:
                pass
            else:
                new_center = new_center/count
            result[l,:] = new_center



    # # loop over all images and do online k-means
    # for i in range(k):
    #     if i >= k:
    #         if i == k:
    #             t = 1
    #         l = i - k
    #     else:
    #         l = i
    #     # get the next image
    #     temp = values[l,:]
    #     # initialize parameters (min distance and that index)
    #     c = np.inf
    #     min_index = 0
    #     # find index of center which is closest to image
    #     for j in range(nr_total_centers):
    #         dist = np.linalg.norm(result[j,:] - temp)
    #         if dist < c:
    #             c = dist
    #             min_index = j
    #     #print(c)
    #     # weigh higher distances more than lower ones
    #     if c > 10:
    #         stepsize = 1.0
    #     # elif c > 10:
    #     #     stepsize = 0.5
    #     # elif c > 7:
    #     #     stepsize = 0.25
    #     # elif c > 6:
    #     #     stepsize = 0.2
    #     # elif c > 5.5:
    #     #     stepsize = 0.15
    #     # elif c > 5:
    #     #     stepsize = 0.1
    #     # elif c > 4.5:
    #     #     stepsize = 0.05
    #     # elif c > 4:
    #     #     stepsize = 0.01
    #     else:
    #         stepsize = 1.0 / t
    #     result[min_index,:] += stepsize*(temp - result[min_index,:])
    #     t += 0.005
    end = time.time()
    print("Reducer time: " + str(end-start))
    yield result
