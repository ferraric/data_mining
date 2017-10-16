import re
import numpy as np

# number of possible shingle values
N = 8192
# number hash functions
k = 8
# those hold the parameters for the hashfunctions used in the min hashing step
a_array = np.zeros(k)
b_array = np.zeros(k)
for i in range(k):
    a_array[i] = int(np.random.uniform(1,N))
    b_array[i] = int(np.random.uniform(0,N))
def hash(x,a,b,n,p):
    return ((a*x + b) % p) % n
# our prime used for min-hashing
p = 8209
# threshold for jaccard similarity
t = 0.85

def mapper(key, value):
    # key: None
    # value: one line of input file

    # extract the page number
    values = value.split(' ')
    pageNrStr = values.pop(0)
    pageNr = re.findall('\d+', pageNrStr)
    pageNr = int(''.join(pageNr))

    # do the min-hashing, and get the documents that hash to the same bucket for the same hash function
    for i in range(k):

        currHash = float("inf")

        for shingle in values:
            hashValue = hash(int(shingle),a_array[i],b_array[i],N,p)
            if hashValue < currHash:
                currHash = hashValue
        # also pass values array to do another pass for the similarity in the reducer
        yield 10000*i+currHash, (pageNr,values)


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key

    # as described above, we report all pairs in values with jaccard similarity > 0.85 = t
    for v1 in values:
        for v2 in values:
            if v1[0] != v2[0]:
                set_sh1 = set(v1[1])
                set_sh2 = set(v2[1])
                intersection = len(set_sh1.intersection(set_sh2))
                union = len(set_sh1.union(set_sh2))
                if (float(intersection) / union) > t:
                    if v1[0] < v2[0]:
                        yield v1[0], v2[0]
                    else:
                        yield v2[0], v1[0]
