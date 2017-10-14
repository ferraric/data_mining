import numpy as np

def mapper(key, value):
    # key: None
    # value: one line of input file (one document)

    # split up the document into the shingles and remove first occurence (document name)
    shingle_list = value.split()
    doc_id = shingle_list.pop(0)[5:]
    # generate hash function h(x) = ((a*x + b) % c) % 8092 where a,b < max(x) && isprime(c) == True and c > x
    def hash(x):
        a = np.random.uniform(0,8192)
        b = np.random.uniform(0,8192)
        c = 8209
        return ((a*x + b) % c) % 8192

    # hash the value given as argument n times and do min hashing to obtain signature "column" for document
    n = 16
    signature = []
    for i in range(0,n):
        hash_values = []
        for shingle in shingle_list:
            hash_values.append(hash(shingle))
        # get the min hash for the i-th hash function
        signature.append(min(hash_values))
    # yield the signature column as the key and the document id as the value
    yield signature, doc_id


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    pairs = zip(values[::2], values[1::2])
    for i in pairs:
        yield i[0], i[1]
