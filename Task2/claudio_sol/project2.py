import numpy as np

# m is the number of dimensions in our final representation
m = 1000
# d is the number of input dimensions
d = 400

# draw m samples from p(omega), and b
b = np.random.uniform(0,2*np.pi,m)
# every row is one sample
omega = np.random.multivariate_normal(np.zeros(d),np.identity(d),m)

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.

    if X.ndim == 1:
        X_new = np.zeros(m)
        for j in range(m):
            X_new[j] = np.cos(np.dot(omega[j,:],X)+b[j])
        X_new = np.sqrt(2/m)*X_new
        print("GOTCHA!")

    elif X.ndim ==2:
        n = X.shape[0]

        X_new = np.zeros((n,m))

        for i in range(n):
            for j in range(m):
                X_new[i,j] = np.cos(np.dot(omega[j,:],X[i,:])+b[j])

        X_new = np.sqrt(2/m)*X_new
    else:
        X_new = 0
        print("ERROR, transform method can only deal with 1d or 2d arrays")

    return X_new


def mapper(key, value):
    # key: None
    # value: one line of input file

    images = value
    n = len(images)
    # numpy matrix to hold our values
    X = np.zeros((n,d))
    y = np.zeros(n)
    i = 0
    for im in images:
        features = im.split(' ')
        y[i] = features.pop(0)
        X [i,:] = features
        i += 1

    X_new = transform(X)
    #initialize w
    w = np.zeros(m)
    # regularization parameter
    C = 1

    # generate random permutation of 1..n
    perm = np.random.permutation(n)

    stepsize = 1
    t = 1
    for i in perm:
        stepsize = 1/t
        t += 1
        if y[i]*np.dot(w,X_new[i,:]) >= 1:
            w = w - stepsize/n * w
        else:
            w = w - stepsize*(w*1/n-C*y[i]*X_new[i,:])


    yield "key", w  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    k = len(values)
    w = 0
    for v in values:
        w += v
    w = w * 1/k

    yield w
