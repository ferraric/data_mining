import numpy as np
import time

# m is the number of dimensions in our final representation
m = 8000
# d is the number of input dimensions
d = 400
sqr = np.sqrt(2.0 / m)

# draw m samples from p(omega), and b
b = np.random.uniform(0,2.0*np.pi,m).astype('float32')
# every row is one sample
#omega = np.random.multivariate_normal(np.zeros(d),np.identity(d),m)
#omega = np.random.laplace(size=(m,d)).astype('float32')
omega = np.random.exponential(scale=1,size=(m,d))
#omega = np.random.standard_cauchy(size=(m,d))


def fourier_transform(x):
    fourier_feature = np.dot(omega, x)
    fourier_feature += b
    np.cos(fourier_feature, fourier_feature)
    fourier_feature *= sqr

    return fourier_feature.astype('float32')

def transform(X):
    if len(X.shape) == 1:
        return fourier_transform(X)
    else:
        # Make sure this function works for both 1D and 2D NumPy arrays.
        return np.array(map(lambda x: fourier_transform(np.array(x)), X))

# def transform(X):
#     # Make sure this function works for both 1D and 2D NumPy arrays.
#     if X.ndim == 1:
#         X_new = np.zeros(m)
#         for j in range(m):
#             X_new[j] = np.cos(np.dot(omega[j,:],X)+b[j])
#         X_new = np.sqrt(2.0/m)*X_new
#     elif X.ndim ==2:
#         n = X.shape[0]
#
#         X_new = np.zeros((n,m))
#
#         for i in range(n):
#             for j in range(m):
#                 X_new[i,j] = np.cos(np.dot(omega[j,:],X[i,:])+b[j])
#
#         X_new = np.sqrt(2.0/m)*X_new
#     else:
#         X_new = 0
#         print("ERROR, transform method can only deal with 1d or 2d arrays")
#
#     return X_new


def mapper(key, value):
    # key: None
    # value: one line of input file
    start = time.time()
    images = value
    n = len(images)
    # numpy matrix to hold our values
    data = np.array(map(lambda x: np.array(x.split(" "), dtype='float32'), value))
    np.random.shuffle(data)

    #initialize w
    w = np.zeros(m)

    # regularization parameter
    C = 100

    stepsize = 1.0
    t = 1.0
    for d in data:
        y = d[0]
        x = transform(d[1:])
        stepsize = 2/np.sqrt(t)
        t += 1.0
        if y*np.dot(w,x) >= 1:
            w = w - stepsize/n * w
            #pass
        else:
            w = w - stepsize*(w*1.0/n-C*y*x)
            # w = w - stepsize*y*x

    # X = np.zeros((n,d))
    # y = np.zeros(n)
    # i = 0
    # for im in images:
    #     features = im.split(' ')
    #     y[i] = features.pop(0)
    #     X [i,:] = features
    #     i += 1
    #
    #
    # #initialize w
    # w = np.zeros(m)
    # #w = np.random.normal(scale=1.0, size=m)
    # # regularization parameter
    # C = 10000
    #
    # # generate random permutation of 1..n
    # perm = np.random.permutation(n)
    #
    #
    # stepsize = 1.0
    # t = 1.0
    # for i in perm:
    #     X_new = transform(X[i,:])
    #     stepsize = 0.1
    #     t += 1.0
    #     if y[i]*np.dot(w,X_new) >= 1:
    #         w = w - stepsize/n * w
    #         #pass
    #     else:
    #         w = w - stepsize*(w*1.0/n-C*y[i]*X_new)
    #         # w = w - stepsize*y[i]*X_new

    end = time.time()
    print("Mapper time: " + str(end - start))

    yield "key", w  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    k = len(values)
    w = np.zeros(m)
    for v in values:
        w += v
    w = w * 1.0/k

    yield w
