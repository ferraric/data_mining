import numpy as np
import math

# Feature size dimension
d = 6

# Articles
art = {}

# HybridUCB init
# statics
A0 = np.identity(d)
A0_inv = np.identity(d)
b0 = np.zeros(d)
beta = np.zeros(d)

# essentials
A = {}
B = {}
b = {}
A_inv = {}

# compositions
w = {}
B_x_A_inv_x = {}
# term 1
art_A0_inv_art = {}
# term 2
art_A0_inv_B_A_inv = {}
# term 3 and 4
A_inv_B_A0_inv_B_A_inv = {}



# Learning rate
r0 = -0.1
r1 = 3.0
alpha = 0.1

print "r0=" + str(r0)
print "r1=" + str(r1)
print "alpha=" + str(alpha)


# User at round t
user_t = 0

# Chosen article id at round t
neo = 0

# [articleId : articleFeatures len 6] - 80 articles total
def set_articles(articles):
    for key, value in articles.iteritems():
        art[key] = np.array(value)


# reward - y_t
# reward of choice neo in round t
def update(reward):
    global A0, b0, beta, A0_inv

    if reward != -1:
        scaled_reward = r1 if reward == 1 else r0

        # statics
        A0 += np.dot(B_x_A_inv_x[neo], B[neo])
        b0 += np.inner(B_x_A_inv_x[neo], b[neo])

        # article dependent
        A[neo] += np.outer(user_t, user_t)
        B[neo] += np.outer(user_t, art[neo])
        b[neo] += scaled_reward * user_t
        A_inv[neo] = np.linalg.inv(A[neo])

        # statics again
        B_x_A_inv_x[neo] = np.dot(B[neo].T, A_inv[neo])

        A0 += np.outer(art[neo], art[neo])
        A0 -= np.dot(B_x_A_inv_x[neo], B[neo])

        b0 += scaled_reward * art[neo]
        b0 -= np.inner(B_x_A_inv_x[neo], b[neo])

        A0_inv = np.linalg.inv(A0)
        beta = np.inner(A0_inv, b0)

        # update caches
        w[neo] = np.dot(A_inv[neo], (b[neo] - np.inner(B[neo], beta)))

        temp = np.dot(np.dot(A0_inv, B[neo].T), A_inv[neo])

        # term 1
        art_A0_inv_art[neo] = np.inner(np.dot(art[neo], A0_inv), art[neo])

        # term 2
        art_A0_inv_B_A_inv[neo] = 2 * np.dot(art[neo], temp)

        # term 3 and 4
        A_inv_B_A0_inv_B_A_inv[neo] = A_inv[neo] + np.dot(np.dot(A_inv[neo], B[neo]), temp)



# time - int timestamp
# user_features - len 6 (user_t)
# choices - articles len 20 -> 20 arms of bandit
def recommend(timestamp, user_features, choices):
    global user_t, neo

    user_t = np.array(user_features)

    # On each round t for each choice we observe
    # 20 feature vectors (articles) and one user

    max_ucb = 0
    neo = 0

    for article_id in choices:
        article_x = art[article_id]

        if article_id not in A:
            # initialize essentials
            A[article_id] = np.identity(d)
            B[article_id] = np.zeros((d,d))
            b[article_id] = np.zeros(d)
            A_inv[article_id] = np.identity(d)

            w[article_id] = np.zeros(6)
            B_x_A_inv_x[article_id] = np.zeros((6,6))

            # term 1
            art_A0_inv_art[article_id] = np.inner(np.dot(article_x, A0_inv), article_x)

            # term 2
            art_A0_inv_B_A_inv[article_id] = np.zeros(6)

            # term 3 and 4
            A_inv_B_A0_inv_B_A_inv[article_id] = np.identity(6)


        # s_t calculation
        s_t = art_A0_inv_art[article_id]

        # term 2
        s_t -= np.inner(art_A0_inv_B_A_inv[article_id], user_t)

        # term 3 and 4
        s_t += np.inner(np.dot(user_t, A_inv_B_A0_inv_B_A_inv[article_id]), user_t)

        # ucb_x calc
        ucb_x = np.inner(article_x, beta)
        ucb_x += np.inner(user_t, w[article_id])
        ucb_x += alpha * np.sqrt(s_t)

        if ucb_x > max_ucb:
            neo = article_id
            max_ucb = ucb_x


    #print("recommend time: " + str(time.time() - start))
    # return the choice we made
    #return np.random.choice(choices)
    return neo


#def jacc_similarity(arr1, arr2):
#    return float(len(np.intersect1d(arr1, arr2))) / float(len(np.union1d(arr1, arr2)))
