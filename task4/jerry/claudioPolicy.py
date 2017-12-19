import numpy as np

# is over articles not over users
#M = {}
M_inv = {}
b = {}
w = {}

# M,B and W are all over articles

# Learning rate
r0 = -2
r1 = 28.4

delta = 0.1
alpha = 1.0 + np.sqrt(np.log(2/delta)/2)
#user and article at round t
user_t = None
article_t = None


k = 6
# articles is a python dictionary
def set_articles(articles):
#     global my_articles
# #    print(articles)
#     my_articles = articles
#     articles_matrix = np.copy(articles)
#     print(len(articles))
    pass



def update(reward):
    global user_t
    global article_t
    global M_inv
    global b
    global w
    # print("compare1:")
    # print(np.linalg.inv(M[article_t]))
    # print(M_inv[article_t])
    #
    if reward != -1:
        scaled_reward = r1 if reward == 1 else r0
        # M[article_t] += np.outer(user_t,user_t)
        # M_inv[article_t] = np.linalg.inv(M[article_t])
        #update with sherman morrison woodbury
        top = np.outer(np.dot(M_inv[article_t],user_t),np.dot(user_t,M_inv[article_t]))
        bottom = (1.0 + np.dot(user_t,np.dot(M_inv[article_t],user_t)))
        M_inv[article_t] -= top/bottom
        # print("compare2:")
        # print(np.linalg.inv(M[article_t]))
        # print(M_inv[article_t])
        b[article_t] += scaled_reward*user_t
    pass


def recommend(time, user_features, choices):
    global user_t
    global article_t
    global my_articles
    global M_inv
    global b
    global w
    z = np.asarray(user_features)


    best_art = None
    # min_dist = float('inf')
    max_ucb = -float('inf')
    for art in choices:
        if not art in M_inv:
            #M[art] = np.identity(k)
            M_inv[art] = np.identity(k)
            b[art] = np.zeros(k)
        # M_inv[art] = np.linalg.inv(M[art])
        w[art] = np.dot(M_inv[art],b[art])
        ucb = np.dot(w[art],z) + alpha * np.sqrt(np.dot(z,np.dot(M_inv[art],z)))
        if ucb > max_ucb:
            max_ucb = ucb
            best_art = art
        # curr = np.asarray(my_articles[art])
        # dist = np.linalg.norm(user_t - curr)
        # if dist < min_dist:
        #     min_dist = dist
        #     best_art = art

    # epsilon = 0.1
    # if(np.random.uniform(0,1) < epsilon):
    #     pass
    #     #exploring
    # else:
    #     pass
        #exploiting

    article_t = best_art
    user_t = z
    return best_art
