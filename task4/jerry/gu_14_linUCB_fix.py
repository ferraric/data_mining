import numpy as np

# Articles feature size
d = 6

# LinUCB inits
M = {}
M_inv = {}
b = {}
w = {}

# Learning rate
r0 = -2
r1 = 28.4
sigma = 0.1
alpha = 1 + np.sqrt(np.log(2 / sigma) / 2)

print "r0=" + str(r0)
print "r1=" + str(r1)
print "sigma=" + str(sigma)
print "alpha=" + str(alpha)


# User at round t
user_t = None

# Chosen article id at round t
neo = None

# [articleId : articleFeatures len 6] - 80 articles total
def set_articles(articles):
    pass


# reward - y_t
# reward of choice neo in round t
def update(reward):
    global user_t, neo
    
    if reward != -1:
        scaled_reward = r1 if reward == 1 else r0
        M[neo] += np.dot(user_t, user_t)
        M_inv[neo] = np.linalg.inv(M[neo])
        b[neo] += scaled_reward * user_t
        w[neo] = np.dot(M_inv[neo], b[neo])
    

# time - int timestamp
# user_features - len 6 (user_t)
# choices - articles len 20 -> 20 arms of bandit
def recommend(timestamp, user_features, choices):
    global user_t, neo
    
    user_t = np.array(user_features)
    
    # On each round t for each choice we observe
    # K=20 feature vectors (articles) with size d=6
    
    max_ucb = None
    neo = None
    
    for article_id in choices:
        
        if article_id not in M:
            M[article_id] = np.identity(d)
            M_inv[article_id] = np.linalg.inv(M[article_id])
            b[article_id] = np.zeros(d)
            w[article_id] = np.dot(M_inv[article_id], b[article_id])
            neo = article_id
            break
        
        ucb_x = np.dot(w[article_id], user_t)
        
        temp = (user_t.dot(M_inv[article_id])).dot(user_t)
        
        ucb_x += (alpha * np.sqrt(temp))
        
        if ucb_x > max_ucb or max_ucb == None:
            neo = article_id
            max_ucb = ucb_x
    
    
    # return the choice we made
    #return np.random.choice(choices)
    return neo

