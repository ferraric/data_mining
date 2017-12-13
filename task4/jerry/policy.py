import numpy as np

# General idea: Create a matrix with user's(defined by the user features), where
# each row corresponds to one specific user. You then have a list of all possible
# article id's in increasing order. In each column, so for each specific article
# you then have an Expected-Return. We will use this expected return to make recommendations
# for future articles. Because we will get choices of articles and then we will
# in a first step take the argmax of the expected reward of
# all articles for that specific users. Later on we will implement some more
# complicated functions thatn argmax which perform better. But for now, we'll
# stick to the simple stuff. Also note that some times with a probability epsilon,
# we will select a random(in a better version not only random) other article, in then
# hope that this will give us a better long term value(= total reward).

total_reward = 0
articles_matrix = np.array([])
selected_article = 0
last_user = 0
global_time = 0
found_user_article = False
found_user = False
user_preferences = {} # create dictionary
# Note, Key are 6 Columns = User features, content is again dictionary of
# 80 columns = all article ID's -> (counted them)
# Also initially the expected reward for all actions is 0

# articles are sorted after their ID. In the actual rows we have the according
# features. I think this is here to pass the articles matrix in this file.
def set_articles(articles):
    #print("In set_articles")
    global user_preferences
    articles_matrix = np.copy(articles)
    # initialize dicitonary with dummy variable such that loop will work
    user_preferences = {(0,0,0,0,0,0):{0:0}}
    return articles


def update(reward):
    #print("In update")
    # Here user preferences are updated
    # we assume global times starts from 1
    global global_time
    global user_preferences
    global last_user
    global selected_article
    global found_user_article
    global found_user
    global total_reward
    global_time = global_time + 1
    new_expected_reward = 0

    if(found_user_article == True):
        #print("Found user and article")
        cur_art = user_preferences[tuple(last_user)]
        #if(selected_article in cur_art):
        #print(cur_art)
        #print(selected_article)
        #print(last_user)
        expected_reward = cur_art[selected_article]
        new_expected_reward = expected_reward + 1.0/global_time*(reward-expected_reward)
        cur_art[selected_article] = new_expected_reward
        #else:
            #new_expected_reward = 1/global_time*reward
            #user_preferences[tuple(last_user)] = {selected_article:new_expected_reward}

    else:
        # Q_0 = 0
        new_expected_reward = 1.0/global_time*reward
        if(found_user == False):
            #print("Found nothing")
            # We create a new article entry for a new user
            user_preferences[tuple(last_user)] = {selected_article:new_expected_reward}
        else:
            #print("Found only user")
            # We create a new article entry for a already existing use --> append dict.
            user_preferences[tuple(last_user)][selected_article] = new_expected_reward

    total_reward += new_expected_reward
    #print("Total reward: ", total_reward)
    return total_reward

# Here we have to recommed an article that the user liked but also use some
# exploration where we just recommend something different.
# When looking up the data we can see that the choices are all(as far as I saw)
# article ID's that appear in the webscope-articles.txt file.
def recommend(time, user_features, choices):
    #print("in recommend")
    global global_time
    global user_preferences
    global last_user
    global selected_article
    global found_user_article
    global found_user
    epsilon = 0.9
    found_user_article = False
    found_user = False
    max_expected_reward = -float('inf')
    global_time = time
    last_user = user_features
    exploit = False
    if(np.random.uniform(0,1) < epsilon):
        exploit = True

    for key, value in user_preferences.items():
        if(key == tuple(last_user) and exploit):
            #print("User match")
            found_user = True
            # We already have information about the preferences of this user
            # return indices of article of maximal expected reward
            for art in choices:
                if(art in value):
                    #print("article in value")
                    #print(art)
                    #print(value)
                    #print(last_user)
                    # Note value is a dictionary with key = art_id and value = expected reward
                    if(value[art] >= max_expected_reward):
                        max_expected_reward = value[art]
                        selected_article = art
                    #print(selected_article)
                    # only take the article if we have a positive expected reward, else take a random one
                    if(value[art] >= 0):
                        found_user_article = True
                    else:
                        found_user_article = False

    if(found_user_article == False):
        #print("no user match, random article")
        # We have no info about preferences for this user ye yet
        sample = np.random.choice(choices,1)
        # select uniformly_random an article
        selected_article = sample[0]
        # insert user and article into our dictionary
        # define last_action tuple...
    return selected_article
