# MIT License
#
# Copyright (c) 2016 las.inf.ethz.ch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Evaluation framework for the Bandit setup (Task4, DM2016)"""

import argparse
import io
import imp
import logging
import numpy as np
import resource
import signal
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def process_line(policy, logline):
    # Gets article number that was chosen by the random selection for specific
    # User.
    chosen = int(logline.pop(7))
    # either 0 or 1, That means if reward = 1 the user clicked on this articles
    # and if reward = 0 the user didn't click on it.
    reward = int(logline.pop(7))
    time = int(logline[0])
    user_features = [float(x) for x in logline[1:7]]
    articles = [int(x) for x in logline[7:]]
    return reward, chosen, policy.recommend(time, user_features, articles)


def evaluate(policy, input_generator):
    score = 0.0
    impressions = 0.0
    n_lines = 0.0
    for line in input_generator:
        n_lines += 1
        reward, chosen, calculated = process_line(
            policy, line.strip().split())
        if calculated == chosen:
            policy.update(reward)
            score += reward
            impressions += 1
        else:
            # This says that if my
            # predicted article does not match the one they selected randomly
            # I will be penalized with -1. But thinking about it, makes sense,
            # In the future I will not want to recommend him this article, not
            # because he didn't like it, but because we have no information about
            # if he likes it nor not. Because in the log file the user has another
            # article associated to it(that was selected randomly). But note That
            # This update of reward of -1 will not affect the public score.
            # But you can use it to affect your internal score.
            policy.update(-1)
    if impressions < 1:
        logger.info("No impressions were made.")
        return 0.0
    else:
        score /= impressions
        logger.info("CTR achieved by the policy: %.5f" % score)
        return score


def import_from_file(f):
    """Import code from the specified file"""
    mod = imp.new_module("mod")
    exec f in mod.__dict__
    return mod


def run(source, log_file, articles_file):
    policy = import_from_file(source)
    articles_np = np.loadtxt(articles_file)
    articles = {}
    for art in articles_np:
        # Here articles are put into row position according to their ID
        # All the rest is put in as row. Note it is a dictionary.
        articles[int(art[0])] = [float(x) for x in art[1:]]
    policy.set_articles(articles)
    with io.open(log_file, 'rb', buffering=1024*1024*512) as inf:
        return evaluate(policy, inf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'log_file', help='File containing the log.')
    parser.add_argument(
        'articles_file', help='File containing the article features.')
    parser.add_argument(
        'source_file', help='.py file implementing the policy.')
    parser.add_argument(
        '--log', '-l', help='Enable logging for debugging', action='store_true')
    args = parser.parse_args()
    with open(args.source_file, "r") as fin:
        source = fin.read()
    run(source, args.log_file, args.articles_file)

#python runner.py data/handout_train.npy data/handout_test.npy project3.py
