###################################
# CS B551 Fall 2015, Assignment #5
#
# Your names and user ids:
#   Vidya Sagar (vkalvaku)    
#   Yang Zhang (zhang505)
# (Based on skeleton code by D. Crandall)
#
#
####
'''
    We need to consider the first word of sentences specially.
    If the program not improve, ignore first word consideration.

    We wrote the main part of our code in another file, which is model.py. and 
    imported the required function for each file

Step 1: Learning:

    In that file the `parse_data` is for training. 
    This method reads the data and uses the dictionary data structure to count each word. 
    To avoid initializing the dictionary value each time, we used `defaultdict`, 
    the interesting part is that when we try to use defaultdict as the default value, 
    the syntax is tricky, it looks like `defaultdict(lambda:defaultdict(int))`.

    `self.word_counter` is the dictionary contain how many time the word occurs.
    `self.neighbor_speech_counter` is the dictionary contain the speech and the previous 
    speech occurs, the key is a tuple. The example is below: 
    
    {
        ("noun", "verb"): 12,
        ("verb", "adj"): 42,
        ...
    }

    `self.speech_container` is a dictionary contain a dictionary. The first key is a speech,
    the value is the dictionary contain the word counter. An example is show below:
    {
        "noun":
            {"dog": 1}
            {"cat": 2}
            ....
        "verb": 
            {"say": 2}
        ...
    }

    `self.first_speech_counter` is to store the first speech occurs in sentences.
    The other attributes in the Brain class is used for cache strategy. 

Step 2: Naive inference:

    In this method the probability of all the parts of speech for every word in the sentence
    is compared and the part of speech with the highest probability is assumed to be the 
    correct tag.

    This depends on the function ps_w( speech, word), it takes part of speech and word as 
    input and returns the probability.

Step 3: Sampling:
 
    For the sampling part, we divide the problem in to two parts.

    The first one we sample the first speech of sentence. We times the number of
    speeches with the possible first speech( The one occurs as the first speech).
    Then we use random.randint to do the random choice. When the number is between the 
    speech(base, base+value) range, we choose the speech.

    The second part is need to consider the previous speech. The one is quite similar
    to the first speech. Instead of using the  `self.first_speech_counter`, we use the 
    `self.neighbor_speech_counter`. That is P(Si+1/Si).

Step 4: Approximate max-marginal inference.

    Here we use the same function as the Sampler but we consider multiple samples and for 
    each word in the sentence we take the part of speech that is returned most number of 
    times by the sampler function.

Step 5: Exact maximum a posteriori inference

    In this method again there are 2 cases:
    a) when the word is a start sentence: When its a start sentence we use the start probability 
       of speech and the emmission probability of that speech of that particular word to calcute 
       the value and store it in a dictionary.
    b) when the word is not in start of sentence then for each possible part of speech we 
       calculate the value and store it in a different dictionary.this value is calculated 
       using formula for vitterbi algorithm.

    Emmission(W|Si) * argmax(Transition Probability(Si+1/ Si) * Probability(previous state))

    We used a dictionary to store the previous state from which the maximum value is generated
    so that while we backtrack we can directly look up on the state we have saved.

    viterbi 
    [
        w1 {
            ("noun"): (0,0.8)    #here in this tuple 0 indicates that it is the first element
                                  ,0.8 is  the value of P(s1)*P(w1/"noun")
            ("verb"): (0,0.21)

            },
        w2 {
            ("noun"): ('verb',0.37) #here the verb is the link to the previous word speech 
                                    which was multiplied, 0.37 is w1[verb]*P(verb/noun)*P(w2/noun)
            ("verb"): ('adp',0.01)

            },
        .
        .
        .
        Wn {
            ("noun"): ('verb',0.37)
            ("verb"): ('adp',0.01)

            }
    ]

Step 6: Whatever works best! :

    We found that existing viterbi algorithm gives the best accuracy for the given 
    training and test data set.

    We implemented methods to handle words and part-of-speech that arent present in 
    the training set. When an unknown word occurs for the first time in the test data, 
    we give slightly higher probability of it being a noun.and for the rest we 
    assigned a constant low probability 
                SMALL_POSSIBILITY =  (1e-20)


For the given test set(bc.test) and given training set (bc.train):

==> So far scored 2000 sentences with 29442 words.
                       Words correct:     Sentences correct: 
    0. Ground truth:      100.00%              100.00%
    1. Naive:              93.92%               47.50%
    2. Sampler:            91.34%               37.40%
    3. Max marginal:       92.40%               40.80%
    4. MAP:                95.20%               55.85%
    5. Best:               95.20%               55.85%



'''
####

import random
import math
from math import log
from model import Brain
# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    def __init__(self):
        self.brain = Brain()

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        # print "posterior", sentence,label
        pws = 1
        for pointer in xrange(len(sentence)):
            pws += log(self.brain.pw_s(sentence[pointer],label[pointer]))
            if pointer == 0:
                ps = log(self.brain.p_s1(label[pointer]))
            else:
                ps += log(self.brain.psi_sii(label[pointer],label[pointer-1]))
        
        return ps+pws

    # Do the training!
    #
    def train(self, data):
        self.brain.parse_data(data)

    # Functions for each algorithm.
    #
    def naive(self, sentence):
        result = []

        for word in sentence:
            speech = self.brain.most_likely_speech(word)
            result.append(speech)
        return [ [result], [] ]

    def mcmc(self, sentence, sample_count):
        result = []
        
        for i in range(sample_count+100):
            x = self.brain.gibbs_sample(sentence)
            if i > 99:
                result.append(x)
               
        # print [len(_) for _ in result]
        # print "sentence_len:%s"%len(sentence)
        return [ result, [] ]

    def best(self, sentence):
        result = self.brain.viterbi(sentence)
        # print result
        return [ [result], [] ]
        # return [ [ [ "noun" ] * len(sentence)], [] ]

    def max_marginal(self, sentence):
        result = []
        prob_list = []
        tmp_array = [ {} for number in xrange(len(sentence))]
        for i in range(100):
            x = self.brain.gibbs_sample(sentence)
            if i > 49:
                for n in xrange(len(sentence)):
                    if x[n] not in tmp_array[n]:tmp_array[n][x[n]]=0
                    tmp_array[n][x[n]] += 1
        # print tmp_array
        for n in xrange(len(sentence)):
            z = max(tmp_array[n], key=tmp_array[n].get)
            result.append(z)
            y = (tmp_array[n][z])/float(sum(tmp_array[n].values()))
            prob_list.append(y)

        return [ [result],  [prob_list] ]

    def viterbi(self, sentence):
        result = self.brain.viterbi(sentence)
        # print result
        return [ [result], [] ]


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #    Most algorithms only return a single labeling per sentence, except for the
    #    mcmc sampler which is supposed to return 5.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for max_marginal() and is the marginal probabilities for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Naive":
            return self.naive(sentence)
        elif algo == "Sampler":
            return self.mcmc(sentence, 5)
        elif algo == "Max marginal":
            return self.max_marginal(sentence)
        elif algo == "MAP":
            return self.viterbi(sentence)
        elif algo == "Best":
            return self.best(sentence)
        else:
            print "Unknown algo!"

