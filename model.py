import random
from math import log
from collections import defaultdict, OrderedDict

SMALL_POSSIBILITY =  (1e-20)

class Brain(object):
    SPEECHES = set(['adv', 'noun', 'adp', 
                    'prt', 'det', 'num', 
                    '.', 'pron', 'verb', 
                    'x', 'conj', 'adj'])

    def __init__(self):

        self.neighbor_speech_counter = defaultdict(lambda: 1)
        self.speech_container = defaultdict(lambda:defaultdict(int))
        self.first_speech_counter = defaultdict(lambda: 1)
        self.word_counter = defaultdict(int)
        self.word_speech_possibility  = {}
        self.speech_word_possibility  = {}

        self.first_speech_possibility = {}
        self.first_speech_sum = None

        self.neighbor_speech_possibility = {}
        self.neightbor_speech_sum = None

        self.speech_word_sum = None


    ## P(s1)
    def p_s1(self, speech):

        if speech in self.first_speech_possibility:
            return self.first_speech_possibility[speech]

        if self.first_speech_sum is None:
            self.first_speech_sum = sum(self.first_speech_counter.values())
            # print self.first_speech_counter
            # print "no_of_sentence", self.first_speech_sum

        possibility =  (float(self.first_speech_counter[speech])/ self.first_speech_sum)
                                    
        self.first_speech_possibility[speech] = possibility
        if possibility < SMALL_POSSIBILITY:possibility=SMALL_POSSIBILITY
        return possibility

    ## P(Si+1|Si)
    def psi_sii(self, speech1, speech2):
        key = (speech1, speech2)
        if key in self.neighbor_speech_possibility:
            return self.neighbor_speech_possibility[key]

        possibility =  (float(self.neighbor_speech_counter[key])/ (self.speech_container[speech2]['0sum']))
        # -self.speech_container[speech2]['last']))
        if possibility < SMALL_POSSIBILITY:possibility=SMALL_POSSIBILITY
                            
        self.neighbor_speech_possibility[key] = possibility
        return possibility

    ## P(Wi|Si)
    def pw_s(self, word, speech):
        key = (word, speech)
        if key in self.word_speech_possibility:
            return self.word_speech_possibility[key]
        word_counter = self.speech_container[speech]
        # print "len:self.speech_container:%s"%len(self.speech_container[speech])
        if word_counter[word] == 0:
            if speech == "noun":
                self.word_speech_possibility[key] = 4*SMALL_POSSIBILITY
                return 4*SMALL_POSSIBILITY
            else: 
                self.word_speech_possibility[key] = SMALL_POSSIBILITY
                return SMALL_POSSIBILITY
            


        speech_sum = sum(word_counter.values())
        possibility = (word_counter[word]/float(speech_sum))
        if possibility < SMALL_POSSIBILITY:possibility=SMALL_POSSIBILITY
        self.word_speech_possibility[key] = possibility
        return possibility

    def ps_w(self, speech, word):
        key = (word, speech)
        if key in self.speech_word_possibility:
            return self.speech_word_possibility[key]

        word_counter = self.speech_container[speech]
        # print "len:self.speech_container:%s"%len(self.speech_container[speech])
        if word_counter[word] == 0 or self.word_counter[word] == 0:
            if speech == "noun":
                self.speech_word_possibility[key] = 4*SMALL_POSSIBILITY
                return 4*SMALL_POSSIBILITY
            else: 
                self.speech_word_possibility[key] = SMALL_POSSIBILITY
                return SMALL_POSSIBILITY
        possibility = word_counter[word]/float(self.word_counter[word])
        if possibility < SMALL_POSSIBILITY:possibility=SMALL_POSSIBILITY
        self.speech_word_possibility[key] = possibility
        return possibility

    def most_likely_speech(self, word):
        max_possibility = -1
        most_likely_speech = None
        for speech in self.SPEECHES:
            possibility = self.ps_w(speech, word)
            if possibility > max_possibility:
                max_possibility = possibility
                most_likely_speech = speech

        return most_likely_speech

    def possible_speech(self, word):
        result = [(speech, self.speech_container[speech][word]) for speech in self.speech_container]
        if sum([_[1] for _ in result]) == 0:
            result = [(speech, 1) for speech in self.speech_container]
        return dict(result)
    # may consider 
    # https://darrenjw.wordpress.com/2011/07/16/gibbs-sampler-in-various-languages-revisited/
    # 
    def gibbs_sample(self, sentence):
        result = []
        previous_speech = None
        first_word = sentence[0]
        possible_speech = self.possible_speech(first_word)
        first_possible_speech_sum = sum([self.first_speech_counter[speech] \
                                                * possible_speech[speech] \
                                                for speech in possible_speech]) 
        random_roll = random.randint(1, first_possible_speech_sum)
        base = 0
        for speech in possible_speech:
            value = self.first_speech_counter[speech]*possible_speech[speech]
            if base < random_roll <= base+value:
                previous_speech = speech
                result.append(speech)
                break
            else:
                base += value

        for word in sentence[1:]:
            possible_speech = self.possible_speech(word)
            # print "=="
            # print possible_speech
            possible_speech_sum = sum([self.neighbor_speech_counter[(speech, previous_speech)] \
                                                    * possible_speech[speech]\
                                                     for speech in possible_speech])
            random_roll = random.randint(1, possible_speech_sum)
            # print "total:%s"%total
            # print "random_roll:%s"%random_roll
            # print "possible_speech:%s"%possible_speech
            # print ""
            # print "result:%s"%result
            base = 0

            for speech in possible_speech:
                key = (speech, previous_speech)
                value = self.neighbor_speech_counter[key]*possible_speech[speech]
                if base < random_roll <= base+value:
                    previous_speech = speech
                    result.append(speech)
                    break
                else:
                    base += value
        #     print "result:%s"%result
        # print "done"
        return result

    def viterbi(self, sentence):
        word_list=[ {} for number in xrange(len(sentence))]
    
        for pointer in range(0,len(sentence)):
            if pointer == 0:
                for pos in self.SPEECHES:
                    word_list[pointer][pos] = ('0',self.p_s1(pos)*self.pw_s(sentence[pointer],pos))

            else:
                for  to_pos in self.SPEECHES:
                    alpha = -1
                    for from_pos in self.SPEECHES:
                        tmp = word_list[pointer-1][from_pos][1]*self.psi_sii(to_pos,from_pos)*self.pw_s(sentence[pointer],to_pos)
                        if tmp > alpha:
                            word_list[pointer][to_pos] = (from_pos,tmp) 
                            # print word_list[pointer][to_pos]
                            alpha = tmp
        alpha = -1
        for key, values in word_list[-1].items():
            if values[1] > alpha:
                alpha = values[1]
                tmp_1 = values[0]
                tmp_2 = key
        tmp_list = [tmp_2]
    
        pointer = len(sentence)-2
        while tmp_1 != '0':
            tmp_list.insert(0,tmp_1)
            tmp_1 = word_list[pointer][tmp_1][0]
            pointer -= 1

        return tmp_list 

    def rev_viterbi(self, sentence):
        word_list=[ {} for number in xrange(len(sentence))]
    
        for pointer in range(0,len(sentence)):
            for to_pos in self.SPEECHES:
                h = 1
                if pointer < len(sentence)-1:
                        for fr_pos in self.SPEECHES:
                            alpha = -1
                            z = self.ps_w(fr_pos, sentence[pointer+1])
                            if z > alpha:
                                alpha = z
                                tmp_pos = fr_pos
                        h = self.psi_sii(to_pos,tmp_pos)
                if pointer == 0:
                    word_list[pointer][to_pos] = ('0',abs(log(h))*self.p_s1(to_pos)*self.ps_w(to_pos,sentence[pointer]))
                else:
                    alpha = -1
                    print log(h)
                    for from_pos in self.SPEECHES:
                        for count in xrange(2):
                            if count == 0:
                                tmp = abs(log(h))*word_list[pointer-1][from_pos][1]*self.psi_sii(to_pos,from_pos)*self.ps_w(to_pos,sentence[pointer])
                            # else:
                            #     tmp = abs(log(h))*word_list[pointer-1][from_pos][1]*self.psi_sii(to_pos,from_pos)*self.pw_s(sentence[pointer],to_pos)
                            if tmp > alpha:
                                word_list[pointer][to_pos] = (from_pos,tmp) 
                                # print word_list[pointer][to_pos]
                                alpha = tmp
        alpha = -1
        for key, values in word_list[-1].items():
            if values[1] > alpha:
                alpha = values[1]
                tmp_1 = values[0]
                tmp_2 = key
        tmp_list = [tmp_2]
    
        pointer = len(sentence)-2
        while tmp_1 != '0':
            tmp_list.insert(0,tmp_1)
            tmp_1 = word_list[pointer][tmp_1][0]
            pointer -= 1

        return tmp_list


    def _recalculate_posibility(self):

        ## for neighbor
        sum_num = sum(self.neighbor_speech_counter.values())
        for key in self.neighbor_speech_counter:
            self.neighbor_speech_possibility[key] = \
                        float(self.neighbor_speech_counter[key]) / sum_num

    def parse_data(self, data):
        for sentence in data:
            last_speech = None
            # print data
            self.first_speech_counter[sentence[1][0]] += 1
            for spell, speech in zip(*sentence):
                # print "[%s:%s]"%(spell, speech)
                if last_speech:
                    self.neighbor_speech_counter[(speech, last_speech)] += 1
                if spell not in self.speech_container[speech]:
                    self.speech_container[speech][spell] = 0
                self.speech_container[speech][spell] += 1
                self.word_counter[spell] += 1
                last_speech = speech
            # if 'last' not in self.speech_container[last_speech]:
            #     self.speech_container[last_speech]['last'] = 0
            # self.speech_container[last_speech]['last'] += 1
        for speech in self.SPEECHES:
            self.speech_container[speech]['0sum']=sum(self.speech_container[speech].values())


if __name__ == "__main__":
    def read_data(fname):
        exemplars = []
        file = open(fname, 'r');
        for line in file:
            data = tuple([w.lower() for w in line.split()])
            exemplars += [ (data[0::2], data[1::2]), ]

        return exemplars

    b = Brain()
    data = read_data("bc.train")
    b.parse_data(data)
    a = b.neighbor_speech_counter.keys()[10]
    for i in b.SPEECHES:
        print b.p_s1(i), i
    print b.psi_sii("verb", "verb"),"verb"
    print b.pw_s("the","det"),"the,det"
    print b.pw_s("any","adj")










