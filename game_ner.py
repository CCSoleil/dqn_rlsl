import numpy as np
import sys
import random
import utilities
from tagger import CRFTagger
import tensorflow as tf
import copy
import random
import os
import collections
import pdb

class NERGame:

    def __init__(self, story, test, dev, max_len, w2v, budget, split, isload, fn):
        # build environment
        # load data as story
        print("Initilizing the game:")
        # import story
        #keep some of the data to train the baseline of the model
        if split:
            #keepnum = 20
            keepnum = 100
            story1, story2 = self.splitStory(story, keepnum, isload, fn) 
            self.base_x, self.base_y, self.base_idx = story2
        else:
            story1 = story
            self.base_x=[]
            self.base_y=[]
            self.base_idx=[]

        self.train_x, self.train_y, self.train_idx = story1
        self.test_x, self.test_y, self.test_idx = test
        self.dev_x, self.dev_y, self.dev_idx = dev
        self.max_len = max_len
        self.w2v = w2v

        print "Story: length = ", len(self.train_x)
        self.order = range(0, len(self.train_x))

        # when queried times is 100, then stop
        self.budget = budget #redine the budget to the no change threshold
        self.queried_times = 0

        # select pool
        #self.queried_x = []
        #self.queried_y = []
        #self.queried_idx = []
        self.queried_x = copy.copy(self.base_x) 
        self.queried_y = copy.copy(self.base_y)
        self.queried_idx = copy.copy(self.base_idx)

        # let's start
        self.episode = 1
        # story frame
        self.currentFrame = 0
        self.terminal = False
        self.make_query = False
        self.performance = 0

        #define the list to keep latest 10  performance for checking the terminal conditions
        self.performancelist = collections.deque(maxlen=10)   
 
    def getSelectedIndex(self, total, keepnum, isload, fn):
        selected = []
        if isload:
           inff = open(fn, 'rb')
           line = inff.readline()
           while line:
               item = line.strip()
               selected.append(int(item))
               line = inff.readline()
           inff.close()

        else: 
           while keepnum>0:
               index = random.randint(0, total)
               if index not in selected:
                   selected.append(index)
                   keepnum-=1

           outff = open(fn, 'wb')
           for i in selected:
               outff.write(str(i)+'\n')
           outff.close()
 
        return selected
    

    def splitStory(self, story, keepnum, isload, fn):
        tx, ty, idx = story
        tx1=[]
        ty1=[]
        idx1=[]
        tx2=[]
        ty2=[]
        idx2=[]
        total = len(tx)

        selected = self.getSelectedIndex(total, keepnum, isload, fn)
      
        for i in range(total):
            if i in selected:
               tx2.append(tx[i])
               ty2.append(ty[i])
               idx2.append(idx[i])
            else:
               tx1.append(tx[i])
               ty1.append(ty[i])
               idx1.append(idx[i])

        story1=[tx1, ty1, idx1]
        story2=[tx2, ty2, idx2]
        return story1, story2  

    # if there is an input from the tagger side, then it is
    # getFrame(self,tagger)
    def get_frame(self, model):
        self.make_query = False
        sentence = self.train_x[self.order[self.currentFrame]]
        sentence_idx = self.train_idx[self.order[self.currentFrame]]
        confidence = 0.
        predictions = []
        if model.name == "CRF":
            confidence = model.get_confidence(sentence)
            predictions = model.get_predictions(sentence)
        else:
            confidence = model.get_confidence(sentence_idx)
            predictions = model.get_predictions(sentence_idx)
        preds_padding = []
        orig_len = len(predictions)
        if orig_len < self.max_len:
            preds_padding.extend(predictions)
            for i in range(self.max_len - orig_len):
                preds_padding.append([0] * 12)
        elif orig_len > self.max_len:
            preds_padding = predictions[0:self.max_len]
        else:
            preds_padding = predictions

        obervation = [sentence_idx, confidence, preds_padding]
        return obervation

    def checkPerformanceChanged(self):
        plen = len(self.performancelist)
        if plen <10:
           return False

        changed = 0
        for i in range(plen-1):
            changed+= abs(self.performancelist[i+1] - self.performancelist[i]) 
        changed = changed/(plen-1)

        if changed < self.budget: #threshold of no change           
           return True
        else:
           return False

    # tagger = crf model
    def feedback(self, action, model, selfstudy):
        reward = 0.
        if action[1] == 1:
            self.make_query = True
            if selfstudy:
                if not model.exists():
                    self.retrainTagger(model)
                self.queryNew(model)
            else:
                self.query()
  
            new_performance = self.get_performance(model)
            reward = new_performance - self.performance
            self.performancelist.append(new_performance)
            if new_performance != self.performance:
                self.performance = new_performance
        else:
            reward = 0.
        # next frame
        isTerminal = False
        nochange = self.checkPerformanceChanged()
        #if self.queried_times == self.budget or self.currentFrame>=len(self.train_idx)-1:

        if nochange or self.currentFrame>=len(self.train_idx)-1:
            self.terminal = True
            # update special reward
            isTerminal = True
            #reward = new_performance * 100
            self.reboot(model)  # set current frame = -1
            next_sentence = self.train_x[self.order[self.currentFrame]]
            next_sentence_idx = self.train_idx[self.order[self.currentFrame]]
        else:
            self.terminal = False
            next_sentence = self.train_x[self.order[self.currentFrame + 1]]
            next_sentence_idx = self.train_idx[self.order[self.currentFrame + 1]]
            self.currentFrame += 1

        confidence = 0.
        predictions = []
        if model.name == "CRF":
            confidence = model.get_confidence(next_sentence)
            predictions = model.get_predictions(next_sentence)
        else:
            confidence = model.get_confidence(next_sentence_idx)
            predictions = model.get_predictions(next_sentence_idx)
        preds_padding = []
        orig_len = len(predictions)
        if orig_len < self.max_len:
            preds_padding.extend(predictions)
            for i in range(self.max_len - orig_len):
                preds_padding.append([0] * 12)
        elif orig_len > self.max_len:
            preds_padding = predictions[0:self.max_len]
        else:
            preds_padding = predictions

        next_observation = [next_sentence_idx, confidence, preds_padding]
        return reward, next_observation, isTerminal

    def query(self):
        if self.make_query == True:
            sentence = self.train_x[self.order[self.currentFrame]]
            # simulate: obtain the labels
            labels = self.train_y[self.order[self.currentFrame]]
            self.queried_times += 1
            # print "Select:", sentence, labels
            self.queried_x.append(sentence)
            self.queried_y.append(labels)
            self.queried_idx.append(
                self.train_idx[self.order[self.currentFrame]])
            print "> Queried times", len(self.queried_x)

    def setRandomOrder(self):
        random.shuffle(self.order)

    def getCurrentFrameSentence(self, incremental):
        sentence = self.train_x[self.order[self.currentFrame]]
        labels = self.train_y[self.order[self.currentFrame]]
        self.currentFrame+=incremental
        return sentence, labels

    def retrainTagger(self, tagger):
        #re-train the model
        print len(self.queried_x), len(self.queried_y)
        train_sents = self.data2sents(self.queried_x, self.queried_y)
        tagger.train(train_sents)
                

    def queryNew(self, tagger):
        if self.make_query == True:

            sentence = self.train_x[self.order[self.currentFrame]]
            # simulate: obtain the labels
            tokens = tagger.sent2tokens(sentence)
            pl = tagger.pred([tokens])
            labels = pl[0]
            self.queried_times += 1
            self.queried_x.append(sentence)
            self.queried_y.append(labels)
            self.queried_idx.append(
                self.train_idx[self.order[self.currentFrame]])
            print "> Queried times", len(self.queried_x)


    # tagger = model
    def get_performance(self, tagger):
        # train on queried_x, queried_y
        # single training: self.model.train(self.queried_x, self.queried_y)
        # train on mutiple epochs
        if tagger.name == "RNN":
            tagger.train(self.queried_idx, self.queried_y)
            performance = tagger.test(self.dev_idx, self.dev_y)
            return performance

        train_sents = self.data2sents(self.queried_x, self.queried_y)
        tagger.train(train_sents)
        # test on development data
        test_sents = self.data2sents(self.dev_x, self.dev_y)
        performance = tagger.test(test_sents)
        return performance

    def reboot(self, model):
        # resort story
        random.shuffle(self.order)
        self.queried_times = 0
        self.terminal = False
        #self.queried_x = []
        #self.queried_y = []
        #self.queried_idx = []
        self.queried_x = copy.copy(self.base_x) 
        self.queried_y = copy.copy(self.base_y)
        self.queried_idx = copy.copy(self.base_idx)
        self.currentFrame = -1
        self.episode += 1
        self.performancelist.clear()

        model.clean()
        print "> Next episode", self.episode


    def data2sents(self, X, Y):
        data = []
        # print Y
        for i in range(len(Y)):
            sent = []
            text = X[i]
            items = text.split()
            for j in range(len(Y[i])):
                sent.append((items[j].decode('utf-8'), Y[i][j]))
            data.append(sent)
        return data
