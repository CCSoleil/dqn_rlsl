# -*- coding: utf-8 -*-
# Natural Language Toolkit: Interface to the CRFSuite Tagger
#
# Copyright (C) 2001-2017 NLTK Project
# Author: Long Duong <longdt219@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

"""
A module for POS tagging using CRFSuite
"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unicodedata
import re 
import os
import math
import pdb

try:
    import pycrfsuite
except ImportError:
    pass

class CRFTagger(object):
    
    def __init__(self, modelfile, feature_func = None, verbose = False, training_opt = {}):
        """
        Initialize the CRFSuite tagger 
        :param feature_func: The function that extracts features for each token of a sentence. This function should take 
        2 parameters: tokens and index which extract features at index position from tokens list. See the build in 
        _get_features function for more detail.   
        :param verbose: output the debugging messages during training.
        :type verbose: boolean  
        :param training_opt: python-crfsuite training options
        :type training_opt : dictionary 
        
        Set of possible training options (using LBFGS training algorithm).  
         'feature.minfreq' : The minimum frequency of features.
         'feature.possible_states' : Force to generate possible state features.
         'feature.possible_transitions' : Force to generate possible transition features.
         'c1' : Coefficient for L1 regularization.
         'c2' : Coefficient for L2 regularization.
         'max_iterations' : The maximum number of iterations for L-BFGS optimization.
         'num_memories' : The number of limited memories for approximating the inverse hessian matrix.
         'epsilon' : Epsilon for testing the convergence of the objective.
         'period' : The duration of iterations to test the stopping criterion.
         'delta' : The threshold for the stopping criterion; an L-BFGS iteration stops when the
                    improvement of the log likelihood over the last ${period} iterations is no greater than this threshold.
         'linesearch' : The line search algorithm used in L-BFGS updates:
                           { 'MoreThuente': More and Thuente's method,
                              'Backtracking': Backtracking method with regular Wolfe condition,
                              'StrongBacktracking': Backtracking method with strong Wolfe condition
                           } 
         'max_linesearch' :  The maximum number of trials for the line search algorithm.
         
        """
        self.name = "CRF"                   
        self._model_file = modelfile
        self._tagger = pycrfsuite.Tagger()
        
        if feature_func is None:
            self._feature_func =  self._get_features
        else:
            self._feature_func =  feature_func
        
        self._verbose = verbose 
        #self._training_options = training_opt
        self._training_options = {
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        }

        self._pattern = re.compile(r'\d')

    def loadmodel(self):
        self._tagger.open(self._model_file)
            
    def _get_features(self, tokens, idx):
        """
        Extract basic features about this word including 
             - Current Word 
             - Is Capitalized ?
             - Has Punctuation ?
             - Has Number ?
             - Suffixes up to length 3
        Note that : we might include feature over previous word, next word ect. 
        
        :return : a list which contains the features
        :rtype : list(str)    
        
        """ 
        token = tokens[idx]
        
        feature_list = []
        
        if not token:
            return feature_list
            
        # Capitalization 
        if token[0].isupper():
            feature_list.append('CAPITALIZATION')
        
        # Number 
        if re.search(self._pattern, token) is not None:
            feature_list.append('HAS_NUM') 
        
        # Punctuation
        punc_cat = set(["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"])
        if all (unicodedata.category(x) in punc_cat for x in token):
            feature_list.append('PUNCTUATION')
        
        # Suffix up to length 3
        if len(token) > 1:
            feature_list.append('SUF_' + token[-1:]) 
        if len(token) > 2: 
            feature_list.append('SUF_' + token[-2:])    
        if len(token) > 3: 
            feature_list.append('SUF_' + token[-3:])
            
        feature_list.append('WORD_' + token )
        
        return feature_list

    def tag_sents(self, sents):
        '''
        Tag a list of sentences. NB before using this function, user should specify the mode_file either by 
                       - Train a new model using ``train'' function 
                       - Use the pre-trained model which is set via ``set_model_file'' function  
        :params sentences : list of sentences needed to tag. 
        :type sentences : list(list(str))
        :return : list of tagged sentences. 
        :rtype : list (list (tuple(str,str))) 
        '''
        if self._model_file == '':
            raise Exception(' No model file is found !! Please use train or set_model_file function')
        
        # We need the list of sentences instead of the list generator for matching the input and output
        result = []  
        for tokens in sents:
            features = [self._feature_func(tokens,i) for i in range(len(tokens))]
            labels = self._tagger.tag(features)
                
            if len(labels) != len(tokens):
                raise Exception(' Predicted Length Not Matched, Expect Errors !')
            
            tagged_sent = list(zip(tokens,labels))
            result.append(tagged_sent)
            
        return result 

    def train(self, train_data):
        '''
        Train the CRF tagger using CRFSuite  
        :params train_data : is the list of annotated sentences.        
        :type train_data : list (list(tuple(str,str)))
        :params model_file : the model will be saved to this file.     
         
        '''
        trainer = pycrfsuite.Trainer(verbose=self._verbose)
        trainer.set_params(self._training_options)
        
        for sent in train_data:
            tokens,labels = zip(*sent)
            features = [self._feature_func(tokens,i) for i in range(len(tokens))]
            trainer.append(features,labels)
                        
        # Now train the model, the output should be model_file
        trainer.train(self._model_file)
        self.loadmodel() 

    def tag(self, tokens):
        '''
        Tag a sentence using Python CRFSuite Tagger. NB before using this function, user should specify the mode_file either by 
                       - Train a new model using ``train'' function 
                       - Use the pre-trained model which is set via ``set_model_file'' function  
        :params tokens : list of tokens needed to tag. 
        :type tokens : list(str)
        :return : list of tagged tokens. 
        :rtype : list (tuple(str,str)) 
        '''
        
        return self.tag_sents([tokens])[0]


#==============================================================#
#                                                              #
#   The following is added to support reinforcement learning   #
#                                                              #
#==============================================================#

    def sent2features(self, sent):
        return [self._feature_func(sent, i) for i in range(len(sent))]

    def sent2tokens(self, sent):
        items = sent.split()
        return [x.decode('utf-8') for x in items]

    def get_predictions(self, sent):
        tokens = self.sent2tokens(sent)
        x = self.sent2features(tokens)
        if not os.path.isfile(self._model_file):
            y_marginals = []
            for i in range(len(x)):
                y_marginals.append([0.0833] * 12)
            return y_marginals
        
        self._tagger.set(x)
        y_marginals = []
        alllabels = ['ADP', 'DET', 'NOUN', 'NUM', '.', 'PRT', 'VERB', 'CONJ', 'ADV', 'PRON', 'ADJ', 'X']
        for i in range(len(x)):
            y_i = []
            for y in alllabels:
                if y in self._tagger.labels():
                    y_i.append(self._tagger.marginal(y, i))
                else:
                    y_i.append(0.)
            y_marginals.append(y_i)
        return y_marginals

    def get_confidence(self, sent):
        tokens = self.sent2tokens(sent)
        x = self.sent2features(tokens)
        if not os.path.isfile(self._model_file):
            confidence = 0.2
            return [confidence]

        self._tagger.set(x)
        y_pred = self._tagger.tag()
        p_y_pred = self._tagger.probability(y_pred)
        confidence = pow(p_y_pred, 1. / len(y_pred))
        return [confidence]

    def get_uncertainty(self, sent):
        tokens = self.sent2tokens(sent)
        x = self.sent2features(tokens)
        if not os.path.isfile(self._model_file):
            unc = random.random()
            return unc

        self._tagger.set(x)
        alllabels = ['ADP', 'DET', 'NOUN', 'NUM', '.', 'PRT', 'VERB', 'CONJ', 'ADV', 'PRON', 'ADJ', 'X']
        ttk = 0.
        for i in range(len(x)):
            y_probs = []
            for y in alllabels:
                if y in self._tagger.labels():
                    y_probs.append(self._tagger.marginal(y, i))
                else:
                    y_probs.append(0.)
            ent = 0.
            for y_i in y_probs:
                if y_i > 0:
                    ent -= y_i * math.log(y_i, 12)
            ttk += ent
        return ttk

    def pred(self, test_sents):
        X_test = [self.sent2features(s) for s in test_sents]
        y_pred = [self._tagger.tag(xseq) for xseq in X_test]
        return y_pred

    def exists(self):
        return os.path.isfile(self._model_file)

    def clean(self):
        if self.exists():
           os.remove(self._model_file)

    def test(self, test_sents):
        X_test = []
        Y_true = []
        for sent in test_sents:
            tokens,labels = zip(*sent)
            features = [self._feature_func(tokens,i) for i in range(len(tokens))]
            X_test.append(features)
            Y_true.append(labels) 

        y_pred = [self._tagger.tag(xseq) for xseq in X_test]
        pre = 0
        pre_tot = 0
        rec = 0
        rec_tot = 0
        corr = 0
        total = 0
        for i in range(len(Y_true)):
            for j in range(len(Y_true[i])):
                total += 1
                if y_pred[i][j] == Y_true[i][j]:
                    corr += 1
                if y_pred[i][j] != '.':  # not 'O'
                    pre_tot += 1
                    if y_pred[i][j] == Y_true[i][j]:
                        pre += 1
                if Y_true[i][j] != '.':
                    rec_tot += 1
                    if y_pred[i][j] == Y_true[i][j]:
                        rec += 1

        res = corr * 1. / total
        print "Accuracy (token level)", res
        if pre_tot == 0:
            pre = 0
        else:
            pre = 1. * pre / pre_tot
        rec = 1. * rec / rec_tot
        print pre, rec

        beta = 1
        f1score = 0
        if pre != 0 or rec != 0:
            f1score = (beta * beta + 1) * pre * rec / \
                (beta * beta * pre + rec)
        print "F1", f1score
        return f1score

