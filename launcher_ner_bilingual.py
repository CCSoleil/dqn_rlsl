import sys
import argparse
from game_ner import NERGame
from robot import RobotCNNDQN
import numpy as np
import utilities
import tensorflow as tf
import random
from tagger import CRFTagger
import shutil

tf.flags.DEFINE_integer("max_seq_len", 120, "sequence")
tf.flags.DEFINE_integer("max_vocab_size", 20000, "vocabulary")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

AGENT = "CNNDQN"
MAX_EPISODE = 0
BUDGET = 0
TRAIN_LANG = []
TRAIN_LANG_NUM = 1
TEST_LANG = []
TEST_LANG_NUM = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', help="require a decision agent")
    parser.add_argument(
        '--episode', help="require a maximum number of playing the game")
    parser.add_argument('--budget', help="requrie a budget for annotating")
    parser.add_argument('--train', help="training phase")
    parser.add_argument('--test', help="testing phase")
    parser.add_argument('--checkpoint', help="checkpoint name")
    parser.add_argument('--saved_robot', help="file name of saved robot")
    parser.add_argument('--replay', help="is replay or not")
    parser.add_argument('--nextepisode', help="next episode")
    args = parser.parse_args()
    global AGENT, MAX_EPISODE, BUDGET, TRAIN_LANG, TEST_LANG, CHECKPOINT, SAVED_ROBOT, REPLAY, NEXTEPISODE
    AGENT = args.agent
    MAX_EPISODE = int(args.episode)
    #BUDGET = int(args.budget)
    BUDGET = float(args.budget)
    CHECKPOINT = args.checkpoint
    REPLAY = (args.replay == 'True')
    NEXTEPISODE = 1
    SAVED_ROBOT = ''
    if REPLAY:
       NEXTEPISODE = int(args.nextepisode)
       SAVED_ROBOT = args.saved_robot

    parts = args.train.split(";")
    if len(parts) % 5 != 0:
        print "Wrong inputs of training"
        raise SystemExit
    global TRAIN_LANG_NUM
    TRAIN_LANG_NUM = len(parts) / 5
    for i in range(TRAIN_LANG_NUM):
        i_lang = i * 5
        train = parts[i_lang + 0]
        test = parts[i_lang + 1]
        dev = parts[i_lang + 2]
        emb = parts[i_lang + 3]
        tagger = parts[i_lang + 4]
        TRAIN_LANG.append((train, test, dev, emb, tagger))
    parts = args.test.split(";")
    if len(parts) % 5 != 0:
        print "Wrong inputs of testing"
        raise SystemExit
    global TEST_LANG_NUM
    TEST_LANG_NUM = len(parts) / 5
    for i in range(TEST_LANG_NUM):
        i_lang = i * 5
        train = parts[i_lang + 0]
        test = parts[i_lang + 1]
        dev = parts[i_lang + 2]
        emb = parts[i_lang + 3]
        tagger = parts[i_lang + 4]
        TEST_LANG.append((train, test, dev, emb, tagger))


def initialise_game(trainFile, testFile, devFile, embFile, budget, split, isload, fn):
    # Load data
    print("Loading data ..")
    train_x, train_y, train_lens = utilities.load_data2labels(trainFile)
    test_x, test_y, test_lens = utilities.load_data2labels(testFile)
    dev_x, dev_y, dev_lens = utilities.load_data2labels(devFile)

    print("Processing data")
    # build vocabulary
    max_len = FLAGS.max_seq_len
    print "Max document length:", max_len
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_document_length=max_len, min_frequency=1)
    # vocab = vocab_processor.vocabulary_ # start from {"<UNK>":0}
    train_idx = np.array(list(vocab_processor.fit_transform(train_x)))
    dev_idx = np.array(list(vocab_processor.fit_transform(dev_x)))
    vocab = vocab_processor.vocabulary_
    vocab.freeze()
    test_idx = np.array(list(vocab_processor.fit_transform(test_x)))

    # build embeddings
    vocab = vocab_processor.vocabulary_
    vocab_size = FLAGS.max_vocab_size
    w2v = utilities.load_crosslingual_embeddings(embFile, vocab, vocab_size)

    # prepare story
    story = [train_x, train_y, train_idx]
    print "The length of the story ", len(train_x), " ( DEV = ", len(dev_x), " TEST = ", len(test_x), " )"
    test = [test_x, test_y, test_idx]
    dev = [dev_x, dev_y, dev_idx]
    # load game
    print("Loading game ..")
    game = NERGame(story, test, dev, max_len, w2v, budget, split, isload, fn)
    return game


def test_agent_batch(robot, game, model, budget):
    i = 0
    queried_x = []
    queried_y = []
    performance = []
    test_sents = utilities.data2sents(game.test_x, game.test_y)
    while i < budget:
        sel_ind = random.randint(0, len(game.train_x))
        # construct the observation
        #observation = game.getFrame(model)
        observation = game.get_frame(model)
        #action = robot.getAction(observation)
        action = robot.get_action(observation)
        if action[1] == 1:
            sentence = game.train_x[sel_ind]
            labels = game.train_y[sel_ind]
            queried_x.append(sentence)
            queried_y.append(labels)
            i += 1
            train_sents = utilities.data2sents(queried_x, queried_y)
            model.train(train_sents)
            performance.append(model.test(test_sents))
    # train a crf and evaluate it
    train_sents = utilities.data2sents(queried_x, queried_y)
    model.train(train_sents)
    performance.append(model.test(test_sents))
    print "***TEST", performance

def test_agent_batchNew(robot, game, model, budget, selfstudy):
    i = 0
    queried_x = []
    queried_y = []
    performance = []
    game.setRandomOrder()
    test_sents = utilities.data2sents(game.test_x, game.test_y)
    while i < budget and game.currentFrame < len(game.train_x):
        #sel_ind = random.randint(0, len(game.train_x))
        # construct the observation
        #observation = game.getFrame(model)
        observation = game.get_frame(model)
        sentence, labels = game.getCurrentFrameSentence(1)
        #action = robot.getAction(observation)
        action = robot.get_action(observation)
        if action[1] == 1:
            #sentence = game.train_x[sel_ind]
            #labels = game.train_y[sel_ind]
            if i>100 and selfstudy:
               items = sentence.split()
               pl = model.pred([items])
               labels = pl[0]

            queried_x.append(sentence)
            queried_y.append(labels)
            i += 1
            train_sents = utilities.data2sents(queried_x, queried_y)
            model.train(train_sents)
            performance.append(model.test(test_sents))
    # train a crf and evaluate it
    train_sents = utilities.data2sents(queried_x, queried_y)
    model.train(train_sents)
    performance.append(model.test(test_sents))
    print "***TEST", performance
    return performance 

def test_agent_batchRandom(game, model, budget, selfstudy):
    i = 0
    queried_x = []
    queried_y = []
    performance = []
    test_sents = utilities.data2sents(game.test_x, game.test_y)
    while i < budget:
        sel_ind = random.randint(0, len(game.train_x)-1)
        sentence = game.train_x[sel_ind]
        if i<=20 or not selfstudy:
            labels = game.train_y[sel_ind]
        else:
            items = sentence.split()
            pl = model.pred([items])
            labels = [int(x) for x in pl[0]]

        queried_x.append(sentence)
        queried_y.append(labels)
        i += 1
        train_sents = utilities.data2sents(queried_x, queried_y)
        model.train(train_sents)
        performance.append(model.test(test_sents))
    # train a crf and evaluate it
    train_sents = utilities.data2sents(queried_x, queried_y)
    model.train(train_sents)
    performance.append(model.test(test_sents))
    print "***TEST", performance
    return performance

def test_agent_batchRandomNew(game, model, budget, selfstudy):
    i = 0
    sel_ind = 0
    queried_x = []
    queried_y = []
    performance = []
    game.setRandomOrder()
    test_sents = utilities.data2sents(game.test_x, game.test_y)
    while i < budget and sel_ind < len(game.train_x):
        act = random.randint(0, 1)
        if act == 1:
           print 'Selecting instance %d'%sel_ind
           sentence = game.train_x[game.order[sel_ind]]
           labels = game.train_y[game.order[sel_ind]]
           if i>20 and selfstudy:
               items = sentence.split()
               pl = model.pred([items])
               labels = [int(x) for x in pl[0]]

           queried_x.append(sentence)
           queried_y.append(labels)

           train_sents = utilities.data2sents(queried_x, queried_y)
           model.train(train_sents)
           performance.append(model.test(test_sents))
           i += 1
        else:
           print 'Skiping instance %d'%sel_ind  

        sel_ind +=1
    # train a crf and evaluate it
    train_sents = utilities.data2sents(queried_x, queried_y)
    model.train(train_sents)
    performance.append(model.test(test_sents))
    print "***TEST", performance
    return performance

def test_agent_compare(robot, game, model1, model2, budget, selfstudy):
    i1 = 0
    i2 = 0
    queried_x1 = []
    queried_y1 = []
    performance1 = []
    queried_x2 = []
    queried_y2 = []
    performance2 = []
    game.setRandomOrder()
    test_sents = utilities.data2sents(game.test_x, game.test_y)
    while (i1 < budget or i2 < budget) and game.currentFrame < len(game.train_x):
        observation = game.get_frame(model1)
        sentence, labels = game.getCurrentFrameSentence(1)

        #handling model1       
        if i1 < budget: 
	        action = robot.get_action(observation)
        	if action[1] == 1:
                    print 'DQN selects instance %d'%game.currentFrame    
        	    labels1 = labels
	            if i1>20 and selfstudy:
        	       items = sentence.split()
       		       pl = model1.pred([items])
	               labels1 = [int(x) for x in pl[0]]

	            queried_x1.append(sentence)
        	    queried_y1.append(labels1)
        	    train_sents1 = utilities.data2sents(queried_x1, queried_y1)
                    model1.train(train_sents1)
	            performance1.append(model1.test(test_sents))
	            i1 += 1

        #handling model2
        if i2 < budget:
                act = random.randint(0, 1)
                if act == 1:
                    print 'Randomly select instance %d'%game.currentFrame    
                    labels2 = labels
	            if i2>20 and selfstudy:
        	       items = sentence.split()
              	       pl = model2.pred([items])
                       labels2 = [int(x) for x in pl[0]]

                    queried_x2.append(sentence)
                    queried_y2.append(labels2)
                    train_sents2 = utilities.data2sents(queried_x2, queried_y2)
                    model2.train(train_sents2)
                    performance2.append(model2.test(test_sents))
                    i2 += 1
              

    print "***TEST1", performance1
    print "***TEST2", performance2
    return performance1, performance2 


def test_agent_online(robot, game, model, budget):
    # to address game -> we have a new game here
    i = 0
    queried_x = []
    queried_y = []
    performance = []
    test_sents = utilities.data2sents(game.test_x, game.test_y)
    while i < budget:
        sel_ind = random.randint(0, len(game.train_x))
        # construct the observation
        #observation = game.getFrame(model)
        observation = game.get_frame(model)
        #action = robot.getAction(observation)
        action = robot.get_action(observation)
        if action[1] == 1:
            sentence = game.train_x[sel_ind]
            labels = game.train_y[sel_ind]
            queried_x.append(sentence)
            queried_y.append(labels)
            i += 1
            train_sents = utilities.data2sents(queried_x, queried_y)
            model.train(train_sents)
            performance.append(model.test(test_sents))

        reward, observation2, terminal = game.feedback(action, model, False)  # game
        robot.update(observation, action, reward, observation2, terminal)
    # train a crf and evaluate it
    train_sents = utilities.data2sents(queried_x, queried_y)
    model.train(train_sents)
    performance.append(model.test(test_sents))
    print "***TEST", performance

def test_agent_onlineNew(robot, game, model, budget, selfstudy):
    # to address game -> we have a new game here
    i = 0
    queried_x = []
    queried_y = []
    performance = []
    game.setRandomOrder()
    test_sents = utilities.data2sents(game.test_x, game.test_y)
    while i < budget:
        #sel_ind = random.randint(0, len(game.train_x))
        # construct the observation
        sentence, labels = game.getCurrentFrameSentence(0)
        observation = game.get_frame(model)
        action = robot.get_action(observation)
        if action[1] == 1:
            if i>20 and selfstudy:
               items = sentence.split()
               pl = model.pred([items])
               labels = [int(x) for x in pl[0]]


            queried_x.append(sentence)
            queried_y.append(labels)
            i += 1
            train_sents = utilities.data2sents(queried_x, queried_y)
            model.train(train_sents)
            performance.append(model.test(test_sents))

        reward, observation2, terminal = game.feedback(action, model, False)  # game
        robot.update(observation, action, reward, observation2, terminal)
    # train a crf and evaluate it
    train_sents = utilities.data2sents(queried_x, queried_y)
    model.train(train_sents)
    performance.append(model.test(test_sents))
    print "***TEST", performance
    return performance


def play_ner(replay, saved_robot, saved_initdata, nextepisode):
    actions = 2
    global AGENT
    if AGENT == "random":
        robot = RobotRandom(actions)
    elif AGENT == "DQN":
        robot = RobotDQN(actions)
    elif AGENT == "CNNDQN":
        robot = RobotCNNDQN(actions)
    else:
        print "** There is no robot."
        raise SystemExit

    if replay:
       robot.restore(saved_robot)

    global TRAIN_LANG, TRAIN_LANG_NUM, BUDGET, CHECKPOINT
    robot.setCheckPoint(CHECKPOINT)
    for i in range(TRAIN_LANG_NUM):
        train = TRAIN_LANG[i][0]
        test = TRAIN_LANG[i][1]
        dev = TRAIN_LANG[i][2]
        emb = TRAIN_LANG[i][3]
        tagger = TRAIN_LANG[i][4]
        # initilise a NER game
        game = initialise_game(train, test, dev, emb, BUDGET, True, replay, saved_initdata)
        # initialise a decision robot
        robot.update_embeddings(game.w2v)
        # tagger
        model = CRFTagger(tagger)
        model.clean()

        # play game
        episode = 1
        if replay:
           episode = nextepisode

        print(">>>>>> Playing game ..")
        while episode <= MAX_EPISODE:
            #copy the baseline model to the saved model
            #print tagger      
            #shutil.copy('eng.model.baseline', tagger) 

            print '>>>>>>> Current game round ', episode, 'Maximum ', MAX_EPISODE
            observation = game.get_frame(model)
            action = robot.get_action(observation)
            print '> Action', action
            reward, observation2, terminal = game.feedback(action, model, True)
            print '> Reward', reward
            robot.update(observation, action, reward, observation2, terminal)

            
            if terminal == True:
                episode += 1
                print '> Terminal <'
    return robot


def test(robot):
    global TEST_LANG, TEST_LANG_NUM, BUDGET
    for i in range(TEST_LANG_NUM):
        train = TEST_LANG[i][0]
        test = TEST_LANG[i][1]
        dev = TEST_LANG[i][2]
        emb = TEST_LANG[i][3]
        tagger = TEST_LANG[i][4]
        game2 = initialise_game(train, test, dev, emb, BUDGET, False, False, '')
        robot.update_embeddings(game2.w2v)
        model = CRFTagger(tagger)
        model.clean()
        test_agent_batchNew(robot, game2, model, 1000, True)
        #test_agent_onlineNew(robot, game2, model, BUDGET)


def main():
    parse_args()
    # budget = 1000
    # game for training
    global CHECKPOINT, SAVED_ROBOT, REPLAY, NEXTEPISODE
    saved_initdata = CHECKPOINT+'_initdata.txt'
    robot = play_ner(REPLAY, SAVED_ROBOT, saved_initdata, NEXTEPISODE)
    robot.save(CHECKPOINT+'-final')
    # new game 2
    test(robot)

if __name__ == '__main__':
    main()
