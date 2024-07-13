'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from utility.load_data import Data
import pickle
from collections import defaultdict
from utility.logger import setup_logger
logger = setup_logger(__name__)

class Elliptic(Data):
    def __init__(self,path,batch_size,arg):
        logger.info('Init Elliptic dataset')
        self.path = path
        self.batch_size = batch_size
        self.args = arg


        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        tot_users, tot_items = set(),set()

        self.exist_users = []
        logger.info("Reading Train File")
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    tot_users.add(uid)
                    tot_items.update(set(items))
                    self.n_train += len(items)

        logger.info("Reading Test File")

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    tot_users.add(uid)
                    tot_items.update(set(items))
                    self.n_test += len(items)

        self.n_pos_items = len(tot_items)
        self.n_users = len(tot_users)
        assert self.n_users == 207
        assert self.n_pos_items == 226
        with open(arg.path_neg_items,'rb') as f:
                    self.neg_receivers,self.neg_items = pickle.load(f)
                    assert len(self.neg_receivers) == 36426
                    assert len(self.neg_items) == 52567
                    #self.neg_items = pickle.load(f)

        self.n_items = self.n_pos_items + len(self.neg_items)
        self.n_users = self.n_users + len(self.neg_receivers)

        self.print_statistics()
        logger.info("Creating R Matrix")
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        
        
        self.train_items, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]

                    for i in train_items:
                        self.R[uid, i] = 1.

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    self.test_set[uid] = test_items
        
        self.data_illicit = self.create_illicit_list() 
        with open(arg.path_licit_datalist,'rb') as f:
            self.data_licit = pickle.load(f)

    def create_illicit_list(self):
        data_illicit = []
        illicit_receivers = list(self.test_set.keys())
        for illicit_receiver in illicit_receivers:
            for illicit_sender in self.test_set[illicit_receiver]:
                data_illicit.append((illicit_receiver,illicit_sender))
        return data_illicit

        

    def negative_pool(self):
        """ Generate negative items for training """
        raise NotImplementedError
    
    def sample(self):
        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user from train
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch
        def sample_neg_items_for_u(u, num):
            return rd.sample(self.neg_items, num)

        """ Sample training instances """
        if self.batch_size <= self.n_users:
            users = rd.sample(self.train_items.keys(), self.batch_size)
        else:
            users = [rd.choice(list(self.train_items.keys())) for _ in range(self.batch_size)]
        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, self.args.n_neg)

        return users, pos_items, neg_items


    
        


        


