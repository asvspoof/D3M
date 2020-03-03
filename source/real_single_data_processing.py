import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from myexp import ex


def get_utt_list(_dir_dataset):
    l_utt = []
    for root, dirs, files in os.walk(_dir_dataset):
        for f in files:
            if os.path.splitext(f)[1] == ".npy":
                l_utt.append(f.split('.')[0])
    return l_utt


def get_utt_and_label_dict(pathToMeta):
    l_utt = []
    d_meta = {}
    with open(pathToMeta, 'r') as f:
        l_meta = f.readlines()
    for line in l_meta:
        _, key, _, _, label = line.strip().split(' ')
        d_meta[key] = 1 if label == 'bonafide' else 0
        l_utt.append(key)

    return l_utt, d_meta


def get_utt_and_label_dict_for_real(pathToMeta):
    l_utt = []
    d_meta = {}
    with open(pathToMeta, 'r') as f:
        l_meta = f.readlines()
    for line in l_meta:
        #  print(line)
        _, key, _, _, _, _, _, _, _, _, _, _, label = line.strip().split('\t')
        d_meta[key] = 1 if label == 'bonafide' else 0
        l_utt.append(key)

    return l_utt, d_meta


def get_utt_and_label_dict_for_PA_system(pathToMeta):
    l_utt = []
    d_meta = {}
    with open(pathToMeta, 'r') as f:
        l_meta = f.readlines()
    for line in l_meta:
        #  print(line)
        _, key, _, label, _ = line.strip().split(' ')
        if label == '-':  # bonafide
            d_meta[key] = 0
        elif label == 'AA':
            d_meta[key] = 1
        elif label == 'AB':
            d_meta[key] = 2
        elif label == 'AC':
            d_meta[key] = 3
        elif label == 'BA':
            d_meta[key] = 4
        elif label == 'BB':
            d_meta[key] = 5
        elif label == 'BC':
            d_meta[key] = 6
        elif label == 'CA':
            d_meta[key] = 7
        elif label == 'CB':
            d_meta[key] = 8
        elif label == 'CC':
            d_meta[key] = 9
        else:
            raise NotImplementedError()

        l_utt.append(key)
    return l_utt, d_meta


def get_utt_and_label_dict_for_Env(pathToMeta):
    l_utt = []
    d_meta = {}
    env_meta = {}
    with open(pathToMeta, 'r') as f:
        l_meta = f.readlines()
    for line in l_meta:
        #  print(line)
        _, key, EnvID, _, label = line.strip().split(' ')
        if EnvID not in env_meta:
            env_meta[EnvID] = len(env_meta)
        print(env_meta)
        d_meta[key] = 1 if label == 'bonafide' else 0
        l_utt.append(key)
    return l_utt, d_meta, env_meta


def split_genu_spoof(l_in, dir_meta, return_dic_meta=False):
    l_gen, l_spo = [], []
    d_meta = {}

    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()
    for line in l_meta:
        _, key, _, _, label = line.strip().split(' ')
        d_meta[key] = 1 if label == 'bonafide' else 0

    for k in d_meta.keys():
        if d_meta[k] == 1:
            l_gen.append(k)
        else:
            l_spo.append(k)

    if return_dic_meta:
        return l_gen, l_spo, d_meta
    else:
        return l_gen, l_spo


def balance_classes(lines_small, lines_big, np_seed):
    '''
    Balance number of sample per class.
    Designed for Binary(two-class) classification.
    :param lines_small:
    :param lines_big:
    :param np_seed:
    :return:
    '''

    len_small_lines = len(lines_small)
    len_big_lines = len(lines_big)
    idx_big = list(range(len_big_lines))

    np.random.seed(np_seed)
    np.random.shuffle(lines_big)
    new_lines = lines_small + lines_big[:len_small_lines]
    np.random.shuffle(new_lines)
    # print(new_lines[:5])

    return new_lines


class Dataset_ASVspoof2019_PA(Dataset):

    def __init__(self, list_IDs, labels, nb_time, base_dir, preload=False):
        '''
        self.list_IDs	: list of strings (each string: utt key)
        self.labels		: dictionary (key: utt key, value: label integer)
        self.nb_time	: integer, the number of timesteps for each mini-batch
        '''
        self.list_IDs = list_IDs
        self.labels = labels
        self.nb_time = nb_time
        self.base_dir = base_dir
        self.audios = None

        if preload:
            self._preload()

    def _preload(self):
        self.audios = []
        '''
        Preload dataset to memory
        :return: 
        '''
        for id in self.list_IDs:
            self.audios.append(np.load(os.path.join(self.base_dir, id + '.npy')))

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        if self.audios is not None:
            # if dataset is preloaded
            X = self.audios[index]
        else:
            X = np.load(os.path.join(self.base_dir, ID + '.npy'))
        # print(X.shape)>>> (1, time, freq)

        nb_time = X.shape[1]
        if nb_time > self.nb_time:
            start_idx = np.random.randint(0, nb_time - self.nb_time)
            X = X[:, start_idx:start_idx + self.nb_time, :]
        elif nb_time < self.nb_time:
            nb_dup = self.nb_time // nb_time + 1
            X = np.tile(X, (1, nb_dup, 1))[:, :self.nb_time, :]

        return X, self.labels[ID]


class Dataset_ASVspoof2019_PA_Multi_Task(Dataset):

    def __init__(self, list_IDs, target1, target2, nb_time, base_dir, preload=False):
        '''
        self.list_IDs	: list of strings (each string: utt key)
        self.labels		: dictionary (key: utt key, value: label integer)
        self.nb_time	: integer, the number of timesteps for each mini-batch
        '''
        self.list_IDs = list_IDs
        self.target1 = target1
        self.target2 = target2
        self.nb_time = nb_time
        self.base_dir = base_dir
        self.audios = None

        if preload:
            self._preload()

    def _preload(self):
        self.audios = []
        '''
        Preload dataset to memory
        :return: 
        '''
        for id in self.list_IDs:
            self.audios.append(np.load(os.path.join(self.base_dir, id + '.npy')))

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        if self.audios is not None:
            # if dataset is preloaded
            X = self.audios[index]
        else:
            X = np.load(os.path.join(self.base_dir, ID + '.npy'))
        # print(X.shape)>>> (1, time, freq)

        nb_time = X.shape[1]
        if nb_time > self.nb_time:
            start_idx = np.random.randint(0, nb_time - self.nb_time)
            X = X[:, start_idx:start_idx + self.nb_time, :]
        elif nb_time < self.nb_time:
            nb_dup = self.nb_time // nb_time + 1
            X = np.tile(X, (1, nb_dup, 1))[:, :self.nb_time, :]

        return X, self.target1[ID], self.target2[ID]

#
# class data_prefetcher(object):
#     def __init__(self, loader):
#         self.loader = iter(loader)
#         self.stream = torch.cuda.Stream()
#         # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
#         # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
#         # With Amp, it isn't necessary to manually convert data to half.
#         # if args.fp16:
#         #     self.mean = self.mean.half()
#         #     self.std = self.std.half()
#         self.preload()
#
#     def preload(self):
#         try:
#             self.next_input, self.next_target1, self.next_target2 = next(self.loader)
#         except StopIteration:
#             self.next_input = None
#             self.next_target1 = None
#             self.next_target2 = None
#             return
#         with torch.cuda.stream(self.stream):
#             self.next_input = self.next_input.cuda(non_blocking=True)
#             self.next_target1 = self.next_target1.cuda(non_blocking=True)
#             self.next_target2 = self.next_target2.cuda(non_blocking=True)
#             # With Amp, it isn't necessary to manually convert data to half.
#             # if args.fp16:
#             #     self.next_input = self.next_input.half()
#             # else:
#             # self.next_input = self.next_input.float()
#             # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
#
#     def next(self):
#         torch.cuda.current_stream().wait_stream(self.stream)
#         input = self.next_input
#         target1 = self.next_target1
#         target2 = self.next_target2
#         self.preload()
#         return input, target1, target2
#
#     def __iter__(self):
#         return self

class data_prefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

    def __iter__(self):
        return self



@ex.capture
def data_loader(train_batch, dev_batch, num_workers, nb_time, train_dir, dev_dir, eval_dir, trainProtocolFile,
                devProtocolFile,
                evalProtocolFile):
    # get 4 utt_lists
    # list_trn = get_utt_list(train_dir)
    # list_dev = get_utt_list(dev_dir)
    # list_eval = get_utt_list(eval_dir)
    #
    # l_gen_trn, l_spo_trn, d_label_trn = split_genu_spoof(l_in=list_trn, dir_meta=trainProtocolFile,
    #                                                      return_dic_meta=True)
    # l_gen_dev, l_spo_dev, d_label_dev = split_genu_spoof(l_in=list_dev, dir_meta=devProtocolFile,
    #                                                      return_dic_meta=True)
    # l_gen_eval, l_spo_eval, d_label_eval = split_genu_spoof(l_in=list_eval, dir_meta=evalProtocolFile,
    #                                                         return_dic_meta=True)
    # del list_trn, list_dev, list_eval

    # Update 2019-7-14: Using weighted CrossEntropyLoss
    # which is particularly useful when you have an unbalanced training set.

    # # get balanced validation utterance list.
    # if len(l_gen_trn) > len(l_spo_trn):
    #     l_train_utt = balance_classes(l_spo_trn, l_gen_trn, np_seed=0)
    # else:
    #     l_train_utt = balance_classes(l_gen_trn, l_spo_trn, np_seed=0)
    # if len(l_gen_dev) > len(l_spo_dev):
    #     l_dev_utt = balance_classes(l_spo_dev, l_gen_dev, np_seed=0)
    # else:
    #     l_dev_utt = balance_classes(l_gen_dev, l_spo_dev, np_seed=0)
    # if len(l_gen_eval) > len(l_spo_eval):
    #     l_eval_utt = balance_classes(l_spo_eval, l_gen_eval, np_seed=0)
    # else:
    #     l_eval_utt = balance_classes(l_gen_eval, l_spo_eval, np_seed=0)
    # del l_gen_trn, l_spo_trn, l_gen_dev, l_spo_dev, l_gen_eval, l_spo_eval

    # define dataset generators
    # l_gen_trn.extend(l_spo_trn)
    # l_trn = l_gen_trn
    # del l_spo_trn
    # l_gen_dev.extend(l_spo_dev)
    # l_dev = l_gen_dev
    # del l_spo_dev
    # l_gen_eval.extend(l_spo_eval)
    # l_eval = l_gen_eval
    # del l_spo_eval

    l_trn, d_label_trn = get_utt_and_label_dict(trainProtocolFile)
    _, d_label2_trn = get_utt_and_label_dict_for_PA_system(trainProtocolFile)
    l_dev, d_label_dev = get_utt_and_label_dict(devProtocolFile)
    _, d_label2_dev = get_utt_and_label_dict_for_PA_system(devProtocolFile)
    # l_eval, d_label_eval = get_utt_and_label_dict(evalProtocolFile)
    # _, d_label2_eval = get_utt_and_label_dict_for_PA_system(evalProtocolFile)

    l_eval, d_label_eval = get_utt_and_label_dict_for_real(evalProtocolFile)

    trainset = Dataset_ASVspoof2019_PA(list_IDs=l_trn,
                                       labels=d_label_trn,
                                       nb_time=nb_time,
                                       base_dir=train_dir)
    train_loader = DataLoader(trainset,
                              batch_size=train_batch,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True
                              )
    devset = Dataset_ASVspoof2019_PA(list_IDs=l_dev,
                                     labels=d_label_dev,
                                     nb_time=nb_time,
                                     base_dir=dev_dir,
                                     preload=False)
    dev_loader = DataLoader(devset,
                            batch_size=dev_batch,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
    evalset = Dataset_ASVspoof2019_PA(list_IDs=l_eval,
                                      labels=d_label_eval,
                                      nb_time=nb_time,
                                      base_dir=eval_dir,
                                      preload=False)
    eval_loader = DataLoader(evalset,
                             batch_size=dev_batch,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    return train_loader, dev_loader, eval_loader


if __name__ == '__main__':
    l_utt = []
    d_meta = {}
    # l_utt,d_meta= get_utt_and_label_dict("/home/student/dyq/anti-spoofing/ASVspoof2019/ASVspoof2019_PA_real/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.real.cm.eval.trl.txt")
    l_utt, d_meta = get_utt_and_label_dict_for_real(
        "/home/student/dyq/anti-spoofing/ASVspoof2019/ASVspoof2019_PA_real/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.real.cm.eval.trl.txt")
    l_utt, d_meta = get_utt_and_label_dict_for_PA_system(
        "/home/student/dyq/anti-spoofing/ASVspoof2019/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt")
    for index in range(len(l_utt)):
        print(l_utt[index] + " " + str(d_meta[l_utt[index]]))
