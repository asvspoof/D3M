import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import apex.amp as amp

from data_processing import data_loader
from ResNet import DKU_SEResNet
from myexp import ex
from utils import train,test,evaluate
from loss import FocalLoss
@ex.config
def my_config():
    access_type = 'PA'
    # feature_type = 'LFCC'

    # set paths to the wave files and protocols
    feature_type = 'GDgram'  # [GDgram, spec, logspec]
    print('training with feature: {}'.format(feature_type))
    server = 1
    if server == 1:  # local test
        pathToASVspoof2019Data = '/data/ASVspoof2019/'
        _dir_dataset = os.path.join(pathToASVspoof2019Data, access_type)
        train_dir = os.path.join(_dir_dataset,'ASVspoof2019_PA_train/{}_magnitude_1024_400_240/'.format(feature_type))
        dev_dir = os.path.join(_dir_dataset,'ASVspoof2019_PA_dev/{}_magnitude_1024_400_240/'.format(feature_type))
        eval_dir = os.path.join(_dir_dataset, 'ASVspoof2019_PA_eval/{}_magnitude_1024_400_240/'.format(feature_type))
    elif server == 0:
        pathToASVspoof2019Data = '/data/dyq/anti-spoofing/ASVspoof2019/'
        _dir_dataset = os.path.join(pathToASVspoof2019Data, access_type)
        train_dir = os.path.join('/fast/dyq/ASVspoof2019_PA_train', '{}_magnitude_1024_400_240/'.format(feature_type))
        dev_dir = os.path.join('/fast/dyq/ASVspoof2019_PA_dev', '{}_magnitude_1024_400_240/'.format(feature_type))
        eval_dir = os.path.join(_dir_dataset, 'ASVspoof2019_PA_eval/{}_magnitude_1024_400_240/'.format(feature_type))

    # train_dir = os.path.join('/fast/dyq/ASVspoof2019_PA_train', 'spec_magnitude_1024_400_240/')
    # dev_dir = os.path.join('/fast/dyq/ASVspoof2019_PA_dev', 'spec_magnitude_1024_400_240/')

    trainProtocolFile = os.path.join(_dir_dataset, 'ASVspoof2019_' + access_type + '_cm_protocols',
                                     'ASVspoof2019.' + access_type + '.cm.train.trn.txt')
    devProtocolFile = os.path.join(_dir_dataset, 'ASVspoof2019_' + access_type + '_cm_protocols',
                                   'ASVspoof2019.' + access_type + '.cm.dev.trl.txt')
    evalProtocolFile = os.path.join(_dir_dataset, 'ASVspoof2019_' + access_type + '_cm_protocols',
                                    'ASVspoof2019.' + access_type + '.cm.eval.trl.txt')
    save_dir = 'results/'
    batch_size = 16
    train_batch = 16
    dev_batch = 48
    nb_time = 500
    epoch = 30
    save_interval = 1000
    log_interval = 200
    lr = 0.0005
    load_model = False
    load_file = save_dir + 'LCNN-1000.pth'
    test_first = False
    eval_mode = False
    num_workers = 1
    model_params = {
        'FOCAL_GAMMA' : 2,  # gamma parameter for focal loss; if obj is not focal loss, set this to None
        'LOSS_WEIGHT': [1., 9.]
    }

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@ex.automain
def main(model_params,writer, eval_mode, load_file,seed):
    setup_seed(seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use", device)
    model = DKU_SEResNet(2).to(device)
    #define loss function
    if model_params['FOCAL_GAMMA']:
        print('Training with focal loss')
        focal_obj = FocalLoss(gamma=model_params['FOCAL_GAMMA'], alpha=model_params['LOSS_WEIGHT'])
    else:
        focal_obj = None
    optimizer = optim.AdamW(model.parameters(), weight_decay = 5e-5)

    # Allow Amp to perform casts as required by the opt_level
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    train_loader, dev_loader, eval_loader = data_loader()
    if eval_mode is False:
        train(model, device, train_loader, dev_loader, optimizer,focal_obj=focal_obj)
    else:
        evaluate(model, device, eval_loader, load_file,focal_obj=focal_obj)

    writer.close()


'''
command:  
python main.py with 'epoch=50' 'lr=0.001'  'load_model=False' 'load_file=results/Model-epoch-25.pth' 'test_first=False' 'num_workers=1' 'eval_mode=False'
'''