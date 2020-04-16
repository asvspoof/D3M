import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib
import matplotlib.pyplot as plt
from ResNet import DKU_ResNet
from loss import FocalLoss

model_params = {
    'FOCAL_GAMMA': 2,  # gamma parameter for focal loss; if obj is not focal loss, set this to None
    'LOSS_WEIGHT': [1., 9.],
    # 'GRL_LAMBDA': GRL_LAMBDA,
    'NUM_SYSTEMS': 10,
    'NUM_CLASSES': 2,  # binary classification (bonafida) (spoof)
}
audio_dir = 'saliency_audio/'
model_checkpoint = 'test_model/best-eer-ep28-1.037037.pt'
# model_checkpoint = 'results/models/best-eer-ep28-1.037037.pt'

class_names = ['Bonafide','Attack type AA','Attack type AB','Attack type AC']
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = DKU_ResNet(2).to(device)

state = torch.load(model_checkpoint)
model.load_state_dict(state)
model.eval()

feature_type = 'logspec'
from extract_feats import extract_logspec, extract_GDgram

if feature_type == 'logspec':
    extract_func = extract_logspec
elif feature_type == 'GDgram':
    extract_func = extract_GDgram

def preprocess(img):
    transform = T.Compose([
        T.ToTensor(),
    ])
    return transform(img)

def compute_saliency_maps(X, y, model):
    X, y = X.to(device), y.to(device)

    model.eval()
    X.requires_grad_()

    focal_obj = FocalLoss(gamma=model_params['FOCAL_GAMMA'], alpha=model_params['LOSS_WEIGHT'])

    print(X.shape)
    output = model(X)
    loss = focal_obj(output, y)
    loss.backward()
    saliency, _ = torch.max(torch.abs(X.grad), axis=1)

    return saliency


def show_saliency_maps(X, y):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([torch.tensor(x) for x in X], dim=0)
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.cpu().numpy()
    N = len(X)
    for i in range(N):
        saliency_i = saliency[i]
        x = np.squeeze(X[i])
        plt.subplot(2, N, i + 1)
        plt.imshow(x,plt.get_cmap('hsv'),norm= matplotlib.colors.Normalize(vmin=x.min(), vmax=x.max()))
        plt.title(class_names[i])
        # plt.xlabel('time')
        plt.ylabel('frequency bin')
        plt.gca()
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency_i, cmap=plt.cm.hot)
        plt.xlabel('time')
        plt.ylabel('frequency bin')
        plt.gcf().set_size_inches(12, 5)
        np.save(l_utt[i]+'_saliency_map.npy',saliency_i)
    plt.show()


if __name__ == '__main__':
    l_utt = ['PA_D_0000001','PA_D_0005401','PA_D_0005603', 'PA_D_0005805']  # list
    y = [1,0,0,0]

    X = []
    to_nb_time=500
    # preprocess
    for utt in l_utt:
        x = np.load(os.path.join(audio_dir, utt+ '.npy'))
        if len(x.shape)==2:
            x = np.expand_dims(x,axis=0)
        nb_time = x.shape[1]
        if nb_time > to_nb_time:
            start_idx = np.random.randint(0, nb_time - to_nb_time)
            x = x[:, start_idx:start_idx + to_nb_time, :]
        elif nb_time < to_nb_time:
            nb_dup = to_nb_time // nb_time + 1
            x= np.tile(x, (1, nb_dup, 1))[:, :to_nb_time, :]

        x = np.expand_dims(x, axis=0).astype(np.float32)  # add 0 dim for torch
        X.append(x)
    show_saliency_maps(X, y)
