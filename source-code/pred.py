import os
import torch
import torch.nn as nn

from ResNet import DKU_ResNet


def predict(model_checkpoint, dir_data):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = DKU_ResNet(2).to(device)

    state = torch.load(model_checkpoint)
    model.load_state_dict(state)
    model.eval()

    l_utt = []
    test_li = []
    feature_type = 'GDgram'
    from extract_feats import extract_logspec,extract_GDgram
    if feature_type == 'logspec':
        extract_func = extract_logspec
    elif feature_type == 'GDgram':
        extract_func = extract_GDgram
    for r, ds, fs in os.walk(dir_data):
        for f in fs:
            # if os.path.splitext(f)[1] != '.wav': continue
            test_li += [f]
            l_utt.append('/'.join([r, f.replace('\\', '/')]))
    feats = extract_func(l_utt,ret=True)
    out = []
    y_score = []
    res = []
    like=[]
    import numpy as np
    with torch.no_grad():
        for sample in feats:
            sample = np.expand_dims(sample,axis=0)
            sample = torch.from_numpy(sample).to(device)
            output = model(sample)
            pred = output.argmax(dim=1, keepdim=True)
            loglikelihood = nn.LogSoftmax(dim=1)(output).cpu().numpy()
            score = loglikelihood[:, 1] - loglikelihood[:, 0]
            y_score+=[score]
            like+=[loglikelihood]
            out += [output.cpu().numpy()]
            res += [pred.cpu().numpy()]
    for i in range(len(test_li)):
        print(test_li[i], out[i], like[i], y_score[i],res[i])


if __name__ == '__main__':
    model_ckpath = 'results/models/best-eer-ep28-0.724280.pt'
    audio_dir = 'audio'
    predict(model_ckpath, audio_dir)

