import os
from time import time

import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import apex.amp as amp

from data_processing import data_prefetcher
from myexp import ex

besteer =99


def save_checkpoint(checkpoint_path, model, optimizer, ep, it):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizer’s states and hyperparameters used.
    state = {
        'epoch': ep,
        'iteration': it,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

    return state['epoch'], state['iteration']


@ex.capture
def train(model, device, train_loader, test_loader, optimizer, focal_obj, model_params, epoch, save_dir, save_interval,
          writer, test_first=False, load_model=False,
          load_file=None, log_interval=100):
    ep_start = 0
    iteration = 0
    if load_model:
        ep_start, iteration = load_checkpoint(load_file, model, optimizer)
    if test_first:
        print('Start initial test: ')
        test(model, device, test_loader, -1, focal_obj=focal_obj)  # evaluate at the first time

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
    for ep in range(ep_start + 1, epoch):
        model.train()  # set training mode

        start = time()
        for batch_idx, (data, target) in enumerate(train_loader):
            correct = 0
            # bring data to the computing device, e.g. GPU
            data, target = data.to(device), target.to(device)

            # forward pass
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = 100. * correct / len(target)

            # compute loss: negative log-likelihood
            # loss = F.nll_loss(output, target)
            if focal_obj is None:
                criterion = nn.CrossEntropyLoss(torch.tensor(model_params['LOSS_WEIGHT']).to(device))
            else:
                criterion = focal_obj
            loss = criterion(output, target)
            # backward pass
            # clear the gradients of all tensors being optimized
            optimizer.zero_grad()
            # accumulate (i.e. add) the gradients from this forward pass

            # using apex.amp for mixed precision training
            # loss.backward() becomes:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # for param in model.parameters():
            #     print(param.grad.data.sum())

            # # start debugger
            # import pdb
            # pdb.set_trace()

            # performs a single optimization step (parameter update)
            optimizer.step()

            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f} Acc: {:.2f}%'.format(
                    ep, batch_idx * len(data), len(train_loader.dataset),
                        100 * batch_idx / len(train_loader), loss.item(), acc
                ))
                # log using SummaryWriter
                info = {'train_accuracy': acc}
                for tag, value in info.items():
                    writer.add_scalar(tag, value, iteration)

                # different from before: saving model checkpoints
                # if iteration % save_interval == 0 and iteration > 0:
                #   save_checkpoint(save_dir + 'LCNN-%i.pth' % iteration, model, optimizer, ep, iteration)
            iteration += 1
        # save checkpoint per epoch
        save_checkpoint(save_dir + 'Model-epoch-%i.pth' % ep, model, optimizer, ep, iteration)

        end = time()
        print('Time consuming: %.2fs' % (end - start))

        eer = test(model, device, test_loader, ep, writer=writer, focal_obj=focal_obj)  # evaluate at the end of epoch
        scheduler.step(eer)
    # save the final model
    save_checkpoint(save_dir + 'Model-%i.pth' % iteration, model, optimizer, epoch - 1, iteration - 1)


@ex.capture
def test(model, device, test_loader, ep, save_dir, writer, focal_obj, model_params):
    model.eval()
    test_loss = 0
    correct = 0
    y_score = []
    y = []
    print('length of devset: ' + str(len(test_loader.dataset)))
    start = time()
    with torch.no_grad():
        for data, target in test_loader:
            y.extend(list(target))
            data, target = data.to(device), target.to(device)
            output = model(data)
            # y_score.extend(output.cpu().numpy()[:, 1])
            # print(type(output.cpu().numpy()[:, 1]))
            loglikelihood = nn.LogSoftmax(dim=1)(output).cpu().numpy()
            y_score.extend(loglikelihood[:, 1] - loglikelihood[:, 0])
            # np.append(y_score, np.log(output.cpu().numpy()[:, 1])-np.log(output.cpu().numpy()[:, 0]))
            if focal_obj is None:
                criterion = nn.CrossEntropyLoss(torch.tensor(model_params['LOSS_WEIGHT']).to(device))
            else:
                criterion = focal_obj
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    end = time()
    test_loss /= len(test_loader.dataset)

    print('\nDev set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\nTime Consuming: {:.2f}s'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), (end - start)))

    # calculate EER
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + 'epoch%i' % ep, 'w') as f_res:
        for _s, _t in zip(y_score, y):
            f_res.write('{score} {target}\n'.format(score=_s, target=_t))
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - interp1d(fpr, tpr)(x) - x, 0., 1.) * 100
    print('EER: %f' % eer)
    ex.info['EER'] = eer

    # log using SummaryWriter
    info = {'dev_loss': test_loss.item(), 'dev_accuracy': 100. * correct / len(test_loader.dataset),
            'dev_EER': eer}

    for tag, value in info.items():
        writer.add_scalar(tag, value, ep)

    # record best validation model

    global besteer

    best_eer = ex.info['best_eer'] = besteer
    print("Current Best EER: " + str(best_eer))

    if float(eer) < best_eer:
        besteer = best_eer = float(eer)
        ex.info['best_eer'] = best_eer
        print('New best EER: %f' % float(eer))
        if not os.path.exists(save_dir + 'models'):
            os.makedirs(save_dir + 'models')
        dir_best_model_weights = save_dir + 'models/%d-%.6f.h5' % (ep, eer)
        # save best model
        # assert singleGPU
        torch.save(model.state_dict(), save_dir + 'models/best-eer-ep%d-%.6f.pt' % (ep, eer))

    return eer


@ex.capture
def evaluate(model, device, data_loader, load_file, save_dir, focal_obj, model_params):
    print('Evaluation start!')
    state = torch.load(load_file)
    model.load_state_dict(state)
    model.eval()

    test_loss = 0
    correct = 0
    y_score = []
    y = []

    # data prefetcher slightly reduces the time cost of data fetching on HDD
    prefetcher = data_prefetcher(data_loader)
    data, target = prefetcher.next()

    print('Length of Evalset: ' + str(len(data_loader.dataset)))
    start = time()
    with torch.no_grad():
        while data is not None:
            y.extend(list(target.cpu().numpy()))
            data, target = data.to(device), target.to(device)
            output = model(data)
            # y_score.extend(output.cpu().numpy()[:, 1])
            loglikelihood = nn.LogSoftmax(dim=1)(output).cpu().numpy()
            y_score.extend(loglikelihood[:, 1] - loglikelihood[:, 0])
            if focal_obj is None:
                criterion = nn.CrossEntropyLoss(torch.tensor(model_params['LOSS_WEIGHT']).to(device))
            else:
                criterion = focal_obj
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            data, target = prefetcher.next()
    end = time()
    test_loss /= len(data_loader.dataset)

    print('\nEval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\nTime Consuming: {:.2f}s'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset), (end - start)))

    # calculate EER

    if not os.path.exists(save_dir + 'evals'):
        os.makedirs(save_dir + 'evals')

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - interp1d(fpr, tpr)(x) - x, 0., 1.) * 100
    print('EER: %f' % eer)
    with open(save_dir + 'evals/' + 'eval_eer_%f' % eer, 'w') as f_res:
        for _s, _t in zip(y_score, y):
            f_res.write('{score} {target}\n'.format(score=_s, target=_t))
    print('The end. Bye!')


