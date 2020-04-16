# -*- coding: utf-8 -*-
import librosa
import librosa.display
import numpy as np
import os
import soundfile as sf
from multiprocessing import Process
from scipy.signal import spectrogram

from modgdf import modified_group_delay_feature


def pre_emp(x):
    '''
    Apply pre-emphasis to given utterance.
    x	: list or 1 dimensional numpy.ndarray
    '''
    return np.append(x[0], np.asarray(x[1:] - 0.97 * x[:-1], dtype=np.float32))


def extract_cqtspec(l_utt, ret=False):
    if not ret:
        for line in l_utt:
            # sig, sr = sf.read(line, dtype='int16')
            sig, sr = librosa.load(line, sr=None)  # normalized
            # sig = pre_emp(sig)
            hop_i = 7
            n_octaves = 9
            bins_per_octave = 57
            n_bins = bins_per_octave * n_octaves
            assert (sr == 16000)
            hop_length = 2 ** hop_i
            fmax = sr / 2
            fmin = fmax / (2 ** n_octaves)
            tuning = 0.0
            filter_scale = 1

            cqtspec = np.abs(librosa.cqt(sig, sr=sr, hop_length=hop_length, fmin=fmin, bins_per_octave=bins_per_octave,
                                         n_bins=n_bins))
            cqtspec = np.log(cqtspec + 1e-8)
            cqtspec = np.expand_dims(cqtspec.T, axis=0).astype(np.float32)  # add 0 dim for torch

            # plt.imshow(C, plt.get_cmap('hsv'))
            # plt.ylim(0, 500.0)
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [sec]')
            # plt.title('Constant-Q power spectrum')
            # plt.show()
            dir_base, fn = os.path.split(line)
            dir_base, _ = os.path.split(dir_base)
            fn, _ = os.path.splitext(fn)
            if server == 0:
                if not os.path.exists(os.path.join(dir_base, _dir_name)):
                    os.makedirs(os.path.join(dir_base, _dir_name))
                np.save(os.path.join(dir_base, _dir_name, fn), cqtspec)
            elif server == 1:
                if not os.path.exists(os.path.join(_save_dir, _dir_name)):
                    os.makedirs(os.path.join(_save_dir, _dir_name))
                np.save(os.path.join(_save_dir, _dir_name, fn), cqtspec)

        return

    elif ret:
        feats = []
        for line in l_utt:
            # sig, sr = sf.read(line, dtype='int16')
            sig, sr = librosa.load(line, sr=None)  # normalized
            # sig = pre_emp(sig)
            hop_i = 7
            n_octaves = 9
            bins_per_octave = 57
            n_bins = bins_per_octave * n_octaves
            assert (sr == 16000)
            hop_length = 2 ** hop_i
            fmax = sr / 2
            fmin = fmax / (2 ** n_octaves)
            tuning = 0.0
            filter_scale = 1

            cqtspec = np.abs(librosa.cqt(sig, sr=sr, hop_length=hop_length, fmin=fmin, bins_per_octave=bins_per_octave,
                                         n_bins=n_bins))
            cqtspec = np.log(cqtspec + 1e-8)
            cqtspec = np.expand_dims(cqtspec.T, axis=0).astype(np.float32)  # add 0 dim for torch

            feats += [cqtspec]

        return feats


def extract_GDgram(l_utt, ret=False):
    '''
    Extracts and saves Group Delay Gram
    :param l_utt:
    :return: None
    '''
    if not ret:
        for line in l_utt:
            sig, sr = librosa.load(line, sr=None)  # normalized
            gdgram = modified_group_delay_feature(sig, rho=_rho, gamma=_gamma, nfft=_nfft, frame_length=_frame_length,
                                                  frame_shift=_frame_shift)

            gdgram = np.expand_dims(gdgram, axis=0).astype(np.float32)  # add 0 dim for torch

            # wrong version, see
            # https://dsp.stackexchange.com/questions/54197/compute-group-delay-of-an-audio-file-from-stft/59827#59827

            # for line in l_utt:
            #     X, _ = sf.read(line, dtype='int16')
            #     X = pre_emp(X)
            #     Y = np.array([(j + 1) * X[j] for j in range(X.shape[0])])
            #     f, t, Xspec = spectrogram(x=X,
            #                               fs=_fs,
            #                               window=_window,
            #                               nperseg=_nperseg,
            #                               noverlap=_noverlap,
            #                               nfft=_nfft,
            #                               mode='magnitude')
            #     _, _, X_STFT = spectrogram(x=X,
            #                                fs=_fs,
            #                                window=_window,
            #                                nperseg=_nperseg,
            #                                noverlap=_noverlap,
            #                                nfft=_nfft,
            #                                mode='complex')
            #     _, _, Y_STFT = spectrogram(x=Y,
            #                                fs=_fs,
            #                                window=_window,
            #                                nperseg=_nperseg,
            #                                noverlap=_noverlap,
            #                                nfft=_nfft,
            #                                mode='complex')
            #     print((X_STFT.real * Y_STFT.real + X_STFT.imag * Y_STFT.imag))
            #     gdgram = (X_STFT.real * Y_STFT.real + X_STFT.imag * Y_STFT.imag) / Xspec ** 2
            #     gdgram[np.isnan(gdgram)] = 0
            #     print(gdgram.shape)
            #     plt.matshow(gdgram[0])
            #     plt.ylabel('Frequency [Hz]')
            #     plt.xlabel('Time [sec]')
            #     plt.show()
            #     gdgram = np.expand_dims(gdgram.T, axis=0).astype(np.float32)  # add 0 dim for torch

            dir_base, fn = os.path.split(line)
            dir_base, _ = os.path.split(dir_base)
            fn, _ = os.path.splitext(fn)
            if server == 0:
                if not os.path.exists(os.path.join(dir_base, _dir_name)):
                    os.makedirs(os.path.join(dir_base, _dir_name))
                np.save(os.path.join(dir_base, _dir_name, fn), gdgram)
            elif server == 1:
                if not os.path.exists(os.path.join(_save_dir, _dir_name)):
                    os.makedirs(os.path.join(_save_dir, _dir_name))
                np.save(os.path.join(_save_dir, _dir_name, fn), gdgram)

        return
    elif ret:
        feats = []
        for line in l_utt:
            sig, sr = librosa.load(line, sr=None)  # normalized
            gdgram = modified_group_delay_feature(sig, rho=_rho, gamma=_gamma, nfft=_nfft, frame_length=_frame_length,
                                                  frame_shift=_frame_shift, fs=sr)
            gdgram = np.expand_dims(gdgram, axis=0).astype(np.float32)  # add 0 dim for torch
            feats += [gdgram]

        return feats


def extract_logspec(l_utt, ret=False):
    '''
    Extracts log spectrogram
    '''
    if not ret:
        for line in l_utt:
            utt, _ = sf.read(line, dtype='int16')
            utt = pre_emp(utt)

            _, _, spec = spectrogram(x=utt,
                                     fs=_fs,
                                     window=_window,
                                     nperseg=_nperseg,
                                     noverlap=_noverlap,
                                     nfft=_nfft,
                                     mode=_mode)
            logspec = np.log(spec + 1e-8)  # wrong : use masked array in numpy to avoid take log of zero
            logspec = np.expand_dims(logspec.T, axis=0).astype(np.float32)  # add 0 dim for torch

            dir_base, fn = os.path.split(line)
            dir_base, _ = os.path.split(dir_base)
            fn, _ = os.path.splitext(fn)
            if server == 0:
                if not os.path.exists(os.path.join(dir_base, _dir_name)):
                    os.makedirs(os.path.join(dir_base, _dir_name))
                np.save(os.path.join(dir_base, _dir_name, fn), logspec)
            elif server == 1:
                if not os.path.exists(os.path.join(_save_dir, _dir_name)):
                    os.makedirs(os.path.join(_save_dir, _dir_name))
                np.save(os.path.join(_save_dir, _dir_name, fn), logspec)
        return
    elif ret:
        feats = []
        for line in l_utt:
            utt, sr = sf.read(line, dtype='int16')
            utt = pre_emp(utt)
            _, _, spec = spectrogram(x=utt,
                                     fs=sr,
                                     window=_window,
                                     nperseg=_nperseg,
                                     noverlap=_noverlap,
                                     nfft=_nfft,
                                     mode=_mode)
            logspec = np.log(spec + 1e-8)  # wrong : use masked array in numpy to avoid take log of zero
            logspec = np.expand_dims(logspec.T, axis=0).astype(np.float32)  # add 0 dim for torch
            feats += [logspec]

        return feats


def extract_spectrograms(l_utt):
    '''
    Extracts spectrograms
    '''

    for line in l_utt:
        utt, _ = sf.read(line, dtype='int16')
        # utt = pre_emp(utt)

        _, _, spec = spectrogram(x=utt,
                                 fs=_fs,
                                 window=_window,
                                 nperseg=_nperseg,
                                 noverlap=_noverlap,
                                 nfft=_nfft,
                                 mode=_mode)
        spec = np.expand_dims(spec.T, axis=0).astype(np.float32)  # add 0 dim for torch

        dir_base, fn = os.path.split(line)
        dir_base, _ = os.path.split(dir_base)
        fn, _ = os.path.splitext(fn)
        if not os.path.exists(os.path.join(_save_dir, _dir_name)):
            os.makedirs(os.path.join(_save_dir, _dir_name))
        np.save(os.path.join(_save_dir, _dir_name, fn), spec)
    return


# ======================================================================#
feature_type = 'logspec'  # [GDgram, spec, logspec,cqtspec]
if feature_type == 'GDgram':
    _rho = 0.7
    _gamma = 0.2
_nb_proc = 6  # numer of sub-processes (set 1 for single process)
_fs = 16000  # sampling rate
_window = 'hamming'  # window type
_mode = 'magnitude'  # [psd, complex, magnitude, angle, phase]
_nfft = 1024  # number of fft bins
_frame_length = 0.025  # secs
_frame_shift = 0.010  # secs
_nperseg = int(_frame_length * _fs)
_noverlap = int(_nperseg - _frame_shift * _fs)
# _nperseg = int(32 * _fs * 0.001)  # window length (in ms)
# _noverlap = int(22 * _fs * 0.001)  # window overlap size (in ms)
# _nperseg = int(50 * _fs * 0.001)  # window length (in ms)
# _noverlap = int(30 * _fs * 0.001)  # window shift size (in ms)
server = 0
if server == 0:
    _dir_dataset = '/data/ASVspoof2019/PA/'  # directory of Dataset
    _save_dir = _dir_dataset  # by default
elif server == 1:
    _dir_dataset = '/home/student/dyq/anti-spoofing/ASVspoof2019/PA/ASVspoof2019_PA_dev/'  # directory of Dataset
    # _save_dir = _dir_dataset  # by default
    _save_dir = '/fast/dyq/ASVspoof2019_PA_dev/'

_dir_name = '{}_{}_{}_{}_{}'.format(feature_type, _mode, _nfft, _nperseg, _noverlap)

if __name__ == '__main__':
    # For Debug
    # for fn in ['PA_T_0014040.flac', 'PA_T_0005050.flac', 'PA_T_0054000.flac', 'PA_T_0033750.flac']:
    #     # x, _ = sf.read(fn, dtype='int16')
    #     # gdgram = compute_gd_gram(x, 1024, 25 * 16, 15 * 16)
    #     logspec = extract_logspec([fn])
    #     print(logspec.shape)
    #
    #     # gdgram = (gdgram-np.min(gdgram))/(np.max(gdgram)-np.min(gdgram))
    #     # plt.imshow(gdgram[0::10, 0::10], cmap=plt.get_cmap('hot'),aspect='auto')
    #     print(logspec.min(), logspec.max())
    #     plt.imshow(logspec[0], cmap=plt.cm.cool)  # 'gray')  # ,vmin=-200,vmax=200)
    #     plt.colorbar()
    #     plt.title(fn)
    #     plt.show()
    # exit(0)

    # feature switch
    if feature_type == 'GDgram':
        extract_func = extract_GDgram
    elif feature_type == 'spec':
        extract_func = extract_spectrograms
    elif feature_type == 'logspec':
        extract_func = extract_logspec
    elif feature_type == 'cqtspec':
        extract_func = extract_cqtspec
    else:
        raise NotImplementedError()
    # For debug
    # gdgram = extract_GDgram(['PA_T_0014040.flac'])
    # plt.imshow(gdgram[0])
    # plt.show()
    # exit(0)
    l_utt = []
    for r, ds, fs in os.walk(_dir_dataset):
        for f in fs:
            if os.path.splitext(f)[1] != '.flac': continue
            l_utt.append('/'.join([r, f.replace('\\', '/')]))

    nb_utt_per_proc = int(len(l_utt) / _nb_proc)
    l_proc = []

    for i in range(_nb_proc):
        if i == _nb_proc - 1:
            l_utt_cur = l_utt[i * nb_utt_per_proc:]
        else:
            l_utt_cur = l_utt[i * nb_utt_per_proc: (i + 1) * nb_utt_per_proc]
        l_proc.append(Process(target=extract_func, args=(l_utt_cur,)))
        print('%d' % i)

    for i in range(_nb_proc):
        l_proc[i].start()
        print('start %d' % i)
    for i in range(_nb_proc):
        l_proc[i].join()
