import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.fftpack import dct, idct
from scipy.signal import medfilt


def pre_emp(x):
    '''
    Apply pre-emphasis to given utterance.
    x	: list or 1 dimensional numpy.ndarray
    '''
    return np.append(x[0], np.asarray(x[1:] - 0.97 * x[:-1], dtype=np.float32))


def enframe_and_add_window(sig, sample_rate, add_window='Hamming', frame_size=0.025, frame_stride=0.01):
    frame_length = int(frame_size * sample_rate)
    frame_step = int(frame_stride * sample_rate)
    num_frames = int(np.ceil(np.abs(float(sig.shape[0] - (frame_length - frame_step))) / frame_step))  # 保证至少有一帧
    # compute pad
    pad_length = num_frames * frame_step + frame_length - frame_step - sig.shape[0]
    pad_sig = np.append(sig, np.zeros(pad_length))
    # 用到np.tile
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) \
              + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    # print(indices, indices.shape)
    frames = pad_sig[indices]
    if add_window == 'Hamming':
        frames *= np.hamming(frame_length)
    # 显式实现
    #   frames *= 0.54-0.46 * np.cos(2*np.pi*np.arange(0,frame_length)/frame_length-1)
    return frames


def compute_gd_gram(sig, _nfft, _nperseg, _noverlap, pre_emphasis=True, add_window='hamming'):
    if pre_emphasis:
        sig = pre_emp(sig)
    _frame_shift = _nperseg - _noverlap
    num_frames = int(np.ceil(np.abs(float(sig.shape[0] - (_nperseg - _frame_shift))) / _frame_shift))  # 保证至少有一帧
    # compute pad
    pad_length = num_frames * _frame_shift + _nperseg - _frame_shift - sig.shape[0]
    pad_sig = np.append(sig, np.zeros(pad_length))
    indices = np.tile(np.arange(0, _nperseg), (num_frames, 1)) \
              + np.tile(np.arange(0, num_frames * _frame_shift, _frame_shift), (_nperseg, 1)).T
    # print(indices, indices.shape)
    frames = pad_sig[indices]
    L = np.ceil((len(sig) - _noverlap) / _frame_shift).astype(int)  # make sure one frame
    gdgram = np.zeros((L, _nfft // 2 + 1))
    assert frames.shape[0] == L
    if add_window == 'hamming':
        frames *= np.hamming(_nperseg)
    elif add_window is None:
        pass
    else:
        raise NotImplementedError()
    return _group_delay_helper(frames, _nfft)


def modified_group_delay_feature(sig, rho=0.4, gamma=0.9, frame_length=0.025,
                                 frame_shift=0.010, fs=16000, nfft=1024, pre_emphasis=False, add_window='hamming'):
    '''
    # rho = 0.7 gamma = 0.2
    :param sig: signal array
    :param rho: a parameter to control the shape of modified group delay spectra
    :param gamma: a parameter to control the shape of the modified group delay spectra
    :param num_coeff: the desired feature dimension
    :param frame_shift:
    :return:
    grp_phase: mod gd spectrogram
    cep: modified group delay cepstral feature
   #  ts: time instants at the center of each analysis frame

    please tune gamma for better performance
    '''

    if pre_emphasis:
        sig = pre_emp(sig)
    if add_window:
        frames = enframe_and_add_window(sig, fs, add_window=add_window, frame_size=frame_length,
                                        frame_stride=frame_shift)
    frame_length = int(frame_length * fs)
    frame_shift = int(frame_shift * fs)
    n_frame = frames.shape[0]
    frame_length = frames.shape[1]
    delay_vector = np.arange(1, frame_length + 1)
    delay_frames = frames * delay_vector

    x_spec = np.fft.rfft(frames, n=nfft)
    y_spec = np.fft.rfft(delay_frames, n=nfft)

    x_mag = np.abs(x_spec)
    dct_spec = dct(medfilt(x_mag + 1e-8, kernel_size=5))
    smooth_spec = idct(dct_spec[:, :30], n=nfft // 2 + 1)

    product_spec = (x_spec.real * y_spec.real + x_spec.imag * y_spec.imag)
    grp_phase1 = product_spec / ((np.sign(smooth_spec) * np.abs(smooth_spec) ** (2 * rho)) + np.finfo(float).eps)
    grp_phase = (grp_phase1 / (np.abs(grp_phase1) + np.finfo(float).eps)) * (np.abs(grp_phase1) ** gamma)
    # grp_phase /= np.max(np.abs(grp_phase))
    #
    grp_phase[np.isnan(grp_phase)] = 0
    log_grp_phase = np.sign(grp_phase) * np.log(np.abs(grp_phase) + 1e-8)
    # grp_phase[np.isnan(grp_phase)] = 0
    # cep = dct(grp_phase)
    # cep = cep[1:num_coeff+1,:]

    # plt.imshow(log_grp_phase)
    # plt.show()
    # print('finished')
    return log_grp_phase


def _group_delay_helper(sig, _nfft):
    b = np.fft.rfft(sig, n=_nfft)
    n_sig = np.multiply(sig, np.arange(1, sig.shape[-1] + 1))
    br = np.fft.rfft(n_sig, n=_nfft)
    return np.divide(br, b + np.finfo(float).eps).real


if __name__ == '__main__':
    for fn in ['PA_T_0014040.flac', 'PA_T_0005050.flac', 'PA_T_0054000.flac', 'PA_T_0033750.flac']:
        x, _ = sf.read(fn, dtype='int16')
        # gdgram = compute_gd_gram(x, 1024, 25 * 16, 15 * 16)
        gdgram = modified_group_delay_feature(x)
        print(gdgram.shape)

        # gdgram = (gdgram-np.min(gdgram))/(np.max(gdgram)-np.min(gdgram))
        # plt.imshow(gdgram[0::10, 0::10], cmap=plt.get_cmap('hot'),aspect='auto')
        print(gdgram.min(), gdgram.max())
        plt.matshow(gdgram, cmap=plt.cm.hot)  # 'gray')  # ,vmin=-200,vmax=200)
        plt.colorbar()
        plt.title(fn)
        plt.show()
