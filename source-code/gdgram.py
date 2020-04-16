import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram


def pre_emp(x):
    '''
    Apply pre-emphasis to given utterance.
    x	: list or 1 dimensional numpy.ndarray
    '''
    return np.append(x[0], np.asarray(x[1:] - 0.97 * x[:-1], dtype=np.float32))


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


def _group_delay_helper(sig, _nfft):
    b = np.fft.rfft(sig, n=_nfft)
    n_sig = np.multiply(sig, np.arange(1, sig.shape[-1] + 1))
    br = np.fft.rfft(n_sig, n=_nfft)
    return np.divide(br, b + np.finfo(float).eps).real


if __name__ == '__main__':
    x, _ = sf.read('PA_T_0054000.flac', dtype='int16')
    gdgram = compute_gd_gram(x, 1024, 25 * 16, 15 * 16)
    print(gdgram.shape)

    # gdgram = (gdgram-np.min(gdgram))/(np.max(gdgram)-np.min(gdgram))
    # plt.imshow(gdgram[0::10, 0::10], cmap=plt.get_cmap('hot'),aspect='auto')
    print(gdgram.min(), gdgram.max())
    plt.matshow(gdgram, cmap='gray', vmin=-200, vmax=200)
    plt.colorbar()
    plt.show()
