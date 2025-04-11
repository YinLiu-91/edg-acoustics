"""This module provides postprocessing functionalities for the edg_acoustics package.

The current version of edg_acoustics.postprocessing processes the simulated microphone data.
"""

from __future__ import annotations
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt

__all__ = ["plot_rec", "compute_transfer_function"]


def plot_rec(
    prec,
    fs: float,
    id: float = 1,
    duration: float = None,
    ymax: float = None,
    ymin: float = None,
    f_low: float = 20,
    f_high: float = 1000,
    dB_min: float = -50,
    title_name: str = "",
):
    """Plot the microphone data in both time and frequency domain, similar to ITA-Toolbox plotWaveform.

    Args:
        prec (torch.Tensor or numpy.ndarray): microphone data.
        fs (float): sampling frequency.
        id (float): id of microphone location.
        duration (float, optional): duration of time to be plotted.
        ymax (float, optional): maximum of y axis in time domain.
        ymin (float, optional): minimum of y axis in time domain.
        f_low (float, optional): lowest frequency to be plotted.
        f_high (float, optional): highest frequency to be plotted.
        dB_min (float, optional): lowest dB value to be plotted.
        title_name (str, optional): title of the plot.
    """
    # 确保输入是PyTorch张量或NumPy数组
    if isinstance(prec, torch.Tensor):
        prec_np = prec.detach().cpu().numpy()
    else:
        prec_np = prec

    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    if duration is None:
        duration = prec_np.shape[1] / fs
    time_range = np.arange(0, duration, 1 / fs)
    plt.plot(time_range, prec_np[int(id - 1), 0: (len(time_range))])
    plt.grid(True)
    plt.xlim([0, duration])
    if ymax is not None:
        plt.ylim([ymin, ymax])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Time Domain Signal {title_name}")

    plt.subplot(212)
    NFFT = int(2 ** np.ceil(np.log2(len(time_range))))
    freq_range = np.fft.rfftfreq(NFFT, 1 / fs)
    fft_data = np.fft.rfft(prec_np[int(id - 1), 0: (len(time_range))], NFFT)
    plt.semilogx(freq_range, 20 * np.log10(np.abs(fft_data)))
    plt.grid(True)
    plt.xlim([f_low, f_high])
    plt.ylim([dB_min, np.max(20 * np.log10(np.abs(fft_data))) + 10])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.title("Frequency Domain Signal")

    plt.tight_layout()
    plt.show()


def compute_transfer_function(
    rec_sig,
    mic_id_1: int,
    mic_id_2: int,
    dt: float,
    fmin: float = 60,
    fmax: float = 200,
    causal: bool = True,
):
    """Compute the transfer function between two microphone signals.

    Args:
        rec_sig (torch.Tensor or numpy.ndarray): microphone data. Dimension 1 corresponds to the microphone id. Dimension 2 corresponds to the time step.
        mic_id_1 (int): id of microphone location 1.
        mic_id_2 (int): id of microphone location 2.
        dt (float): time step.
        fmin (float, optional): lowest frequency to be used for coherence computation.
        fmax (float, optional): highest frequency to be used for coherence computation.
        causal (bool, optional): whether the transfer function should be causal or acausal.

    Returns:
        tf (numpy.ndarray): transfer function from microphone 1 to microphone 2.
        mean_coh (float): mean coherence between microphone 1 and microphone 2, for frequencies from fmin to fmax.
        f (numpy.ndarray): frequency array.
        COH (numpy.ndarray): coherence array.
    """
    # 确保输入是PyTorch张量或NumPy数组
    if isinstance(rec_sig, torch.Tensor):
        rec_sig_np = rec_sig.detach().cpu().numpy()
    else:
        rec_sig_np = rec_sig

    fs = 1 / dt  # sampling frequency
    x = rec_sig_np[mic_id_1 - 1, :]  # mic 1 signal
    y = rec_sig_np[mic_id_2 - 1, :]  # mic 2 signal

    # filter signals between fmin and fmax
    b, a = scipy.signal.butter(
        1, [2 * fmin / fs, 2 * fmax / fs], btype="bandpass")
    x = scipy.signal.filtfilt(b, a, x)
    y = scipy.signal.filtfilt(b, a, y)

    N = len(x)
    NFFT = 2 ** int(np.ceil(np.log2(N)))  # next power of 2

    # compute FFT
    X = np.fft.rfft(x, NFFT)
    Y = np.fft.rfft(y, NFFT)

    # compute cross spectral density
    PXY = X * np.conjugate(Y)

    # compute auto spectral density
    PXX = X * np.conjugate(X)
    PYY = Y * np.conjugate(Y)

    # compute coherence
    COH = np.abs(PXY) ** 2 / (PXX * PYY)

    # transfer function
    H = PXY / PXX

    # frequency array
    f = np.fft.rfftfreq(NFFT, dt)

    # compute mean coherence
    freq_band = (f >= fmin) & (f <= fmax)
    if np.sum(freq_band) > 0:
        mean_coh = np.mean(COH[freq_band])
    else:
        mean_coh = 0

    # compute transfer function in time domain
    if causal:
        # causal transfer function
        h = np.fft.irfft(H, NFFT)
        return h, mean_coh, f, COH
    else:
        # acausal transfer function (in frequency domain)
        return H, mean_coh, f, COH
