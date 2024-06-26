from torchaudio.transforms import MFCC
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import torch
import time
from functools import partial


def plot_spectrogram(sg, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    im = ax.imshow(sg.log2()[0], aspect="auto", origin="lower")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)


def shift_waveform(waveform, sample_rate, time_offset):
    samples_offset = int(time_offset * sample_rate)
    shifted_waveform = torch.zeros_like(waveform)
    shifted_waveform[:, samples_offset:] = waveform[:, :-samples_offset]
    return shifted_waveform


def estimate_time_offset(waveform1, waveform2, sample_rate):

    hop_length = 200
    n_fft = 2048
    n_mels = 256
    n_mfcc = 16

    mfcc_transform = MFCC(
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "mel_scale": "htk",
        },
    )

    waveform2_duration = waveform2.shape[1] / sample_rate

    mfcc1 = mfcc_transform(waveform1)
    mfcc2 = mfcc_transform(waveform2)

    pad_size = mfcc2.shape[2]
    corr = torch.nn.functional.conv1d(mfcc1, mfcc2, padding=pad_size, stride=1)
    max_corr_idx = torch.argmax(corr)
    offset = max_corr_idx - mfcc2.shape[2]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    im = ax.imshow(corr[0], aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Time offset (s)")

    factor = 1 / mfcc2.shape[2] * waveform2_duration

    estimated_time_offset = offset * factor

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{factor * (x - mfcc2.shape[2]):.1f}")
    )
    ax.yaxis.set_visible(False)
    ax.axvline(max_corr_idx, color="red", linestyle="--")
    ax.text(max_corr_idx + 2, 0, f"{estimated_time_offset:.2f}", color="red")
    ax.set_title("Cross-correlation between MFCCs")

    plt.show()

    return estimated_time_offset


def show_corrected_waveforms(reference_waveform, unsynced_waveform, time_offset):
    samples_offset = int(time_offset * sample_rate)
    corrected_waveform = torch.roll(unsynced_waveform, samples_offset, dims=1)

    fig, axs = plt.subplots(3)
    axs[0].plot(reference_waveform[0], label="Reference waveform")
    axs[0].set_title("Reference waveform")

    axs[1].plot(unsynced_waveform[0], label="Shifted waveform")
    axs[1].set_title("Unsynced waveform")

    axs[2].plot(reference_waveform[0], label="Reference waveform")
    axs[2].plot(corrected_waveform[0], label="Synced waveform")
    axs[2].legend()
    plt.show()


waveform1, sample_rate = torchaudio.load(
    os.path.join(os.path.dirname(__file__), "data/gopro1.aac")
)
waveform2, sample_rate = torchaudio.load(
    os.path.join(os.path.dirname(__file__), "data/gopro2.aac")
)

start = time.time()
estimated_time_offset = estimate_time_offset(waveform1, waveform2, sample_rate)
print(f"Time taken: {time.time() - start:.2f}s")
print(f"Estimated time offset: {estimated_time_offset}")

show_corrected_waveforms(waveform1, waveform2, estimated_time_offset)

# sample_audio_path = os.path.join(os.path.dirname(__file__), "data/audio.wav")
# waveform, sample_rate = torchaudio.load(sample_audio_path)
# gt_time_offset = 1
# shifted_waveform = shift_waveform(waveform, sample_rate, gt_time_offset)
# estimated_time_offset = estimate_time_offset(waveform, shifted_waveform, sample_rate)
# error = torch.abs(-gt_time_offset - estimated_time_offset)
# print(f"Time offset error: {error * 1000:.2f}ms")
# show_corrected_waveforms(waveform, shifted_waveform, estimated_time_offset)
