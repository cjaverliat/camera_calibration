from torchaudio.transforms import MFCC
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import torch
from rich import print
from rich.table import Column
from rich.progress import (
    Progress,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
    BarColumn,
    SpinnerColumn,
    Group,
    Live,
)
from camera_calibration.utils.ffmpeg_progress_socket import (
    run_ffmpeg_with_progress_info,
)
import ffmpeg
from typing import Iterator


def plot_waveforms(waveforms, sample_rate, titles, main_title):

    fig, axs = plt.subplots(
        len(waveforms), 1, figsize=(12, 6 * len(waveforms)), sharex=True
    )

    for i, (waveform, title) in enumerate(zip(waveforms, titles)):
        axs[i].xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x / sample_rate:.1f}")
        )
        axs[i].plot(waveform[0])
        axs[i].set_title(title)

    fig.suptitle(main_title)
    axs[-1].set_xlabel("Time (s)")
    axs[len(waveforms) // 2].set_ylabel("Amplitude")
    return fig, axs


def plot_unsynced_vs_synced_waveforms(waveforms, sample_rate, time_offsets):
    corrected_waveforms = [
        shift_waveform(waveform, sample_rate, time_offset)
        for waveform, time_offset in zip(waveforms, time_offsets)
    ]

    fig, axs = plt.subplots(len(waveforms), 2, sharex=True)

    for i, (waveform, corrected_waveform, time_offset) in enumerate(
        zip(waveforms, corrected_waveforms, time_offsets)
    ):
        axs[i, 0].xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x / sample_rate:.1f}")
        )
        axs[i, 0].plot(waveform[0])
        axs[i, 0].set_title("Before correction", fontsize=10)

        axs[i, 1].xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x / sample_rate:.1f}")
        )
        axs[i, 1].plot(corrected_waveform[0])
        axs[i, 1].set_title(
            f"After correction  (offset: {time_offset:.2f}s)", fontsize=10
        )

    fig.suptitle("Waveforms before and after synchronization")
    axs[-1, 0].set_xlabel("Time (s)")
    axs[-1, 1].set_xlabel("Time (s)")
    axs[len(waveforms) // 2, 0].set_ylabel("Amplitude")
    return fig, axs


def plot_spectrogram(sg, title=None, ylabel="freq_bin"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    im = ax.imshow(sg.log2()[0], aspect="auto", origin="lower")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig, ax


def plot_cross_correlation(cross_correlation, x_formatter=lambda x: str(x)):
    max_cross_correlation_idx = torch.argmax(cross_correlation)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    im = ax.imshow(cross_correlation[0], aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Time offset (s)")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: x_formatter(x)))
    ax.axvline(max_cross_correlation_idx, color="red", linestyle="--")
    ax.text(
        max_cross_correlation_idx + 1000,
        0,
        x_formatter(max_cross_correlation_idx),
        color="red",
    )

    ax.yaxis.set_visible(False)
    ax.set_title("Cross-correlation between MFCCs")
    return fig, ax


def shift_waveform(waveform, sample_rate, time_offset):
    samples_offset = int(time_offset * sample_rate)
    shifted_waveform = torch.zeros_like(waveform)

    if samples_offset > 0:
        shifted_waveform[..., samples_offset:, :] = waveform[..., :-samples_offset, :]
    elif samples_offset < 0:
        shifted_waveform[..., :samples_offset, :] = waveform[..., -samples_offset:, :]
    else:
        shifted_waveform = waveform
    return shifted_waveform


def _extract_audio(
    video_path: str, target_sample_rate: int = 48000, target_n_channels: int = 2
) -> torch.Tensor:
    """
    Extract audio from a video file and return it as a waveform.

    :param video_path: path to the video file
    :param target_sample_rate: target sample rate for the audio
    :return: audio waveform as a float32 in range [-1; 1] with shape (1, n_samples, n_channels)
    """

    probe = ffmpeg.probe(video_path)
    audio_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "audio"),
        None,
    )

    if audio_stream is None:
        raise ValueError("No audio stream found in video")

    # PCM 16-bit audio = 2 bytes per sample
    pcm_s16_sample_size = 2
    n_bytes = audio_stream["duration_ts"] * target_n_channels * pcm_s16_sample_size

    proc = (
        ffmpeg.input(video_path)
        .output(
            "pipe:",
            format="wav",
            acodec="pcm_s16le",
            ar=target_sample_rate,
            ac=target_n_channels,
        )
        .run_async(pipe_stdout=True, pipe_stderr=False, quiet=True)
    )

    audio_data = bytearray(proc.stdout.read(n_bytes))
    proc.stdout.close()
    proc.wait()

    audio = torch.frombuffer(audio_data, dtype=torch.int16)
    audio = torch.reshape(audio, (1, -1, target_n_channels))
    # Convert to float32 and normalize to range [-1; 1]
    audio = audio / 32768.0
    return audio


def _estimate_audio_time_offset(
    waveform1: torch.Tensor,
    waveform2: torch.Tensor,
    sample_rate: int,
    show_cross_correlation_plot=False,
) -> tuple[float, torch.Tensor]:

    # Keep only the first channel
    waveform1 = waveform1[..., 0]
    waveform2 = waveform2[..., 0]

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

    mfcc1 = mfcc_transform(waveform1).mean(dim=1, keepdim=True)
    mfcc2 = mfcc_transform(waveform2).mean(dim=1, keepdim=True)

    padding_size = mfcc2.shape[-1] - 1
    cross_correlation = torch.nn.functional.conv1d(
        mfcc1, mfcc2, padding=padding_size, stride=1
    )
    max_corr_idx = torch.argmax(cross_correlation)
    offset = max_corr_idx - padding_size

    hops_to_seconds = 1 / mfcc2.shape[-1] * waveform2_duration
    estimated_time_offset = offset * hops_to_seconds

    if show_cross_correlation_plot:
        fig, _ = plot_cross_correlation(
            cross_correlation, lambda x: f"{hops_to_seconds * (x - padding_size):.2f}"
        )
        fig.show()

    return estimated_time_offset, cross_correlation


def _export_synced_video(
    video_path: str, output_path: str, time_offset: float, duration: float
) -> Iterator[tuple[str, float]]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_duration = float(ffmpeg.probe(video_path)["format"]["duration"])

    for _, key, value in run_ffmpeg_with_progress_info(
        ffmpeg.input(video_path)
        .output(
            output_path,
            ss=-time_offset,
            t=duration,
            acodec="copy",
            vcodec="copy",
        )
        .overwrite_output(),
        quiet=True,
    ):
        if key == "out_time_us":
            elapsed_time = float(value) / 1_000_000.0
            yield "progress", elapsed_time / total_duration
        if key == "progress" and value == "end":
            yield "done", None


def _estimate_audio_time_offsets_with_progress(
    waveforms: list[torch.Tensor], sample_rate: int
) -> tuple[list[float], list[torch.Tensor]]:

    job_progress = Progress(
        TextColumn(
            "[bold green]{task.description}",
            justify="right",
        ),
        SpinnerColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    tasks_progress = Progress(
        TextColumn(
            "[blue]{task.description}",
            justify="right",
            table_column=Column(overflow="ellipsis", no_wrap=True),
        ),
        SpinnerColumn(),
        TimeElapsedColumn(),
    )

    job_task = job_progress.add_task("Estimating time offsets ⏱", total=len(waveforms))

    tasks = [tasks_progress.add_task(f"Waveform {i+1}") for i in range(len(waveforms))]

    progress_group = Group(job_progress, tasks_progress)

    with Live(progress_group, refresh_per_second=10, transient=True):
        time_offsets = []
        cross_correlations = []

        for waveform, task in zip(waveforms, tasks):
            time_offset, cross_correlation = _estimate_audio_time_offset(
                waveforms[0], waveform, sample_rate
            )
            time_offsets.append(time_offset)
            cross_correlations.append(cross_correlation)
            tasks_progress.update(task, completed=100)
            tasks_progress.stop_task(task)
            job_progress.update(job_task, advance=1)

        print(
            f"[green]Estimated {len(waveforms)} time offsets relative to [blue][bold]Waveform 1[/]"
        )

        for i, time_offset in enumerate(time_offsets):
            print(
                f"[bold blue]Waveform {i+1}[/bold blue] [white bold]{time_offset:.2f}s"
            )

        return time_offsets, cross_correlations


def _extract_audios_with_progress(
    video_paths: list[str],
    target_sample_rate: int = 48000,
    target_n_channels: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:

    job_progress = Progress(
        TextColumn(
            "[bold green]{task.description}",
            justify="right",
        ),
        SpinnerColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    tasks_progress = Progress(
        TextColumn(
            "[blue]{task.description}",
            justify="right",
            table_column=Column(overflow="ellipsis", no_wrap=True),
        ),
        SpinnerColumn(),
        TimeElapsedColumn(),
    )

    job_task = job_progress.add_task("Extracting audio 🎵", total=len(video_paths))

    tasks = [
        tasks_progress.add_task(os.path.relpath(video_path, os.getcwd()))
        for video_path in video_paths
    ]

    progress_group = Group(job_progress, tasks_progress)

    with Live(progress_group, refresh_per_second=10, transient=True):
        audios = []

        for video_path, task in zip(video_paths, tasks):
            audio = _extract_audio(video_path, target_sample_rate, target_n_channels)
            audios.append(audio)
            tasks_progress.update(task, completed=100)
            tasks_progress.stop_task(task)
            job_progress.update(job_task, advance=1)

        print(f"[green]Extracted {len(video_paths)} audio tracks.[/green]")
        return audios


def _export_synced_videos_with_progress(
    video_paths, output_paths, waveforms, time_offsets, sample_rate
):

    job_progress = Progress(
        TextColumn(
            "[bold green]{task.description}",
            justify="right",
        ),
        SpinnerColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    tasks_progress = Progress(
        TextColumn(
            "[blue]{task.description}",
            justify="right",
            table_column=Column(overflow="ellipsis", no_wrap=True),
        ),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    job_task = job_progress.add_task(
        "Exporting synchronized videos 📦", total=len(video_paths)
    )

    tasks = []

    for i in range(len(video_paths)):
        tasks.append(
            tasks_progress.add_task(os.path.relpath(output_paths[i], os.getcwd()))
        )

    progress_group = Group(job_progress, tasks_progress)

    with Live(progress_group, refresh_per_second=10, transient=True):

        max_time_offset = max(time_offsets)
        time_offsets = [time_offset - max_time_offset for time_offset in time_offsets]
        max_duration = min(
            audios.shape[1] / sample_rate + time_offset
            for audios, time_offset in zip(waveforms, time_offsets)
        )

        for i, (video_path, output_path, time_offset) in enumerate(
            zip(video_paths, output_paths, time_offsets)
        ):
            for status, p in _export_synced_video(
                video_path, output_path, time_offset, max_duration
            ):
                if status == "done":
                    job_progress.update(job_task, advance=1)
                    tasks_progress.update(tasks[i], completed=100)
                    tasks_progress.stop_task(tasks[i])
                elif status == "progress":
                    tasks_progress.update(tasks[i], completed=100 * p)

    print(f"[green]Exported {len(video_paths)} synchronized videos 🎉[/green]")


def sync_videos(
    video_paths: list[str],
    output_paths: list[str],
):
    """
    Synchronize a list of videos to a reference video using audio cross-correlation.

    :param video_paths: list of paths to the video files. The first video is the reference video.
    :param output_paths: list of paths to the output video files
    """

    target_sample_rate = 48000

    waveforms = _extract_audios_with_progress(
        video_paths, target_sample_rate=target_sample_rate, target_n_channels=1
    )

    time_offsets, _ = _estimate_audio_time_offsets_with_progress(
        waveforms, target_sample_rate
    )

    _export_synced_videos_with_progress(
        video_paths, output_paths, waveforms, time_offsets, target_sample_rate
    )


dir = "/mnt/hdd_storage/Charles_JAVERLIAT/Captations Guedelon/Rushs/Avril 2023/Guedelon 19_04_2023/Carriere Luc/"

sync_videos(
    [
        os.path.join(dir, "GoPro1/GX010037.MP4"),
        os.path.join(dir, "GoPro2/GX010035.MP4"),
        os.path.join(dir, "GoPro3/GX010035.MP4"),
        os.path.join(dir, "GoPro4/GX010036.MP4"),
        os.path.join(dir, "GoPro5/GX010034.MP4"),
        os.path.join(dir, "GoPro6/GX010038.MP4"),
    ],
    [
        os.path.join(dir, "synced_videos", "GoPro1.mp4"),
        os.path.join(dir, "synced_videos", "GoPro2.mp4"),
        os.path.join(dir, "synced_videos", "GoPro3.mp4"),
        os.path.join(dir, "synced_videos", "GoPro4.mp4"),
        os.path.join(dir, "synced_videos", "GoPro5.mp4"),
        os.path.join(dir, "synced_videos", "GoPro6.mp4"),
    ],
)
