import ffmpeg
import os
import subprocess
from rich.progress import Progress
from rich.progress import BarColumn, TimeRemainingColumn, MofNCompleteColumn
import matplotlib.pyplot as plt
import numpy as np


def extract_frames_timestamps(video_path):

    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    n_frames = int(video_stream["nb_frames"])

    process = subprocess.Popen(
        [f"ffmpeg -i \"{video_path}\" -vf 'showinfo' -vsync 0 -f null -"],
        shell=True,
        stderr=subprocess.PIPE,
        text=True,
    )

    frames_timestamps: list[tuple[int, float]] = []

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    ) as progress:

        task = progress.add_task("Collecting frame timestamps", total=n_frames)

        while True:
            output = process.stderr.readline()
            if output == "" and process.poll() is not None:
                break
            if (
                output
                and "Parsed_showinfo" in output
                and "pts_time:" in output
                and "n:" in output
            ):
                time = float(output.split("pts_time:")[1].strip().split(" ")[0])
                n = int(output.split("n:")[1].strip().split(" ")[0])
                frames_timestamps.append((n, time))
                progress.update(task, advance=1)

        progress.update(task, completed=n_frames)

    process.stderr.close()
    process.wait()
    return frames_timestamps


def plot_cameras_frames_timestamps_hist(cameras_frames_timestamps):

    all_time_diffs = []

    for camera_frames_timestamps in cameras_frames_timestamps:
        n, times = zip(*camera_frames_timestamps)
        time_diffs = [
            round((times[i] - times[i - 1]) * 1_000_000) for i in range(1, len(times))
        ]
        all_time_diffs.append(time_diffs)

    non_empty_bins = set(time_diffs)
    bins = np.linspace(min(non_empty_bins), max(non_empty_bins), 100)

    fig, ax = plt.subplots()
    ax.hist(all_time_diffs, bins=bins, histtype="bar")
    ax.set_xticks(list(non_empty_bins))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    plt.xlabel("Time between frames (us)")
    plt.ylabel("Number of frames")
    plt.title("Time between frames (us)")
    plt.legend([f"GoPro{i+1}" for i in range(6)])
    plt.show()


if __name__ == "__main__":

    dir = "/mnt/hdd_storage/Charles_JAVERLIAT/Captations Guedelon/Rushs/Avril 2023/Guedelon 19_04_2023/Carriere Luc/synced_videos/"

    cameras_frames_timestamps = []

    for i in range(6):
        video_path = os.path.join(dir, f"GoPro{i+1}.mp4")

        frames_timestamps_filepath = os.path.join(dir, f"GoPro{i+1}_frames_ts.txt")

        if os.path.exists(frames_timestamps_filepath):
            camera_frames_timestamps = []

            with open(frames_timestamps_filepath, "r") as f:
                for line in f:
                    n, time = line.strip().split(" ")
                    camera_frames_timestamps.append((int(n), float(time)))
        else:
            camera_frames_timestamps = extract_frames_timestamps(video_path)

            with open(camera_frames_timestamps, "w") as f:
                for n, time in camera_frames_timestamps:
                    f.write(f"{n} {time}\n")

        cameras_frames_timestamps.append(camera_frames_timestamps)

    plot_cameras_frames_timestamps_hist(cameras_frames_timestamps)
    # plot_cameras_frames_timestamps_curves(cameras_frames_timestamps)

    # time = "00:01:02"
    # output_path = os.path.join(dir, f"GoPro{i+1}_frame.jpg")
    # extract_frame(video_path, time, output_path)
