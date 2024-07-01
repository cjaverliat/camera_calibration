import os
import ffmpeg


def extract_frame(video_path, time, output_path):
    # Extract frame at a given time
    (
        ffmpeg.input(video_path, ss=time)
        .output(output_path, vframes=1)
        .run(overwrite_output=True)
    )


if __name__ == "__main__":

    dir = "/mnt/hdd_storage/Charles_JAVERLIAT/Captations Guedelon/Rushs/Avril 2023/Guedelon 19_04_2023/Carriere Luc/synced_videos/"

    cameras_frames_timestamps = []

    for i in range(6):
        video_path = os.path.join(dir, f"GoPro{i+1}.mp4")

        time = "00:01:02"
        output_path = os.path.join(dir, f"GoPro{i+1}_frame.jpg")
        extract_frame(video_path, time, output_path)
