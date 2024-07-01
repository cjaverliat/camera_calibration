"""
Adapted from https://github.com/kkroening/ffmpeg-python/blob/master/examples/show_progress.py

Usage:

    total_duration = float(ffmpeg.probe(video_path)["format"]["duration"])

    for key, value in run_with_progress_info(
        ffmpeg.input(video_path).output(
            output_path,
            acodec="copy",
            vcodec="copy",
        )
    ):
        if key == "out_time_us":
            elapsed_time = float(value) / 1_000_000
            print(f"Progress: {elapsed_time / total_duration:.2%}")
        if key == "progress" and value == "end":
            print("Done")
"""

from __future__ import print_function, unicode_literals
import os
import shutil
import socket
import tempfile
import contextlib
import ffmpeg
from subprocess import Popen
from typing import Iterator


@contextlib.contextmanager
def _tmpdir_scope():
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


def run_ffmpeg_with_progress_info(
    out_stream: ffmpeg.nodes.OutputStream,
    pipe_stdin=False,
    pipe_stdout=False,
    pipe_stderr=False,
    quiet=False,
    overwrite_output=False,
    *kwargs,
) -> Iterator[tuple[Popen, str, str]]:
    """Run ffmpeg command and yield progress info."""
    with _tmpdir_scope() as tmpdir:
        socket_filename = os.path.join(tmpdir, "sock")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with contextlib.closing(sock):
            sock.bind(socket_filename)
            sock.listen(1)

            p = out_stream.global_args(
                "-progress", f"unix://{socket_filename}"
            ).run_async(
                pipe_stdin=pipe_stdin,
                pipe_stdout=pipe_stdout,
                pipe_stderr=pipe_stderr,
                quiet=quiet,
                overwrite_output=overwrite_output,
                *kwargs,
            )

            connection, _ = sock.accept()
            data = b""
            try:
                while True:
                    more_data = connection.recv(16)
                    if not more_data:
                        break
                    data += more_data
                    lines = data.split(b"\n")
                    for line_bytes in lines[:-1]:
                        line = line_bytes.decode()
                        parts = line.split("=")
                        key = parts[0] if len(parts) > 0 else None
                        value = parts[1] if len(parts) > 1 else None
                        yield (p, key, value)
                    data = lines[-1]
            finally:
                connection.close()
