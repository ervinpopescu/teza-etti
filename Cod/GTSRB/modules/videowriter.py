import subprocess

import ffmpeg
import numpy as np


class VideoWriter:
    def __init__(
        self,
        fn,
        vcodec="libx264",
        fps=60,
        in_pix_fmt="rgb24",
        out_pix_fmt="yuv420p",
        input_args=None,
        output_args=None,
    ):
        self.fn = fn
        self.process: subprocess.Popen = None
        self.input_args = {} if input_args is None else input_args
        self.output_args = {} if output_args is None else output_args
        self.input_args["framerate"] = fps
        self.input_args["pix_fmt"] = in_pix_fmt
        self.output_args["pix_fmt"] = out_pix_fmt
        self.output_args["vcodec"] = vcodec

    def add(self, frame: np.ndarray):
        if self.process is None:
            h, w = frame.shape[:2]
            self.process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    s="{}x{}".format(w, h),
                    **self.input_args,
                )
                .output(self.fn, **self.output_args)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
        self.process.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self):
        if self.process is None:
            return
        self.process.stdin.close()
        self.process.wait()


def vidwrite(fn, images, **kwargs):
    writer = VideoWriter(fn, **kwargs)
    for image in images:
        writer.add(image)
    writer.close()
