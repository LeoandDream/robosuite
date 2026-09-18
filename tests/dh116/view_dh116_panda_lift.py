"""Panda + DH116 无头视频录制工具。

通过 EGL 获取 ``sideview`` 图像，并使用 FFmpeg 编码为 H.264 MP4；默认输出到
当前目录的 ``vedio/dh116_panda_lift.mp4``，参数可调整时长、帧率和分辨率。
"""

import argparse
import os
import subprocess
from pathlib import Path

# Use EGL by default so the script can run without an X display.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import robosuite as suite
import robosuite.macros as macros


# Video encoders expect frames with the OpenCV (top-left origin) convention.
macros.IMAGE_CONVENTION = "opencv"


def parse_args():
    default_output = Path(__file__).resolve().parent / "vedio" / "dh116_panda_lift.mp4"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output, help="Output MP4 path")
    # sideview 可以同时看到机械臂基座、完整关节链、DH116、桌面和方块。
    parser.add_argument("--camera", default="sideview", help="Robosuite camera name")
    parser.add_argument("--seconds", type=float, default=10.0, help="Video duration")
    parser.add_argument("--fps", type=int, default=20, help="Output video frame rate")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seconds <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("seconds, fps, width, and height must all be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("width and height must be even for yuv420p video")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, round(args.seconds * args.fps))

    print("=" * 80)
    print("              DH116 HEADLESS VIDEO RENDER")
    print("=" * 80)
    print(f"camera={args.camera}, frames={frame_count}, fps={args.fps}")

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        gripper_types="DH116",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        control_freq=args.fps,
        horizon=frame_count,
        ignore_done=True,
    )

    ffmpeg_process = None
    try:
        observation = env.reset()
        robot = env.robots[0]
        arm = robot.arms[0]
        gripper = robot.gripper[arm]
        gripper_action_name = robot.get_gripper_name(arm)
        image_key = f"{args.camera}_image"

        ffmpeg_process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{args.width}x{args.height}",
                "-framerate",
                str(args.fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.output),
            ],
            stdin=subprocess.PIPE,
        )
        for frame_index in range(frame_count):
            action = robot.create_action_vector(
                {gripper_action_name: np.full(gripper.dof, 0.5)}
            )
            observation, _, _, _ = env.step(action)
            frame = np.ascontiguousarray(observation[image_key], dtype=np.uint8)
            ffmpeg_process.stdin.write(frame.tobytes())

            if (frame_index + 1) % args.fps == 0 or frame_index + 1 == frame_count:
                print(f"Rendered {frame_index + 1}/{frame_count} frames")
    finally:
        return_code = 0
        if ffmpeg_process is not None:
            if ffmpeg_process.stdin is not None:
                ffmpeg_process.stdin.close()
            return_code = ffmpeg_process.wait()
        env.close()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg exited with status {return_code}")

    print(f"Video saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
