"""DH116 渲染环境诊断工具。

集中替代旧的 GLFW、MUJOCO_GL、robosuite Viewer 和源码检查脚本。默认只输出
环境信息；可按需检查 GLFW、定位 Viewer 关键源码或运行短时 GUI。
"""

import argparse
import inspect
import os
import platform
import sys
import time


def print_environment():
    """Report graphics variables and installed package locations."""
    import mujoco
    import robosuite
    from robosuite.renderers.viewer.mjviewer_renderer import MjviewerRenderer

    print("[Environment]")
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "MUJOCO_GL", "PYOPENGL_PLATFORM", "LIBGL_ALWAYS_INDIRECT"):
        print(f"{name}={os.environ.get(name)}")
    print(f"platform={platform.platform()}")
    print(f"python={sys.version.split()[0]}")
    print(f"mujoco={mujoco.__version__} ({mujoco.__file__})")
    print(f"robosuite={getattr(robosuite, '__version__', 'unknown')} ({robosuite.__file__})")
    print(f"viewer_source={inspect.getfile(MjviewerRenderer)}")


def check_glfw():
    """Check whether this process can create a minimal GLFW context."""
    import glfw

    if not glfw.init():
        raise RuntimeError("GLFW initialization failed; check DISPLAY or Wayland configuration")
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(64, 64, "DH116 GLFW check", None, None)
        if window is None:
            raise RuntimeError("GLFW could not create a window")
        glfw.destroy_window(window)
        print("GLFW context check passed.")
    finally:
        glfw.terminate()


def print_viewer_source_matches():
    """Print only the Viewer source lines relevant to initialization and refresh."""
    from robosuite.renderers.viewer.mjviewer_renderer import MjviewerRenderer

    keywords = ("__init__", "launch", "render", "update", "viewer")
    source = inspect.getsource(MjviewerRenderer).splitlines()
    print("[MjviewerRenderer key lines]")
    for line_number, line in enumerate(source, 1):
        if any(keyword in line.lower() for keyword in keywords):
            print(f"{line_number:4d}: {line}")


def run_viewer(seconds, use_dh116):
    """Run a minimal robosuite GUI loop with Panda or Panda + DH116."""
    import numpy as np
    import robosuite as suite

    kwargs = {"gripper_types": "DH116"} if use_dh116 else {}
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        **kwargs,
    )
    try:
        env.reset()
        env.viewer.update()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            env.step(np.zeros(env.action_dim))
            env.viewer.update()
            time.sleep(0.01)
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glfw", action="store_true", help="Create a minimal GLFW context")
    parser.add_argument("--source", action="store_true", help="Print key MjviewerRenderer source lines")
    parser.add_argument("--viewer", action="store_true", help="Run a short GUI viewer check")
    parser.add_argument("--dh116", action="store_true", help="Use DH116 in the GUI check")
    parser.add_argument("--seconds", type=float, default=10.0, help="GUI duration")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seconds <= 0:
        raise ValueError("seconds must be positive")
    print_environment()
    if args.glfw:
        check_glfw()
    if args.source:
        print_viewer_source_matches()
    if args.viewer:
        run_viewer(args.seconds, args.dh116)


if __name__ == "__main__":
    main()
