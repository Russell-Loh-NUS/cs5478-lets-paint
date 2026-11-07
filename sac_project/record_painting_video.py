#!/usr/bin/env python3
"""Record PyBullet painting with safe SIGINT/SIGTERM shutdown."""

import argparse
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p


class StopRecording(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall", default="north")
    ap.add_argument("--output", default="sac_project/runs/north_wall_painting.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--capture-every", type=int, default=12)
    ap.add_argument("--camera-wait", type=float, default=20)
    args = ap.parse_args()

    if args.fps <= 0 or args.capture_every <= 0:
        ap.error("FPS and capture interval must be positive")

    import imageio.v2 as imageio
    import imageio_ffmpeg

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".mkv")

    demo = Path(__file__).resolve().parent / "four_wall_coverage_demo.py"

    writer = None
    frames = 0
    physics_steps = 0
    camera_ready = False
    stop_requested = False

    original_step = p.stepSimulation
    original_connect = p.connect
    original_disconnect = p.disconnect
    original_argv = sys.argv[:]

    camera = {
        "distance": 5.5,
        "yaw": 45.0,
        "pitch": -30.0,
        "target": [1.8, 1.2, 1.0],
    }

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("\n[video] Stop requested; finalising recording...", flush=True)
        raise StopRecording()

    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def read_camera():
        try:
            info = p.getDebugVisualizerCamera()
            if len(info) >= 12:
                camera["yaw"] = float(info[8])
                camera["pitch"] = float(info[9])
                camera["distance"] = float(info[10])
                camera["target"] = list(info[11])
                return True
        except Exception:
            pass
        return False

    def capture():
        nonlocal writer, frames

        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera["target"],
            distance=camera["distance"],
            yaw=camera["yaw"],
            pitch=camera["pitch"],
            roll=0,
            upAxisIndex=2,
        )

        projection = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=args.width / args.height,
            nearVal=0.05,
            farVal=100,
        )

        image = p.getCameraImage(
            args.width,
            args.height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
        )

        rgb = np.asarray(image[2], dtype=np.uint8).reshape(
            args.height, args.width, 4
        )[:, :, :3]

        if writer is None:
            writer = imageio.get_writer(
                str(temporary),
                format="FFMPEG",
                fps=args.fps,
                codec="libx264",
                quality=7,
                macro_block_size=16,
                output_params=["-f", "matroska"],
            )

        writer.append_data(rgb)
        frames += 1

        if frames % 100 == 0:
            print(f"[video] frames={frames}", flush=True)

    def recorded_step(*a, **kw):
        nonlocal physics_steps, camera_ready

        if stop_requested:
            raise StopRecording()

        if not camera_ready:
            print(
                "\n[video] Adjust the PyBullet camera now."
                f"\n[video] Waiting {args.camera_wait:g} seconds...",
                flush=True,
            )

            deadline = time.monotonic() + args.camera_wait
            while time.monotonic() < deadline:
                time.sleep(min(0.1, deadline - time.monotonic()))

            if read_camera():
                print("[video] Camera saved.", flush=True)
            else:
                print("[video] Using default camera.", flush=True)

            camera_ready = True

        result = original_step(*a, **kw)
        physics_steps += 1

        if physics_steps % args.capture_every == 0:
            capture()

        return result

    def recorded_connect(*a, **kw):
        return original_connect(*a, **kw)

    def recorded_disconnect(*a, **kw):
        return original_disconnect(*a, **kw)

    p.stepSimulation = recorded_step
    p.connect = recorded_connect
    p.disconnect = recorded_disconnect

    demo_args = [
        str(demo),
        "--paint-actor", "sac_project/runs/stage3_dagger_v4/best_actor.zip",
        "--dip-actor", "sac_project/runs/stage4_dip_v2/best_actor.zip",
        "--walls", args.wall,
        "--paint-distance-m", "12",
        "--low-fraction", "0.15",
        "--max-refills", "100",
        "--max-actions", "30000",
        "--gui",
        "--out", "sac_project/runs/recorded_demo",
    ]

    try:
        sys.argv = demo_args
        runpy.run_path(str(demo), run_name="__main__")

    except (StopRecording, KeyboardInterrupt):
        print("[video] Recording stopped by user.", flush=True)

    finally:
        p.stepSimulation = original_step
        p.connect = original_connect
        p.disconnect = original_disconnect
        sys.argv = original_argv

        if writer is not None:
            try:
                writer.close()
                print("[video] Recording container closed.", flush=True)

                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                subprocess.run(
                    [
                        ffmpeg, "-y", "-i", str(temporary),
                        "-c", "copy", str(output),
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u sac_project/four_wall_coverage_demo.py \
  --paint-actor sac_project/runs/stage3_dagger_v4/best_actor.zip \
  --dip-actor sac_project/runs/stage4_dip_v2/best_actor.zip \
  --walls north \
  --paint-distance-m 12 \
  --low-fraction 0.15 \
  --max-refills 100 \
  --max-actions 30000 \
  --gui \
  --out sac_project/runs/north_wall_recording                    ],
                    check=True,
                )

                print(f"\n[video] SAVED: {output}", flush=True)
                print(f"[video] Frames: {frames}", flush=True)
                print(f"[video] Duration: {frames / args.fps:.1f} seconds")

            except Exception as exc:
                print(f"[video-error] Finalisation failed: {exc}", flush=True)
                print(f"[video] Intermediate recording: {temporary}", flush=True)
        else:
            print("[video] No video frames captured.", flush=True)

        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    main()
