#!/usr/bin/env python3
"""
Stillness — video art installation prototype

Only things that have been completely still for a defined period of time
appear in the output. Moving subjects are invisible; the world reveals
itself only through patience and stillness.

Controls
--------
  Q / Escape   Quit
  + / -        Increase / decrease stillness threshold (1 s steps)
  Space        Reset all stillness counters
  C            Cycle to next camera
  S            Save a screenshot
  D            Toggle debug motion-heat overlay
"""

import sys
import cv2
import numpy as np

# ── Default configuration ─────────────────────────────────────────────────────

CAMERA_INDEX        = 0      # Try 0, 1, 2 … or press C to cycle at runtime
STILL_THRESHOLD_SEC = 5.0   # Seconds of stillness before a pixel appears
FADE_IN_SEC         = 1.5   # Seconds to fade in once threshold is reached
PIXEL_DIFF_THRESH   = 18    # Max per-pixel change (0–255) still counted as "still"
BLUR_KERNEL         = 7     # Gaussian blur size (odd number) applied before diff

BACKGROUND = (255, 255, 255)   # White void for moving / unknown pixels
WINDOW_TITLE = "Stillness"


# ── Helpers ───────────────────────────────────────────────────────────────────

def open_camera(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        # Warm-up: grab a couple of frames so exposure settles
        for _ in range(3):
            cap.read()
        return cap
    cap.release()
    return None


def to_gray_blurred(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)


def draw_hud(img: np.ndarray, threshold_sec: float, cam_index: int) -> None:
    h = img.shape[0]
    label = (
        f"Stillness threshold: {threshold_sec:.0f}s   "
        f"cam {cam_index}   "
        f"+/- adjust   Space reset   C camera   S save   Q quit"
    )
    cv2.putText(
        img, label,
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (160, 160, 160), 1, cv2.LINE_AA,
    )


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(start_camera: int = CAMERA_INDEX) -> None:
    cam_index = start_camera

    cap = open_camera(cam_index)
    if cap is None:
        print(f"Could not open camera {cam_index}. Try pressing C to cycle.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame.")
        sys.exit(1)

    h, w = frame.shape[:2]
    still_counter = np.zeros((h, w), dtype=np.float32)
    prev_gray     = to_gray_blurred(frame)
    frozen        = frame.astype(np.float32)   # last committed still image

    threshold_sec  = STILL_THRESHOLD_SEC
    debug_mode     = False
    screenshot_num = 0

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    print(f"Camera {cam_index} opened  |  {w}×{h}  |  {fps:.1f} fps")
    print(f"Initial threshold: {threshold_sec:.0f}s")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        threshold_frames = threshold_sec * fps
        fade_frames      = max(FADE_IN_SEC * fps, 1.0)

        # ── Motion detection ──────────────────────────────────────────────────
        curr_gray = to_gray_blurred(frame)
        diff      = cv2.absdiff(curr_gray, prev_gray).astype(np.float32)
        prev_gray = curr_gray

        moved = diff > PIXEL_DIFF_THRESH
        still_counter[moved]  = 0
        still_counter[~moved] = np.minimum(
            still_counter[~moved] + 1,
            threshold_frames + fade_frames,   # cap to avoid overflow
        )

        # ── Update frozen buffer where pixels are fully still ─────────────────
        fully_still = still_counter >= (threshold_frames + fade_frames)
        frozen[fully_still] = frame.astype(np.float32)[fully_still]

        # ── Fade in: blend live frame toward frozen over the fade window ──────
        alpha  = np.clip(
            (still_counter - threshold_frames) / fade_frames,
            0.0, 1.0,
        )
        alpha3 = alpha[:, :, np.newaxis]   # broadcast over colour channels

        output = (
            frame.astype(np.float32) * alpha3
            + frozen * (1.0 - alpha3)
        ).astype(np.uint8)

        # ── Optional debug overlay (motion heat, top-left corner) ─────────────
        if debug_mode and diff.max() > 0:
            heat  = (diff / diff.max() * 255).astype(np.uint8)
            heat  = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            qh, qw = h // 4, w // 4
            output[:qh, :qw] = cv2.resize(heat, (qw, qh))

        draw_hud(output, threshold_sec, cam_index)
        cv2.imshow(WINDOW_TITLE, output)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):           # Q or Escape → quit
            break

        elif key in (ord('+'), ord('=')):   # + → longer threshold
            threshold_sec = min(threshold_sec + 1, 120)
            print(f"Threshold: {threshold_sec:.0f}s")

        elif key == ord('-'):               # - → shorter threshold
            threshold_sec = max(threshold_sec - 1, 1)
            print(f"Threshold: {threshold_sec:.0f}s")

        elif key == ord(' '):               # Space → reset
            still_counter[:] = 0
            frozen[:] = frame.astype(np.float32)
            print("Counters reset.")

        elif key == ord('s'):               # S → screenshot
            fname = f"stillness_{screenshot_num:04d}.png"
            cv2.imwrite(fname, output)
            screenshot_num += 1
            print(f"Saved {fname}")

        elif key == ord('d'):               # D → debug toggle
            debug_mode = not debug_mode
            print(f"Debug: {'on' if debug_mode else 'off'}")

        elif key == ord('c'):               # C → cycle camera
            cap.release()
            cam_index = (cam_index + 1) % 10
            new_cap = None
            for _ in range(10):
                new_cap = open_camera(cam_index)
                if new_cap is not None:
                    break
                cam_index = (cam_index + 1) % 10

            if new_cap is None:
                print("No other camera found, keeping current.")
                cap = open_camera(start_camera)
            else:
                cap = new_cap
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                ret, frame = cap.read()
                h2, w2 = frame.shape[:2]
                if (h2, w2) != (h, w):
                    h, w = h2, w2
                    still_counter = np.zeros((h, w), dtype=np.float32)
                else:
                    still_counter[:] = 0
                frozen    = frame.astype(np.float32)
                prev_gray = to_gray_blurred(frame)
                print(f"Switched to camera {cam_index}  {w}×{h}  {fps:.1f}fps")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else CAMERA_INDEX
    run(cam)
