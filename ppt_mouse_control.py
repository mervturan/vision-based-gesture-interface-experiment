import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from HandDetector import HandDetector

import pyautogui
pyautogui.FAILSAFE = True  # slam mouse to top-left to emergency stop

# ----------------------------
# Config
# ----------------------------
CAM_INDEX = 0
CAM_W, CAM_H = 1280, 720

ALPHA = 0.35  # higher = faster (more jitter), lower = smoother (more lag)

# Optional: only control mouse when hand is inside this camera region
# Set to None to disable.
ACTIVE_REGION = None
# Example active region (uncomment to enable):
# ACTIVE_REGION = (200, 100, 880, 520)  # x, y, w, h

# ----------------------------
# Setup
# ----------------------------
screen_w, screen_h = pyautogui.size()
smooth_x, smooth_y = screen_w // 2, screen_h // 2
dragging = False

webcam_capture = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
webcam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
webcam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

hand_detector = HandDetector()

def in_active_region(x, y):
    if ACTIVE_REGION is None:
        return True
    rx, ry, rw, rh = ACTIVE_REGION
    return rx <= x <= rx + rw and ry <= y <= ry + rh

# ----------------------------
# Main loop
# ----------------------------
while True:
    success, img = webcam_capture.read()
    if not success or img is None:
        continue

    img = cv2.flip(img, 1)

    img = hand_detector.process_hands(img)
    landmark_list = hand_detector.get_positions(img)

    cursor_x, cursor_y = None, None
    pinching = False

    if landmark_list:
        indicator_landmark = landmark_list[8]  # index fingertip
        cursor_x, cursor_y = indicator_landmark[1], indicator_landmark[2]
        pinching = hand_detector.is_click(landmark_list)

    # If hand disappears while dragging, release to avoid "stuck mouse down"
    if not landmark_list and dragging:
        pyautogui.mouseUp()
        dragging = False

    if cursor_x is not None and cursor_y is not None and in_active_region(cursor_x, cursor_y):
        # Map camera coords -> screen coords
        target_x = int(cursor_x / CAM_W * screen_w)
        target_y = int(cursor_y / CAM_H * screen_h)

        # Smooth cursor
        smooth_x = int(ALPHA * target_x + (1 - ALPHA) * smooth_x)
        smooth_y = int(ALPHA * target_y + (1 - ALPHA) * smooth_y)

        # Move OS cursor
        pyautogui.moveTo(smooth_x, smooth_y)

        # Pinch = click-and-drag latch
        if pinching and not dragging:
            pyautogui.mouseDown()
            dragging = True
        elif not pinching and dragging:
            pyautogui.mouseUp()
            dragging = False

    # Debug view (camera only)
    if ACTIVE_REGION is not None:
        rx, ry, rw, rh = ACTIVE_REGION
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

    cv2.imshow("PPT Mouse Control (press q to quit)", img)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

# Cleanup
if dragging:
    pyautogui.mouseUp()

webcam_capture.release()
cv2.destroyAllWindows()
