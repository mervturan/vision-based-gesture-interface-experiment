import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from HandDetector import HandDetector

# ----------------------------
# Config / Constants
# ----------------------------
COLOR_GREEN = (0, 255, 0)
RECTANGLE_SIZE = 200

# Draggable PNG
DB_IMG_PATH = "db_icon.png"
DB_W, DB_H = 120, 120
db_x, db_y = 100, 100

# Drop zone
DROP_X, DROP_Y = 800, 200
DROP_W, DROP_H = 200, 200

# State
is_dragging = False

# ----------------------------
# Helpers
# ----------------------------
def overlay_image(bg, img, x, y):
    """Alpha-blend a BGRA/BGR image onto bg at (x,y), clipped to bounds."""
    h, w = img.shape[:2]
    H, W = bg.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x1 >= x2 or y1 >= y2:
        return

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    overlay = img[oy1:oy2, ox1:ox2]

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            bg[y1:y2, x1:x2, c] = alpha * overlay[:, :, c] + (1 - alpha) * bg[y1:y2, x1:x2, c]
    else:
        bg[y1:y2, x1:x2] = overlay


def is_inside_drop_zone(x, y, w, h):
    center_x = x + w // 2
    center_y = y + h // 2
    return (DROP_X < center_x < DROP_X + DROP_W) and (DROP_Y < center_y < DROP_Y + DROP_H)

# ----------------------------
# Setup
# ----------------------------
db_img = cv2.imread(DB_IMG_PATH, cv2.IMREAD_UNCHANGED)
db_img = cv2.resize(db_img, (DB_W, DB_H))

webcam_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# If these cause issues on some laptops, comment them out
webcam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

hand_detector = HandDetector()

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
        # index fingertip
        indicator_landmark = landmark_list[8]
        cursor_x, cursor_y = indicator_landmark[1], indicator_landmark[2]

        pinching = hand_detector.is_click(landmark_list)

    # -------- Drag + Drop logic (single source of truth) --------
    if cursor_x is not None and cursor_y is not None:
        if pinching:
            is_dragging = True
            db_x = cursor_x - DB_W // 2
            db_y = cursor_y - DB_H // 2
        else:
            if is_dragging:
                # released
                if is_inside_drop_zone(db_x, db_y, DB_W, DB_H):
                    db_x = DROP_X + DROP_W // 2 - DB_W // 2
                    db_y = DROP_Y + DROP_H // 2 - DB_H // 2
                is_dragging = False

    # -------- Draw UI --------
    # Draggable image
    overlay_image(img, db_img, db_x, db_y)

    # Drop zone
    cv2.rectangle(img, (DROP_X, DROP_Y), (DROP_X + DROP_W, DROP_Y + DROP_H), COLOR_GREEN, 3)

    cv2.imshow("Drag and Drop OpenCV", img)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

webcam_capture.release()
cv2.destroyAllWindows()
