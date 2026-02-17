import cv2
from HandDetector import HandDetector

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


COLOR_PINK = (255, 0, 255)
COLOR_GREEN = (0, 255, 0)
RECTANGLE_SIZE = 200

rectangle_color = COLOR_PINK
rectangle_x, rectangle_y = 100, 100
db_img = cv2.imread("db_icon.png", cv2.IMREAD_UNCHANGED) # Load the database icon image

# Set the initial position size of the dataabse image
db_w, db_h = 120, 120
db_x, db_y = 100, 100
db_img = cv2.resize(db_img, (db_w, db_h))


webcam_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
webcam_capture.set(3, 1280)
webcam_capture.set(4, 720)

hand_detector = HandDetector()


def is_indicator_in_rectangle():
    return rectangle_x < indicator_landmark[1] < rectangle_x + RECTANGLE_SIZE and rectangle_y < indicator_landmark[2] < rectangle_y + RECTANGLE_SIZE


# Function to overlay an image with transparency onto the background
def overlay_image(bg, img, x, y):
    h, w = img.shape[:2]
    H, W = bg.shape[:2]

    # Clip overlay region to background bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)

    # If completely outside, do nothing
    if x1 >= x2 or y1 >= y2:
        return

    # Corresponding region in the overlay image
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    overlay = img[oy1:oy2, ox1:ox2]

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            bg[y1:y2, x1:x2, c] = alpha * overlay[:, :, c] + (1 - alpha) * bg[y1:y2, x1:x2, c]
    else:
        bg[y1:y2, x1:x2] = overlay


while True:
    success, img = webcam_capture.read()

    # Avoid crashing frame if webcam fails to capture
    if not success or img is None:
     continue

    img = cv2.flip(img, 1)

    img = hand_detector.process_hands(img)
    landmark_list = hand_detector.get_positions(img)

    # Draw Image Each Frame
    overlay_image(img, db_img, db_x, db_y)

    # Updated if block to work weith png
    if landmark_list:
        indicator_landmark = landmark_list[8]
        cursor_x, cursor_y = indicator_landmark[1], indicator_landmark[2]

        # Optional: keep rectangle logic as a target zone, not the draggable object
        if is_indicator_in_rectangle():
            rectangle_color = COLOR_GREEN
        else:
            rectangle_color = COLOR_PINK

        # Only move DB icon when pinching (click gesture)
        if hand_detector.is_click(landmark_list):
            db_x = cursor_x - db_w // 2
            db_y = cursor_y - db_h // 2

        
    cv2.rectangle(img, (rectangle_x, rectangle_y), (rectangle_x + RECTANGLE_SIZE, rectangle_y + RECTANGLE_SIZE), rectangle_color, cv2.FILLED)

    cv2.imshow("Drag and Drop OpenCV", img)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
    
webcam_capture.release()
cv2.destroyAllWindows()

