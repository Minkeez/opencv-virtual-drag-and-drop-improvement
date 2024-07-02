import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

# Predefined main colors
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 0, 128), (255, 165, 0)
]

class DragRect():
    def __init__(self, posCenter, size=[200, 200], color=(255, 0, 255), label="Rect"):
        self.posCenter = posCenter
        self.size = size
        self.color = color
        self.label = label
        self.resizing = False

    def snap_to_grid(self, value, grid_size=50):
        return round(value / grid_size) * grid_size

    def update(self, cursor, mode, rectList, snap_grid=False):
        cx, cy = self.posCenter[0], self.posCenter[1]
        w, h = self.size

        if mode == "resize" and self.resizing:
            # Calculate the new size
            new_w = max(cursor[0] - (cx - w // 2), 50)
            new_h = max(cursor[1] - (cy - h // 2), 50)
            if snap_grid:
                new_w = self.snap_to_grid(cursor[0], grid_size=50) - (cx - w // 2)
                new_h = self.snap_to_grid(cursor[1], grid_size=50) - (cy - h // 2)
            self.size = [new_w, new_h]
            return

        if mode == "drag":
            if (cx - w // 2 < cursor[0] < cx + w // 2) and (cy - h // 2 < cursor[1] < cy + h // 2):
                for rect in rectList:
                    if rect != self:
                        other_cx, other_cy = rect.posCenter
                        distance = np.hypot(cursor[0] - other_cx, cursor[1] - other_cy)
                        if distance < 150:
                            return
                new_cx, new_cy = cursor[0], cursor[1]
                if snap_grid:
                    new_cx = self.snap_to_grid(new_cx, grid_size=50)
                    new_cy = self.snap_to_grid(new_cy, grid_size=50)
                self.posCenter[0], self.posCenter[1] = new_cx, new_cy

    def check_resize(self, cursor):
        cx, cy = self.posCenter[0], self.posCenter[1]
        w, h = self.size
        if (cx + w // 2 - 20 < cursor[0] < cx + w // 2 + 20) and (cy + h // 2 - 20 < cursor[1] < cy + h // 2 + 20):
            self.resizing = True
        else:
            self.resizing = False

    def change_color(self, new_color):
        self.color = new_color

    def draw(self, img):
        cx, cy = self.posCenter[0], self.posCenter[1]
        w, h = self.size
        cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), self.color, cv2.FILLED)
        cv2.putText(img, self.label, (cx - w // 4, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_grid(img, grid_size=50):
    for x in range(0, img.shape[1], grid_size):
        cv2.line(img, (x, 0), (x, img.shape[0]), (255, 255, 255), 1)
    for y in range(0, img.shape[0], grid_size):
        cv2.line(img, (0, y), (img.shape[1], y), (255, 255, 255), 1)
    return img

def draw_color_palette(img, colors):
    h, w, _ = img.shape
    palette_height = 50
    for i, color in enumerate(colors):
        cv2.rectangle(img, (i * 100, h - palette_height), ((i + 1) * 100, h), color, -1)
    return img

rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

mode = "drag"  # Initial mode is drag
snap_grid = False  # Initial state of snap to grid
show_palette = False
selected_rect = None
color_picker_mode = False

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r') and not color_picker_mode:
        mode = "resize"
    elif key == ord('d') and not color_picker_mode:
        mode = "drag"
    elif key == ord('g') and not color_picker_mode:
        snap_grid = not snap_grid
    elif key == ord('c'):
        color_picker_mode = True  # Enter color picker mode
        show_palette = True  # Show color palette
    elif key == ord('q'):
        break

    if hands:
        lmList = hands[0]['lmList']
        if len(lmList) > 12:
            cursor = lmList[8]  # index finger tip landmark for dragging

            if color_picker_mode:
                if not selected_rect:
                    # Select a rectangle with the index finger
                    for rect in rectList:
                        if (rect.posCenter[0] - rect.size[0] // 2 < cursor[0] < rect.posCenter[0] + rect.size[0] // 2) and \
                           (rect.posCenter[1] - rect.size[1] // 2 < cursor[1] < rect.posCenter[1] + rect.size[1] // 2):
                            selected_rect = rect
                            break
                else:
                    # Display color palette at the bottom of the screen
                    img = draw_color_palette(img, colors)

                    # Check for color selection with index finger
                    h, w, _ = img.shape
                    palette_height = 50
                    if h - palette_height < cursor[1] < h:
                        selected_color_index = cursor[0] // 100
                        if 0 <= selected_color_index < len(colors):
                            selected_rect.change_color(colors[selected_color_index])
                            color_picker_mode = False  # Exit color picker mode
                            show_palette = False  # Hide palette after selecting color
                            selected_rect = None  # Deselect rectangle

            if not color_picker_mode:
                if mode == "drag":
                    l, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
                    if l < 40:
                        # Update rectangles for dragging
                        for rect in rectList:
                            rect.update(cursor, mode, rectList, snap_grid)

                if mode == "resize":
                    # Check for resizing
                    for rect in rectList:
                        rect.check_resize(cursor)

                    # Update rectangles for resizing
                    for rect in rectList:
                        rect.update(cursor, mode, rectList, snap_grid)

    # Draw transparent rectangles
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        rect.draw(imgNew)
        cx, cy = rect.posCenter[0], rect.posCenter[1]
        w, h = rect.size
        cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    # Draw the grid if snap_grid is enabled
    if snap_grid:
        out = draw_grid(out)

    # Display the current mode and snap to grid status
    cv2.putText(out, f"Mode: {mode.capitalize()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(out, f"Snap to Grid: {'On' if snap_grid else 'Off'}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display a message if in color picker mode
    if color_picker_mode:
        cv2.putText(out, "Color Picker Mode", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", out)

cap.release()
cv2.destroyAllWindows()
