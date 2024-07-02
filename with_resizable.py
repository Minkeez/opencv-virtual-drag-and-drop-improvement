import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size
        self.resizing = False

    def update(self, cursor, mode, rectList):
        cx, cy = self.posCenter[0], self.posCenter[1]
        w, h = self.size

        if mode == "resize" and self.resizing:
            self.size = [max(cursor[0] - cx + w // 2, 50), max(cursor[1] - cy + h // 2, 50)]
            return

        if mode == "drag":
            if (cx - w // 2 < cursor[0] < cx + w // 2) and (cy - h // 2 < cursor[1] < cy + h // 2):
                for rect in rectList:
                    if rect != self:
                        other_cx, other_cy = rect.posCenter
                        distance = np.hypot(cursor[0] - other_cx, cursor[1] - other_cy)
                        if distance < 150:
                            return
                self.posCenter[0], self.posCenter[1] = cursor[0], cursor[1]

    def check_resize(self, cursor):
        cx, cy = self.posCenter[0], self.posCenter[1]
        w, h = self.size
        if (cx + w // 2 - 20 < cursor[0] < cx + w // 2 + 20) and (cy + h // 2 - 20 < cursor[1] < cy + h // 2 + 20):
            self.resizing = True
        else:
            self.resizing = False

rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

mode = "drag"  # Initial mode is drag

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        mode = "resize"
    elif key == ord('d'):
        mode = "drag"
    elif key == ord('q'):
        break

    if hands:
        lmList = hands[0]['lmList']
        if len(lmList) > 12:
            cursor = lmList[8]  # index finger tip landmark for dragging
            middle_finger = lmList[12]  # middle finger tip landmark for dragging

            if mode == "drag":
                l, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
                if l < 40:
                    # Update rectangles for dragging
                    for rect in rectList:
                        rect.update(cursor, mode, rectList)
            elif mode == "resize":
                # Check for resizing
                for rect in rectList:
                    rect.check_resize(cursor)

                # Update rectangles for resizing
                for rect in rectList:
                    rect.update(cursor, mode, rectList)

    # Draw transparent rectangles
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter[0], rect.posCenter[1]
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    # Display the current mode
    cv2.putText(out, f"Mode: {mode.capitalize()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", out)

cap.release()
cv2.destroyAllWindows()
