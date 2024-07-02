import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

# cx, cy, w, h = 100, 100, 200, 200

class DragRect():
  def __init__(self, posCenter, size=[200, 200]):
    self.posCenter = posCenter
    self.size = size
    
  def update(self, cursor, rectList):
    cx, cy = self.posCenter[0], self.posCenter[1]
    w, h = self.size
    
    # If the index finger tip is in the rectangle region
    if (cx-w//2 < cursor[0] < cx+w//2) and (cy-h//2 < cursor[1] < cy+h//2):
      for rect in rectList:
        if rect != self:
          other_cx, other_cy = rect.posCenter
          distance = np.hypot(cursor[0] - other_cx, cursor[1] - other_cy)
          if distance < 150:
            return
          
      self.posCenter[0], self.posCenter[1] = cursor[0], cursor[1]

rectList = []
for x in range(5):
  rectList.append(DragRect([x * 250 + 150, 150]))

while True:
  success, img = cap.read()
  if not success:
    break

  img = cv2.flip(img, 1)
  hands, img = detector.findHands(img)

  if hands:
    lmList = hands[0]['lmList']
    if len(lmList) > 12:
      p1 = lmList[8][:2] # coordinates of the index finger tip
      p2 = lmList[12][:2] # coordinates of the middle finger tip
      l, _, _ = detector.findDistance(p1, p2, img)
      print(l)
      if l < 40:
        cursor = lmList[8] # index finger tip landmark
        
        # call the update here
        for rect in rectList:
          rect.update(cursor, rectList)

  # Draw solid rectangle
  # for rect in rectList:
  #   cx, cy = rect.posCenter[0], rect.posCenter[1]
  #   w, h = rect.size
  #   cv2.rectangle(img, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), colorR, cv2.FILLED)
  #   cvzone.cornerRect(img, (cx-w//2, cy-h//2, w, h), 20, rt=0)
  
  # Draw transparent rectangle
  imgNew = np.zeros_like(img, np.uint8)
  for rect in rectList:
    cx, cy = rect.posCenter[0], rect.posCenter[1]
    w, h = rect.size
    cv2.rectangle(imgNew, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), colorR, cv2.FILLED)
    cvzone.cornerRect(img, (cx-w//2, cy-h//2, w, h), 20, rt=0)
  out = img.copy()
  alpha = 0.5
  mask = imgNew.astype(bool)
  # print(mask.shape)
  out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
  
  cv2.imshow("Image", out)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
