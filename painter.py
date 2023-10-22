import cv2
import os
import numpy as np
import handtracker


folderPath = "header/tools"
imglist = os.listdir(folderPath)
overlayimg = []
imglist.sort()
brushThickness = 15
headerMain = cv2.imread("header/headerMain.png")
for imPath in imglist:
    if imPath.lower().endswith(('.png')):
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayimg.append(image)

color = (218, 56, 50)
canvas = np.zeros((720, 1280, 3), np.uint8)
header = overlayimg[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
xp, yp = 0, 0
detector = handtracker.handDetector(detectionCon=0.85)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)

    if (lmlist != []):

        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        fingers = detector.fingersUp()
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25),
                          color, cv2.FILLED)
            if x1 < 131:
                if 40 < y1 < 150:
                    header = overlayimg[0]
                    color = (29, 0, 218)
                elif 150 < y1 < 270:
                    header = overlayimg[1]
                    color = (12, 242, 253)
                elif 270 < y1 < 390:
                    header = overlayimg[2]
                    color = (64, 164, 23)
                elif 390 < y1 < 510:
                    header = overlayimg[3]
                    color = (234, 156, 21)
                elif 510 < y1 < 630:
                    header = overlayimg[4]
                    color = (255, 255, 255)
                elif 630 < y1 < 720:
                    header = overlayimg[5]
                    color = (0, 0, 0)

        if fingers[1] and fingers[2] == False:

            cv2.circle(img, (x1, y1), 15, color, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if color == (0, 0, 0):
                brushThickness = 45
            cv2.line(img, (xp, yp), (x1, y1), color, brushThickness)
            cv2.line(canvas, (xp, yp), (x1, y1), color, brushThickness)
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    img[0:720, 0:131] = header
    img[0:40, 0:1280] = headerMain
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", canvas)
    cv2.waitKey(1)
