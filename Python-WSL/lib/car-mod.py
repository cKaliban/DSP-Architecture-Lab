import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image

img = cv2.imread('car3.jpg',cv2.IMREAD_COLOR)

img = cv2.resize(img, (620,480) )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
t = 10
k = 5
while(True):
    gray2 = cv2.bilateralFilter(gray, 20, t, t) #Blur to reduce noise
    edged = cv2.Canny(gray2, 40, 200) #Perform Edge detection
    cv2.imshow('edged', edged)
    print(t)
    t+=1
    k = cv2.waitKey(0) & 0xff
    if k==27:
        break
# exit()

gray2 = cv2.bilateralFilter(gray, 20, 37, 37) #Blur to reduce noise
edged = cv2.Canny(gray2, 40, 200) #Perform Edge detection
cv2.imshow('edged', edged)
    
# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
print(cnts)
screenCnt = None

# loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    coeff = 0.0005
    while(True and coeff < 0.1):
        manual_check = ""
        approx = cv2.approxPolyDP(c, coeff * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            print("Coeff == " + str(coeff) + "\n")
            # test_img = img.copy()
            cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
            # test_img = cv2.bitwise_and(img,img,mask=mask)
            cv2.imshow("test", img)
            # cv2.waitKey(0)
            k = cv2.waitKey(0) & 0xff
            if k==27:
                break
            
            
            # if manual_check == "Y":
            #     break
            # else:
            coeff += 0.001
        else:
            coeff += 0.001
        if screenCnt is not None and manual_check == "Y":
            break

if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
    detected = 1

# cv2.imshow('cntr', edged)
# cv2.waitKey(0)

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
if detected == 1:
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)

# Now crop
(x, y) = np.where(mask == 255)
if detected == 1:
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]
    cv2.imshow('Cropped',Cropped)

cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
