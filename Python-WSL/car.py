import cv2
import imutils
import numpy as np
import pytesseract
import scipy.ndimage.interpolation as inter
from PIL import Image
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def determine_score(arr, angle):
  data = inter.rotate(arr, angle, reshape=False, order=0)
  histogram = np.sum(data, axis=1)
  score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
  return histogram, score


img = cv2.imread('car2.jpg',cv2.IMREAD_COLOR)

img = cv2.resize(img, (620,480) )
cv2.imwrite("s1.jpg", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale

cv2.imwrite("s2.jpg", gray)
gray = cv2.bilateralFilter(gray, 15, 50, 50) #Blur to reduce noise

cv2.imwrite("s3.jpg", gray)
edged = cv2.Canny(gray, 30, 200) #Perform Edge detection

cv2.imwrite("s4.jpg",edged)
# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

# loop over our contours
for c in cnts:
 # approximate the contour
 peri = cv2.arcLength(c, True)
 approx = cv2.approxPolyDP(c, 0.041 * peri, True)
 
 # if our approximated contour has four points, then
 # we can assume that we have found our screen
 if len(approx) == 4:
  screenCnt = approx
  break

if screenCnt is None:
 detected = 0
 print ("No contour detected")
else:
 detected = 1

if detected == 1:
 cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

# Now crop
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

cv2.imshow('image',img)
cv2.imwrite("s5.jpg", img)
cv2.imshow('Cropped',Cropped)
cv2.imwrite("s6.jpg", Cropped)
thresh = cv2.threshold(Cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imwrite("s7.jpg", thresh)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
# inv = cv2.bitwise_not(thresh)
# blur = cv2.GaussianBlur(inv, (3,3), 0)
scores = []
delta = 1
limit = 5
angles = np.arange(-limit, limit + delta, delta)
for angle in angles:
    histogram, score = determine_score(thresh, angle)
    scores.append(score)

best_angle = angles[scores.index(max(scores))]

(h, w) = thresh.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
rotated = cv2.warpAffine(thresh, M, (w, h), borderMode=cv2.BORDER_REPLICATE)



# thresh = deskew(thresh)
# cv2.imshow('thresh',rotated)
img_new = cv2.imread("car1_mod2.jpg")
result = pytesseract.image_to_string(img_new, lang = 'eng', config="--oem 1 --psm 11")
print(result)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("car1_mod.jpg", thresh)


