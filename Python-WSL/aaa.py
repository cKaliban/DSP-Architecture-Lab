
import pytesseract
import cv2
img_new = cv2.imread("car2_mod2.jpg")

for i in range(3,12):
    # conf = "--oem 1 --psm " + str(i)
    result = pytesseract.image_to_string(img_new, lang = 'eng', config="--oem 1 --psm " + str(i))
    print(result)

cv2.imshow("aa", img_new)
cv2.waitKey(0)