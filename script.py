import cv2
import imutils  # resize
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# reading image
image = cv2.imread('testcase4.jpg')
# resizing image to standardize size (500)
image = imutils.resize(image, width=500)
# showing original image first
cv2.imshow("Original", image)
cv2.waitKey(0)  # wont execute till anything is pressed

# grey-scale conversion
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grey-Scale", grey)
cv2.waitKey(0)

#de-noising and smoothening
grey = cv2.bilateralFilter(grey, 11, 17, 17)
cv2.imshow("De-noised and Smoothened", grey)
cv2.waitKey(0)

# edges
edged = cv2.Canny(grey, 150, 200)
cv2.imshow("Canny", edged)
cv2.waitKey(0)

# finding contours
cnts, new = cv2.findContours(
    edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# new = hierarchy relationship
# RETR_LIST = retrieves all contours, but doesn't create parent-child relationship
# CHAIN_APPROX_SIMPLE = removes all redundant points and compress contours by saving memory

image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Canny after Contouring", image1)
cv2.waitKey(0)

# we dont want all contours, but cant always directly locate the licence plate
# so we select 30 maximum areas, and sort them in descending order
# default order is min to max, so we will have to reverse it

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCount = None

image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
cv2.imshow("30 Max Contours", image2)
cv2.waitKey(0)

# finding best possible contour of expected license plate
count = 0
name = 1

for i in cnts:
    perimeter = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
    if(len(approx) == 4):
        NumberPlateCount = approx
        x, y, w, h = cv2.boundingRect(i)
        crp_img = image[y:y+h, x:x+w]
        cv2.imwrite(str(name)+'.png', crp_img)
        name += 1

        break

# drawing contours on our identified possible number plate
cv2.drawContours(image, [NumberPlateCount], -1, (0, 255, 0), 3)
cv2.imshow("Final Image", image)
cv2.waitKey(0)

# cropping only to number plate
crop_img_loc = '1.png'
cv2.imshow("Cropped", cv2.imread(crop_img_loc))

# converting image to text using Pytesseract module
text = pytesseract.image_to_string(crop_img_loc, lang='eng')
print("M Number is : ", text)
cv2.waitKey(0)
