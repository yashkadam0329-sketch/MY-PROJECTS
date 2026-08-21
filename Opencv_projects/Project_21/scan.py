from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the image to be scanned")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
ratio=image.shape[0]/500.0
orig=image.copy()
image=imutils.resize(image,height=500)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(5,5),0)
edged=cv2.Canny(gray,75,200)

"""print("STEP 1: Edge Detection")
cv2.imshow("Image",image)
cv2.imshow("Edaged",edged)
cv2.waitkey(0)
cv2.destroyAllWindows()"""

cnts=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:20]

screencnt = None

for c in cnts:
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.04*peri,True)
    print("contour points", len(approx), "area", cv2.contourArea(c))   
    if len(approx)==4:
        screencnt=approx
        break

if screencnt is None:
    print("Could not find the paper.")
    exit()


print("STEP 2: Find contours of paper")
cv2.drawContours(image,[screencnt],-1,(0,255,0),2)
cv2.imshow("image",image)
cv2.waitKey(5000)
cv2.destroyAllWindows()
print(screencnt.reshape(4,2) * ratio)
wraped=four_point_transform(orig,screencnt.reshape(4,2)*ratio)
wraped=cv2.cvtColor(wraped,cv2.COLOR_BGR2GRAY)
T=threshold_local(wraped,11,offset=10,method="gaussian")
wraped=(wraped>T).astype("uint8")*255

print("STEP 3: Apply perspective transform")
#wraped=cv2.medianBlur(wraped,1)

cv2.imshow("Scanned", imutils.resize(wraped, height = 650))
key=cv2.waitKey(0)
if key=="q":
    cv2.destroyAllWindows()