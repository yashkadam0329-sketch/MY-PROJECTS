# import the necessary packages
from transform import four_point_transform
import numpy as np
import argparse
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-c", "--coords",
	help = "comma seperated list of source points")
args = vars(ap.parse_args())
# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them
image = cv2.imread(args["image"])
if image is None:
    print(" the image did load ")
pts = np.array(eval(args["coords"]), dtype = "float32")
# apply the four point tranform to obtain a "birds eye view" of
# the image
warped = four_point_transform(image, pts)
# show the original and warped image
image1=cv2.resize(image,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_NEAREST)
cv2.imshow("Original", image1)
cv2.waitKey(4000)
cv2.destroyAllWindows()
warped=cv2.resize(warped,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
cv2.imshow("Warped", warped)
key=cv2.waitKey(0)
if key=='q':
    cv2.destroyAllWindows()

#example of how to run on the terminal:  
#transform_example.py --i img20.jpeg --c"[(263, 20),(347, 159),(179, 242),(115, 106)]"