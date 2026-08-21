import numpy as np
import cv2

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
    rect=np.zeros((4,2), dtype="float32") #makes an array with 4rows and 2 column and all the values will be "0." because of ".zeros" and "dtype=float32"

    # the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
    s=pts.sum(axis=1)#does sum row wise as the axis is 1.If it would have been axis=0 then column wise so the sum of pts is stored in s.
    rect[0]=pts[np.argmin(s)]#the minimum value of s is searched and then rect[0]=pts[minimum value's index],so hence the minimum value is stored in 0th index of array rect
    rect[2]=pts[np.argmax(s)]#the maximum value of s is searched and then rect[2]=pts[maximum value's index],so hence the maximum value is stored in 2th index of array rect


    # now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
    diff=np.diff(pts,axis=1)#does diff row wise as the axis=1.If it would have been axis=0 then column wise so the diff of pts is stored in diff.
    rect[1]=pts[np.argmin(diff)]#the minimum value of diff is searched and then rect[1]=pts[minimum value's index],so hence the minimum diff value is stored in 1th index of array rect
    rect[3]=pts[np.argmax(diff)]#the maximum value of diff is searched and then rect[3]=pts[maximum value's index],so hence the maximum diff value is stored in 3th index of array rect


    # return the ordered coordinates
    return rect

def four_point_transform(image,pts):
    # obtain a consistent order of the points and unpack them
	# individually
    rect=order_points(pts)
    (tl,tr,br,bl)=rect 

    # compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
    widthA=np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2)) #formula used d=(((x2-x1)**2)+(y2-y1)**2))**0.5 
    widthB=np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2)) #formula used d=(((x2-x1)**2)+(y2-y1)**2))**0.5
    maxWidth=max(int(widthA),int(widthB)) #maximum of from both is taken

    # compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)) #formula used d=(((x2-x1)**2)+(y2-y1)**2))**0.5
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)) #formula used d=(((x2-x1)**2)+(y2-y1)**2))**0.5
    maxHeight = max(int(heightA), int(heightB)) #maximum of from both is taken

    # now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped