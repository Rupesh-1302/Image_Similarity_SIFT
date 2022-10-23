import cv2
import glob
import numpy as np

class RootSIFT:
	def __init__(self):
		# initialize the SIFT feature extractor
		self.extractor = cv2.xfeatures2d.SIFT_create()
	def compute(self, image, eps=1e-7):
		# compute SIFT descriptors
		(kps, descs) = self.extractor.detectAndCompute(image,None)
		# if there are no keypoints or descriptors, return an empty tuple
		if len(kps) == 0:
			return ([], None)
		# apply the Hellinger kernel by first L1-normalizing and taking the
		# square-root
		descs /= (descs.sum(axis=1, keepdims=True) + eps)
		descs = np.sqrt(descs)
		#descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
		# return a tuple of the keypoints and descriptors
		return (kps, descs)


def BF_FeatureMatcher(des1,des2):
    brute_force = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
    no_of_matches = brute_force.match(des1,des2)
    # for i in no_of_matches:
    #   print(i.distance)
    similar_regions = [i for i in no_of_matches if i.distance*1000 < 300]

    if len(no_of_matches) == 0: 
      return 0

    return len(similar_regions)/len(no_of_matches)


def Similarity_Function(image1,image2):
       # Converting image to grayscale
    gray_img_1= cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray_img_2= cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

    rs = RootSIFT()
    kp1,desc1 = rs.compute(gray_img_1)
    kp2,desc2 = rs.compute(gray_img_2)

    similarity = BF_FeatureMatcher(desc1,desc2)
    return similarity


def Valid_Image(img):
    pan_card_images = [cv2.imread(file) for file in glob.glob("./assets/pan_card/*.png")] 
    similarity_sum = 0
    for pan_card in pan_card_images:
        similarity = Similarity_Function(img,pan_card)
        similarity_sum += similarity
    return similarity_sum/len(pan_card_images)
        


if __name__ == '__main__':

    img_1 = cv2.imread('./assets/temp/img_5.jpeg')
    img_2 = cv2.imread('./assets/temp/g2.jpg')
    img_3 = cv2.imread('./assets/temp/img_4.png')
    img_4 = cv2.imread('./assets/temp/random_image.jpg')
    img_5 = cv2.imread('./assets/temp/addhar_card2.jpeg')
    
 
    print(Valid_Image(img_1) * 100)
