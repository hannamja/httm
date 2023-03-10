import math
import cv2
import numpy as np
from skimage.feature import local_binary_pattern  # # pip install scikit-image
import glob
KERNEL_WIDTH = 7
KERNEL_HEIGHT = 7
SIGMA_X = 3
SIGMA_Y = 3


def main():
    cv_img = []
    for img in glob.glob("C:/Users/ACER/PycharmProjects/pythonProject1/data/data/train/benign/*.png"):
        n = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        cv_img.append(n)

    i = 0
    # LBP
    for img in cv_img:
        # Gaussian blur + LBP
        blur_img = cv2.GaussianBlur(img, ksize=(KERNEL_WIDTH, KERNEL_HEIGHT), sigmaX=SIGMA_X, sigmaY=SIGMA_Y)
        blur_out = local_binary_pattern(image=blur_img, P=8, R=1, method='default')
        path = 'C:/Users/ACER/PycharmProjects/sae/dataset/train/benign/blur'+str(i)+".png"
        print(path)
        cv2.imwrite(path, blur_out)
        i+=1
        print("Saved image @ blur.jpg")
        print("Saved image @ blur_lbp.jpg")


if __name__ == "__main__":
    main()
    print('---------')
    print('* Follow me @ ' + "\x1b[1;%dm" % (34) + ' https://www.facebook.com/kyznano/' + "\x1b[0m")
    print('* Minh fanpage @ ' + "\x1b[1;%dm" % (34) + ' https://www.facebook.com/minhng.info/' + "\x1b[0m")
    print('* Join GVGroup @ ' + "\x1b[1;%dm" % (34) + 'https://www.facebook.com/groups/ip.gvgroup/' + "\x1b[0m")
    print('* Thank you ^^~')