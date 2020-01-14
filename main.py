import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian, sobel
from skimage import io
from scipy import ndimage
import matplotlib.pyplot as plot
from skimage.filters import apply_hysteresis_threshold
from skimage.filters import threshold_otsu
import os

""" SETTINGS """
save_image = True #Optional saving of the processed image
image_dir = 'images/' #Directory of the images
image_name = 'rgbhouse1' #Name of the image
image_ext = '.jpg' #Image extension
output_image_dir = image_dir +'processed/' +image_name #Directory for the processed images
highThresholdRatio = 0.09
lowThresholdRatio = 0.05
otsu = False #Otsu threshold
skimage_sobel = False #if false, use sobel_filter
""" END OF SETTINGS """

""" FUNCTIONS """

"""
Sobel filter for Edge detection
Source for calculations: https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm
"""
def sobel_filter(image):
    """ 3x3 sobel filter"""
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    """ Convolution of the image with filters both x and y axis"""
    img_X = ndimage.convolve(image, Gx)
    img_Y = ndimage.convolve(image, Gy)

    """ Gradient magnitude"""
    G = np.sqrt(img_X*img_X + img_Y*img_Y)

    """ The angle of orientation of the edge (relative to the pixel grid)"""
    theta = np.degrees(np.arctan2(img_Y, img_X))

    """ Normalization"""

    G = G / G.max() * 255

    """Return gradient magnitute and theta"""
    return (G,theta)

"""
Non-max supression
Credit to Sofiane Sahir https://github.com/FienSoP
"""
def non_max_suppression(img, theta):

    """ Zero matrix with the size of img"""
    X, Y = img.shape
    Z = np.zeros((X, Y))

    theta[theta < 0] += 180

    """ Identifying the angle value for non_max suppression"""
    for i in range(1, X - 1):
        for j in range(1, Y - 1):
            try:
                """ Pixel intesity for comparing"""
                q = 255
                r = 255

                 # 0 degrees
                if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # 45 degrees
                elif (22.5 <= theta[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # 90 degrees
                elif (67.5 <= theta[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # 135 degrees
                elif (112.5 <= theta[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                """ Changing pixel intensity """
                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z
""" END OF FUNCTIONS """

"""Step 1: Converting to grayscale"""

if(otsu):
    #TODO improve Otsu threshold for foreground / background extraction, currently better without it
    image = rgb2gray(io.imread(image_dir + image_name+image_ext))
    thresh = threshold_otsu(image)
    grayImage = image > thresh
else:
    grayImage = rgb2gray(io.imread(image_dir + image_name+image_ext))

"""Plots, uncomment if needed"""
#plot.imshow(grayImage, cmap = plot.cm.gray)
#plot.show()

"""Step 2: Reducing the salt-and-pepper noise"""
img = gaussian(grayImage, sigma=0.2)

"""Plots, uncomment if needed"""
#plot.imshow(img, cmap = plot.cm.gray)
#plot.show()
#Step 3: Sobel filter

if(skimage_sobel):
    """ using skimage """
    image_skimage = sobel(img)
else:
    """ using my function """
    filteredImage,theta = sobel_filter(img)

""" Comparison of the two, uncomment if needed """
#plot.subplot(121)
#plot.imshow(image_skimage, cmap = plot.cm.gray)
#plot.subplot(122)
#plot.imshow(filteredImage, cmap = plot.cm.gray)
#plot.show()

""" Step 4: Non-max suppresion """
suppressionImage = non_max_suppression(filteredImage,theta)
"""Plots, uncomment if needed"""
#plot.imshow(supressionImage, cmap = plot.cm.gray)
#plot.show()

""" Step 5: Double threshold + hysteresis """

#TODO Further smooth out remaining weak edges

high = suppressionImage.max()*highThresholdRatio
low = high*lowThresholdRatio
final = ((suppressionImage > high).astype(int)) + apply_hysteresis_threshold(suppressionImage, low, high)
""" Plot of the final result """
plot.imshow(final, cmap = plot.cm.gray)
plot.title("Final picture")
plot.show()

if(save_image):
    dirName = 'images/processed'

    try:
        os.mkdir(dirName)
    except FileExistsError:
        pass
    #TODO Fix UserWarning
    io.imsave(output_image_dir+ '_processed'+image_ext, final)
