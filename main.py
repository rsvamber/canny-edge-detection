import numpy as np
from skimage.color import rgb2gray
from skimage import io, transform, feature, filters
from scipy import ndimage
import matplotlib.pyplot as plot
import os

""" SETTINGS """
save_image = True  # Optional saving of the processed image
image_dir = 'images/'  # Directory of the images
image_name = ''  # Name of the image
image_ext = ''  # Image extension
output_image_dir = image_dir + 'processed/' + image_name  # Directory for the processed images
otsu = False  # Otsu threshold
canny_skimage = True  # if false, use canny function
hough_threshold = 12  # threshold for detected lines
canvas = True  # if True, saves extracted edges on a blank canvas
vp_detection = True  # if True, visualizes vanishing points in the image
""" END OF SETTINGS """

""" FUNCTIONS """


def sobel_filter(gaussian_image):
    """Sobel filter for Edge detection
    Parameters:
    ---------------------------
    gaussian_image: ndarray
        Image after being processed by Gaussian blur

    Returns:
    ---------------------------
    G: ndarray
        Gradient magnitude
    theta: ndarray
        Angle of orientation
    Reference:
    ---------------------------
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm
    """

    """ 3x3 sobel filter"""
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    """ Convolution of the image with filters both x and y axis"""
    img_X = ndimage.convolve(gaussian_image, Gx)
    img_Y = ndimage.convolve(gaussian_image, Gy)

    G = np.sqrt(img_X * img_X + img_Y * img_Y)
    theta = np.degrees(np.arctan2(img_Y, img_X))

    """ Normalization"""

    G = G / G.max() * 255

    """Return gradient magnitude and theta"""
    return G, theta


def non_max_suppression(filtered_image, theta):
    """Non-max suppression
    Parameters:
    ---------------------------
    filtered_image: ndarray
        Image after being processed by sobel filter blur
    theta: ndarray
        From sobel filter
    Returns:
    ---------------------------
    Z: ndarray
        Pixel intensity

    Reference:
    ---------------------------
    Credit to Sofiane Sahir https://github.com/FienSoP
    """

    """ Zero matrix with the size of the filtered image"""
    X, Y = filtered_image.shape
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
                    q = filtered_image[i, j + 1]
                    r = filtered_image[i, j - 1]
                # 45 degrees
                elif 22.5 <= theta[i, j] < 67.5:
                    q = filtered_image[i + 1, j - 1]
                    r = filtered_image[i - 1, j + 1]
                # 90 degrees
                elif 67.5 <= theta[i, j] < 112.5:
                    q = filtered_image[i + 1, j]
                    r = filtered_image[i - 1, j]
                # 135 degrees
                elif 112.5 <= theta[i, j] < 157.5:
                    q = filtered_image[i - 1, j - 1]
                    r = filtered_image[i + 1, j + 1]

                """ Changing pixel intensity """
                if (filtered_image[i, j] >= q) and (filtered_image[i, j] >= r):
                    Z[i, j] = filtered_image[i, j]
                else:
                    Z[i, j] = 0

            except IndexError:
                pass

    return Z


def hough(edges):
    """Hough probabilistic line transformation
    Parameters:
    ---------------------------
    edges: ndarray
        Extracted edges using canny

    Returns:
    ---------------------------
    locations: ndarray of shape (n_h_edges, 2)
            Locations of each of the h_edges.
    directions: ndarray of shape (n_h_edges, 2)
            Direction of the edge (tangent) at each of the edgelet.
    strengths: ndarray of shape (n_h_edges, 2)
            Length of the line segments detected for the edgelet.

    Reference:
    ---------------------------
    Credit to chsasank https://github.com/chsasank/Image-Rectification
    """

    """ Probablistic Hough transform """
    lines = transform.probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)
    locations = []
    directions = []
    strengths = []
    """Looping through lines to visualize them"""
    for p0, p1 in lines:
        p0, p1 = np.array(p0), np.array(p1)
        locations.append((p0 + p1) / 2)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    """ Converting to numpy array """
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)
    """ Normalization """
    directions = np.array(directions) / \
        np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return locations, directions, strengths


def h_lines(h_edges):
    """Compute lines in homogenous system for hough edges.
    Parameters:
    ---------------------------
    h_edges: tuple of ndarrays
        (locations, directions, strengths) as computed by `h_edges`.
    Returns:
    ---------------------------
    lines: ndarray of shape (n_h_edges, 3)
        Lines at each of h_edges locations in homogenous system.
    Reference:
    ---------------------------
    All of the credit goes to chsasank https://github.com/chsasank/Image-Rectification
    """
    locations, directions, strengths = h_edges
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines


def detect_vp(h_edges, num_ransac_iter=2000, threshold_inlier=5):
    """Estimate vanishing point using Ransac.
        Parameters:
        ---------------------------
        h_edges: tuple of ndarrays
            (locations, directions, strengths) as computed by `compute_h_edges`.
        num_ransac_iter: int
            Number of iterations to run ransac.
        threshold_inlier: float
            threshold to be used for computing inliers in degrees.
        Returns:
        ---------------------------
        best_model: ndarry of shape (3,)
            Best model for vanishing point estimated.
        Reference:
        ---------------------------
        All of the credit goes to chsasank https://github.com/chsasank/Image-Rectification
        """

    """ Unpacking of the values from h_edges """
    locations, directions, strengths = h_edges
    lines = h_lines(h_edges)
    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_model = None
    best_votes = np.zeros(num_pts)
    """ Starting the iteration for the best model computation """
    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)

        """Cleaning up degenerate candidates"""
        if np.sum(current_model ** 2) < 1 or current_model[2] == 0:
            continue

        current_votes = compute_votes(
            h_edges, current_model, threshold_inlier)
        if current_votes.sum() > best_votes.sum():
            best_model = current_model
            best_votes = current_votes

    return best_model


def compute_votes(h_edges, model, threshold_inlier=5):
    """Compute votes for each of the edges extracted by 'hough' against a given vanishing point.
    Votes for h_edges which lie inside threshold are same as their strengths,
    otherwise zero.
    Parameters:
    ---------------------------
    h_edges: tuple of ndarrays
        (locations, directions, strengths) as computed by 'hough'.
    model: ndarray of shape (3,)
        Vanishing point model in homogeneous coordinate system.
    threshold_inlier: float
        Threshold to be used for computing inliers in degrees. Angle between
        h_edge direction and line connecting the Vanishing point model and
        h_edge location is used to threshold.
    Returns:
    ---------------------------
    votes: ndarry of shape (n_h_edges,)
        Votes towards vanishing point model for each of the edges.
    Reference:
    ---------------------------
    All of the credit goes to chsasank https://github.com/chsasank/Image-Rectification """

    vp = model[:2] / model[2]

    locations, directions, strengths = h_edges

    est_directions = locations - vp
    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
        np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    theta = np.arccos(np.abs(cosine_theta))

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths


def vis_vp(model, on_blank=False, save_num = '', save_vp=True):
    """Visualizing vanishing points computed by 'detect_vp'
    Parameters:
    ---------------------------
    image: ndarray
        Original image
    model: ndarray
        Extracted best model from detect_vp function
    on_blank: boolean
        Whether the model should be visualized on a blank canvas or not
    save_vp: boolean
        Optional saving of the model

    Reference:
    ---------------------------
    Credit to chsasank https://github.com/chsasank/Image-Rectification
    """

    edgelets = h
    locations, directions, strengths = edgelets
    inliers = compute_votes(edgelets, model, 10) > 0

    edgelets = (locations[inliers], directions[inliers], strengths[inliers])
    locations, directions, strengths = edgelets
    if on_blank:
        im_map = image.copy()
        im_map.fill(255)
        plot.imshow(im_map)
    else:
        plot.imshow(image)
    vp = model / model[2]
    plot.plot(vp[0], vp[1], 'bo')

    for i in range(locations.shape[0]):
        xax = [locations[i, 0], vp[0]]
        yax = [locations[i, 1], vp[1]]
        plot.plot(xax, yax, 'b-.')
        plot.axis('off')

    fig = plot.gcf()
    if save_vp:
        fig.savefig(output_image_dir + '_' + save_num + 'vp.png', dpi=200)
    plot.title('Vanishing point '+save_num)
    plot.show()


def save_hough(lines):
    """Saving the extracted lines using Hough
    Parameters:
    ---------------------------
    image: ndarray
        Original image
    lines: ndarray
        Extracted lines

    Reference:
    ---------------------------
    Credit to chsasank https://github.com/chsasank/Image-Rectification
    """
    locations, directions, strengths = lines
    plot.figure(figsize=(10, 10))
    if canvas:
        im_map = image.copy()
        im_map.fill(255)
        plot.imshow(im_map)
    else:
        plot.imshow(image)
    for i in range(locations.shape[0]):
        xax = [locations[i, 0] - directions[i, 0] * strengths[i] / 2,
               locations[i, 0] + directions[i, 0] * strengths[i] / 2]
        yax = [locations[i, 1] - directions[i, 1] * strengths[i] / 2,
               locations[i, 1] + directions[i, 1] * strengths[i] / 2]
        plot.axis('off')
        plot.plot(xax, yax, '-r')
    if save_image:
        dir_name = 'images/processed'

        try:
            os.mkdir(dir_name)
        except FileExistsError:
            pass
        plot.savefig(output_image_dir + '_processed.png', dpi=200)
    plot.title("Hough lines")
    plot.show()

def canny():
    """Canny edge detection in 5 steps: rgb2gray, gaussian filter, Sobel filter,
                                        non-max suppression, Double threshold + hysteresis

    Returns:
    ---------------------------
    edges: ndarray
        Extracted edges from the image
    """

    """Step 1: Converting to grayscale"""
    if otsu:
        # TODO improve Otsu threshold, currently better without it
        gray_image = rgb2gray(image)
        thresh = filters.threshold_otsu(gray_image)
        gray_image = gray_image > thresh
    else:
        gray_image = rgb2gray(image)

    """Plots, uncomment if needed"""
    # plot.imshow(gray_image, cmap = plot.cm.gray)
    # plot.show()

    """Step 2: Reducing the salt-and-pepper noise"""
    gaussian_image = filters.gaussian(gray_image, sigma=0.2)

    """Plots, uncomment if needed"""
    # plot.imshow(gaussian_image, cmap = plot.cm.gray)
    # plot.title("Gaussian blur")
    # plot.show()

    """ Step 3 and 4: Sobel filter + non-max suppression"""
    filtered_image, theta = sobel_filter(gaussian_image)
    # plot.imshow(filtered_image, cmap=plot.cm.gray)
    # plot.title("sobel filter")
    # plot.show()
    suppression_image = non_max_suppression(filtered_image, theta)

    """Plots, uncomment if needed"""
    # plot.imshow(suppression_image, cmap=plot.cm.gray)
    # plot.title("Suppression")
    # plot.show()

    """ Step 5: Double threshold + hysteresis """

    sigma = 0.33
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    if canny_skimage:
        edges = feature.canny(gray_image, 3)
    else:
        edges = ((suppression_image > upper).astype(int)) \
                + filters.apply_hysteresis_threshold(suppression_image, lower, upper)

    """ Plot of Canny edges """
    plot.imshow(edges, cmap = plot.cm.gray)
    plot.title("Canny edges")
    plot.show()
    return edges


def remove_inliers(model, h):
    """Remove all inlier edglets of a given model.
    Parameters:
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    h: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.
    Returns:
    -------
    edgelets_new: tuple of ndarrays
        All Edgelets except those which are inliers to model.

    Reference:
    ---------------------------
    Credit to chsasank https://github.com/chsasank/Image-Rectification
    """
    inliers = compute_votes(h, model, 10) > 0
    locations, directions, strengths = h
    locations = locations[~inliers]
    directions = directions[~inliers]
    strengths = strengths[~inliers]
    edgelets = (locations, directions, strengths)
    return edgelets


""" END OF FUNCTIONS """

""" MAIN """
if __name__ == '__main__':
    try:
        image = io.imread(image_dir + image_name + image_ext)
    except IOError:
        print("Incorrect file name or extension")
        exit()
    c = canny()
    h = hough(c)

    if vp_detection:
        """first vanishing point"""
        vp1 = detect_vp(h)
        vis_vp(vp1, save_num = '1')

        """second vanishing point"""
        h2 = remove_inliers(vp1, h)
        vp2 = detect_vp(h2, 2000)
        vis_vp(vp2, save_num = '2')


        """third vanishing point"""
        h3 = remove_inliers(vp2, h2)
        vp3 = detect_vp(h3, 2000)
        vis_vp(vp3, save_num='3')

    """ Saving extracted Hough lines """
    save_hough(h)

""" END OF MAIN """
