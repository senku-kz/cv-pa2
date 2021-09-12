# credit: Juan Carlos Niebles and Ranjay Krishna

import numpy as np


def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding as used in the previous assignment can make
    # derivatives at the image boundary very big.

    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0, pad_width0), (pad_width1, pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    kernel = np.fliplr(np.flipud(kernel))
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(padded[i:i + Hk, j:j + Wk] * kernel)
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out


def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp
    
    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """

    kernel = np.zeros((size, size))

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    k = (size - 1) / 2
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1.0 / (2.0 * np.pi * sigma ** 2)) * \
                           np.exp(-((i - k) ** 2 + (j - k) ** 2) / (2.0 * sigma ** 2))
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return kernel


def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints: 
        - You may use the conv function which is defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    d_x = np.array([[0.5, 0.0, -0.5]])
    out = conv(img, d_x)
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out


def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints: 
        - You may use the conv function which is defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    d_y = np.array([[0.5], [0.0], [-0.5]])
    out = conv(img, d_y)
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out


def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    g_x = partial_x(img)
    g_y = partial_y(img)
    G = np.sqrt(g_x ** 2 + g_y ** 2)
    theta = np.arctan2(g_y, g_x)
    theta = (np.rad2deg(theta) + 180) % 360
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    theta = theta % 360
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            current_angle = theta[i, j]
            if current_angle == 0 or current_angle == 180:
                neighbors = [G[i, j - 1], G[i, j + 1]]
            elif current_angle == 45 or current_angle == 225:
                neighbors = [G[i - 1, j - 1], G[i + 1, j + 1]]
            elif current_angle == 90 or current_angle == 270:
                neighbors = [G[i - 1, j], G[i + 1, j]]
            elif current_angle == 135 or current_angle == 315:
                neighbors = [G[i - 1, j + 1], G[i + 1, j - 1]]
            else:
                raise RuntimeError(
                    "Wrong theta value {}- should be one of the following[0,45,90,135,180,225,270,315]".format(
                        current_angle))
            if G[i, j] >= np.max(neighbors):
                out[i, j] = G[i, j]
            # else:
            #     out[i, j] = 0
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    strong_edges = img >= high
    weak_edges = (img < high) & (img > low)

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y - 1, y, y + 1):
        for j in (x - 1, x, x + 1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors


def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    for i, j in indices:
        neighbors = get_neighbors(i, j, H, W)
        edges[i, j] = 1.0
        viewed = set()
        while neighbors:
            y, x = neighbors.pop(0)
            if (y, x) not in viewed and weak_edges[y, x]:
                edges[y, x] = 1.0
                neighbors.extend(get_neighbors(y, x, H, W))
            viewed.add((y, x))
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return edges
