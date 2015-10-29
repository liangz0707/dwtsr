# coding:utf-8
__author__ = 'liangz14'
import math
import numpy as np
from skimage import color
from scipy import concatenate, dot, zeros_like, array, newaxis, mod
from skimage import data
import skimage.io as io
import matplotlib.pyplot as plt
from scipy import misc, sum, zeros_like

def classic2polyphase(c, d):
    """
    Transforms the coefficients of the low- and high-pass filter to a polyphase form
    :param c: low-pass coefficients
    :param d: high-pass coefficients
    :return: polyphase format containing c and d
    """
    return concatenate((array(c)[newaxis, :], array(d)[newaxis, :]), axis=0)

def daubechies(order, polyphase=True):
    """
    returns the lowpass and highpass filter coefficients for a Daubechies wavelet
    :param order: the order of the wavelet function
    :param polyphase: whether the output should be return in polyphase form
    :return: the daubieches-<order> coefficients
    """
    k=0.5
    if order is 0:
        c = [k, k]
        d = [-k ,k]
    elif order is 1:
        c = [0.7071067812, 0.7071067812]
        d = [-0.7071067812, 0.7071067812]
    elif order is 2:
        c = [-0.1294095226, 0.2241438680, 0.8365163037, 0.4829629131]
        d = [-0.4829629131, 0.8365163037, -0.2241438680, -0.1294095226]
    elif order is 3:
        c = [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569]
        d = [-0.3326705529509569, 0.8068915093133388, -0.4598775021193313, -0.13501102001039084, 0.08544127388224149, 0.035226291882100656]
    else:
        raise ValueError('order is not supported')

    if polyphase:
        return classic2polyphase(c, d)

    return c, d

def dwt(s, poly, l=1):
    """
    Computes the discrete wavelet transform for a 1D signal
    :param s: the signal to be processed
    :param poly: polyphase filter matrix cointing the lowpass and highpass coefficients
    :param l: amount of transforms to be applied
    :return: the transformed signal
    """
    assert len(s) % (2**l) == 0, 'signal length ({}) does not allow for a {}-level decomposition'.format(len(s), l)

    detail = []
    approximation = array(s)
    for level in range(l):
        s = approximation.reshape((approximation.shape[0]/2, 2)).transpose()

        decomposition = zeros_like(s, dtype=float)
        for z in range(poly.shape[1]/2):
            decomposition += dot(poly[:, 2*z:2*z+2], concatenate((s[:, z:], s[:, :z]), axis=1))

        approximation = decomposition[0, :]
        detail.append(decomposition[1, :])

    return approximation, detail


def idwt(a, d, poly, l=1):
    """
    Computes the inverse discrete wavelet transform for a 1D signal
    :param a: the approximation coefficients at the deepest level
    :param d: a list of detail coefficients for each level
    :param poly: polyphase filter matrix cointing the lowpass and highpass coefficients
    :param l: amount of transforms to be applied
    :return: the transformed signal
    """
    assert len(d) == l, 'insufficient detail coefficients provided for reconstruction depth {}'.format(l)

    if len(a.shape) == 1:
        a = a[newaxis, :]

    for level in reversed(range(l)):
        decomposition = concatenate((a, d[level][newaxis, :]), axis=0)

        reconstruction = zeros_like(decomposition, dtype=float)
        for z in range(poly.shape[1]/2):
            reconstruction += dot(poly[:, 2*z:2*z+2].transpose(), concatenate(
                (decomposition[:, decomposition.shape[1]-z:], decomposition[:, :decomposition.shape[1]-z]), axis=1))

        a = reconstruction.transpose().reshape(1, 2*a.shape[1])

    return a

def dwt_2d(image, poly, l=1):
    """
    Computes the discrete wavelet transform for a 2D input image
    :param image: input image to be processed
    :param poly: polyphase filter matrix cointing the lowpass and highpass coefficients
    :param l: amount of transforms to be applied
    :return: the transformed image
    """
    assert max(mod(image.shape, 2**l)) == 0, 'image dimension ({}) does not allow for a {}-level decomposition'.format(image.shape, l)

    image_ = image.copy()
    for level in range(l):
        sub_image = image_[:(image.shape[0]/(2**level)), :(image.shape[1]/(2**level))]

        for row in range(sub_image.shape[0]):
            s = sub_image[row, :]
            a, d = dwt(s, poly)

            sub_image[row, :] = concatenate((a[newaxis, :], d[0][newaxis, :]), axis=1)

        for col in range(sub_image.shape[1]):
            s = sub_image[:, col]
            a, d = dwt(s, poly)

            sub_image[:, col] = concatenate((a, d[0]), axis=0)

    return image_

def idwt_2d(image, poly, l=1):
    """
    Computes the inverse discrete wavelet transform for a 2D input image
    :param image: input image to be processed
    :param poly: polyphase filter matrix containing the lowpass and highpass coefficients
    :param l: amount of transforms to be applied
    :return: the transformed image
    """
    assert max(mod(image.shape, 2**l)) == 0, 'image dimension ({}) does not allow for a {}-level reconstruction'.format(image.shape, l)

    image_ = image.copy()
    for level in reversed(range(l)):

        sub_image = image_[:(image.shape[0]/(2**level)), :(image.shape[1]/(2**level))]

        for col in range(sub_image.shape[1]):
            a = sub_image[:sub_image.shape[0]/2, col]
            d = sub_image[sub_image.shape[0]/2:, col]

            sub_image[:, col] = idwt(a, [d], poly)

        for row in range(sub_image.shape[0]):
            a = sub_image[row, :sub_image.shape[1]/2]
            d = sub_image[row, sub_image.shape[1]/2:]

            sub_image[row, :] = idwt(a, [d], poly)

    return image_


def test_2():
    # obtain a grey-scale image of Lena
    image = misc.lena().astype(dtype=float)
    src_rgb_img = io.imread('Child_input.png')
    src_rgb_img = io.imread('aero.png')

    #src_Lab_img = color.rgb2lab(src_rgb_img)
    #image = src_Lab_img[:,:,0]
    #image =src_rgb_img
    # specific decomposition depth
    l = 1

    poly = daubechies(order=1)
    print poly
    # apply an l-level wavelet decomposition
    decomposition = dwt_2d(image, poly, l)

    # erase the detail
    compression = zeros_like(decomposition)
    compression[:(compression.shape[0]/(2**l)), :(compression.shape[1]/(2**l))] = decomposition[:(compression.shape[0]/(2**l)), :(compression.shape[1]/(2**l))].copy()

    # reconstruct the image without detail
    image_hat = idwt_2d(decomposition, poly, l)

    # plotting
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('mean squared reconstruction difference = {}'.format(sum(sum((image_hat - image)**2))/(image.shape[0] * image.shape[1])))

    ax[0, 0].imshow(image, cmap=plt.cm.gray)
    ax[0, 0].set_title('original image')

    ax[0, 1].imshow(np.abs(decomposition), cmap=plt.cm.gray)
    ax[0, 1].set_title('result after {}-level wavelet decomposition'.format(l))

    ax[1, 0].imshow(image_hat, cmap=plt.cm.gray)
    ax[1, 0].set_title('result after {}-level wavelet decomposition'.format(l))
    plt.show()

test_2()