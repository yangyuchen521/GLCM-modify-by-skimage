#skimage/feature/texture.py the file which I have modified 

"""
Methods to characterize image textures.
"""

import warnings

import numpy as np
from .._shared.utils import check_nD
from ..color import gray2rgb#colorconv.py
from ..util import img_as_float#dtype.py
from ._texture import _glcm_loop, _local_binary_pattern, _multiblock_lbp


'''
check_nD(array, ndim, arg_name='image'):
    
    Verify an array meets the desired ndims and array isn't empty.

    Parameters
    ----------
    array : array-like
        Input array to be validated
    ndim : int or iterable of ints
        Allowable ndim or ndims for the array.
    arg_name : str, optional
        The name of the array in the original function.

    
    array = np.asanyarray(array)
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if array.size == 0:
        raise ValueError(msg_empty_array % (arg_name))
    if array.ndim not in ndim:
        raise ValueError(
            msg_incorrect_dim % (arg_name, '-or-'.join([str(n) for n in ndim]))
        )

'''
'''
def img_as_float(image, force_copy=False):
    """Convert an image to floating point format.

    This function is similar to `img_as_float64`, but will not convert
    lower-precision floating point arrays to `float64`.

    Parameters
    ----------
    image : ndarray
        Input image.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.

    Returns
    -------
    out : ndarray of float
        Output image.

    Notes
    -----
    The range of a floating point image is [0.0, 1.0] or [-1.0, 1.0] when
    converting from unsigned or signed datatypes, respectively.
    If the input image has a float type, intensity values are not modified
    and can be outside the ranges [0.0, 1.0] or [-1.0, 1.0].

    """
    return _convert(image, np.floating, force_copy)

def _convert(image, dtype, force_copy=False, uniform=False):
    """
    Convert an image to the requested data-type.

    Warnings are issued in case of precision loss, or when negative values
    are clipped during conversion to unsigned integer types (sign loss).

    Floating point values are expected to be normalized and will be clipped
    to the range [0.0, 1.0] or [-1.0, 1.0] when converting to unsigned or
    signed integers respectively.

    Numbers are not shifted to the negative side when converting from
    unsigned to signed integer types. Negative values will be clipped when
    converting to unsigned integers.

    Parameters
    ----------
    image : ndarray
        Input image.
    dtype : dtype
        Target data-type.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    uniform : bool, optional
        Uniformly quantize the floating point range to the integer range.
        By default (uniform=False) floating point values are scaled and
        rounded to the nearest integers, which minimizes back and forth
        conversion errors.

    .. versionchanged:: 0.15
        ``_convert`` no longer warns about possible precision or sign
        information loss. See discussions on these warnings at:
        https://github.com/scikit-image/scikit-image/issues/2602
        https://github.com/scikit-image/scikit-image/issues/543#issuecomment-208202228
        https://github.com/scikit-image/scikit-image/pull/3575

    References
    ----------
    .. [1] DirectX data conversion rules.
           https://msdn.microsoft.com/en-us/library/windows/desktop/dd607323%28v=vs.85%29.aspx
    .. [2] Data Conversions. In "OpenGL ES 2.0 Specification v2.0.25",
           pp 7-8. Khronos Group, 2010.
    .. [3] Proper treatment of pixels as integers. A.W. Paeth.
           In "Graphics Gems I", pp 249-256. Morgan Kaufmann, 1990.
    .. [4] Dirty Pixels. J. Blinn. In "Jim Blinn's corner: Dirty Pixels",
           pp 47-57. Morgan Kaufmann, 1998.

    """
    image = np.asarray(image)
    dtypeobj_in = image.dtype
    if dtype is np.floating:
        dtypeobj_out = np.dtype('float64')
    else:
        dtypeobj_out = np.dtype(dtype)
    dtype_in = dtypeobj_in.type
    dtype_out = dtypeobj_out.type
    kind_in = dtypeobj_in.kind
    kind_out = dtypeobj_out.kind
    itemsize_in = dtypeobj_in.itemsize
    itemsize_out = dtypeobj_out.itemsize

    # Below, we do an `issubdtype` check.  Its purpose is to find out
    # whether we can get away without doing any image conversion.  This happens
    # when:
    #
    # - the output and input dtypes are the same or
    # - when the output is specified as a type, and the input dtype
    #   is a subclass of that type (e.g. `np.floating` will allow
    #   `float32` and `float64` arrays through)

    if np.issubdtype(dtype_in, dtype):
        if force_copy:
            image = image.copy()
        return image

    if not (dtype_in in _supported_types and dtype_out in _supported_types):
        raise ValueError(f'Cannot convert from {dtypeobj_in} to ' f'{dtypeobj_out}.')

    if kind_in in 'ui':
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max
    if kind_out in 'ui':
        imin_out = np.iinfo(dtype_out).min
        imax_out = np.iinfo(dtype_out).max

    # any -> binary
    if kind_out == 'b':
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    # binary -> any
    if kind_in == 'b':
        result = image.astype(dtype_out)
        if kind_out != 'f':
            result *= dtype_out(dtype_range[dtype_out][1])
        return result

    # float -> any
    if kind_in == 'f':
        if kind_out == 'f':
            # float -> float
            return image.astype(dtype_out)

        if np.min(image) < -1.0 or np.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        # floating point -> integer
        # use float type that can represent output integer type
        computation_type = _dtype_itemsize(
            itemsize_out, dtype_in, np.float32, np.float64
        )

        if not uniform:
            if kind_out == 'u':
                image_out = np.multiply(image, imax_out, dtype=computation_type)
            else:
                image_out = np.multiply(
                    image, (imax_out - imin_out) / 2, dtype=computation_type
                )
                image_out -= 1.0 / 2.0
            np.rint(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        elif kind_out == 'u':
            image_out = np.multiply(image, imax_out + 1, dtype=computation_type)
            np.clip(image_out, 0, imax_out, out=image_out)
        else:
            image_out = np.multiply(
                image, (imax_out - imin_out + 1.0) / 2.0, dtype=computation_type
            )
            np.floor(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        return image_out.astype(dtype_out)

    # signed/unsigned int -> float
    if kind_out == 'f':
        # use float type that can exactly represent input integers
        computation_type = _dtype_itemsize(
            itemsize_in, dtype_out, np.float32, np.float64
        )

        if kind_in == 'u':
            # using np.divide or np.multiply doesn't copy the data
            # until the computation time
            image = np.multiply(image, 1.0 / imax_in, dtype=computation_type)
            # DirectX uses this conversion also for signed ints
            # if imin_in:
            #     np.maximum(image, -1.0, out=image)
        elif kind_in == 'i':
            # From DirectX conversions:
            # The most negative value maps to -1.0f
            # Every other value is converted to a float (call it c)
            # and then result = c * (1.0f / (2⁽ⁿ⁻¹⁾-1)).

            image = np.multiply(image, 1.0 / imax_in, dtype=computation_type)
            np.maximum(image, -1.0, out=image)

        else:
            image = np.add(image, 0.5, dtype=computation_type)
            image *= 2 / (imax_in - imin_in)

        return np.asarray(image, dtype_out)

    # unsigned int -> signed/unsigned int
    if kind_in == 'u':
        if kind_out == 'i':
            # unsigned int -> signed int
            image = _scale(image, 8 * itemsize_in, 8 * itemsize_out - 1)
            return image.view(dtype_out)
        else:
            # unsigned int -> unsigned int
            return _scale(image, 8 * itemsize_in, 8 * itemsize_out)

    # signed int -> unsigned int
    if kind_out == 'u':
        image = _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out)
        result = np.empty(image.shape, dtype_out)
        np.maximum(image, 0, out=result, dtype=image.dtype, casting='unsafe')
        return result

    # signed int -> signed int
    if itemsize_in > itemsize_out:
        return _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out - 1)

    image = image.astype(_dtype_bits('i', itemsize_out * 8))
    image -= imin_in
    image = _scale(image, 8 * itemsize_in, 8 * itemsize_out, copy=False)
    image += imin_out
    return image.astype(dtype_out)




'''
'''
def gray2rgb(image, *, channel_axis=-1):
    """Create an RGB representation of a gray-level image.

    Parameters
    ----------
    image : array_like
        Input image.
    channel_axis : int, optional
        This parameter indicates which axis of the output array will correspond
        to channels.

    Returns
    -------
    rgb : (..., C=3, ...) ndarray
        RGB image. A new dimension of length 3 is added to input image.

    Notes
    -----
    If the input is a 1-dimensional image of shape ``(M,)``, the output
    will be shape ``(M, C=3)``.
    """
    return np.stack(3 * (image,), axis=channel_axis)
'''
def graycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False):
    """Calculate the gray-level co-occurrence matrix.

    A gray level co-occurrence matrix is a histogram of co-occurring
    grayscale values at a given offset over an image.

    .. versionchanged:: 0.19
               `greymatrix` was renamed to `graymatrix` in 0.19.

    Parameters
    ----------
    image : array_like
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    distances : array_like
        List of pixel pair distance offsets.
    angles : array_like
        List of pixel pair angles in radians.
    levels : int, optional
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of gray-levels counted
        (typically 256 for an 8-bit image). This argument is required for
        16-bit images or higher and is typically the maximum of the image.
        As the output matrix is at least `levels` x `levels`, it might
        be preferable to use binning of the input image rather than
        large values for `levels`.
    symmetric : bool, optional
        If True, the output matrix `P[:, :, d, theta]` is symmetric. This
        is accomplished by ignoring the order of value pairs, so both
        (i, j) and (j, i) are accumulated when (i, j) is encountered
        for a given offset. The default is False.
    normed : bool, optional
        If True, normalize each matrix `P[:, :, d, theta]` by dividing
        by the total number of accumulated co-occurrences for the given
        offset. The elements of the resulting matrix sum to 1. The
        default is False.

    Returns
    -------
    P : 4-D ndarray
        The gray-level co-occurrence histogram. The value
        `P[i,j,d,theta]` is the number of times that gray-level `j`
        occurs at a distance `d` and at an angle `theta` from
        gray-level `i`. If `normed` is `False`, the output is of
        type uint32, otherwise it is float64. The dimensions are:
        levels x levels x number of distances x number of angles.

    References
    ----------
    .. [1] M. Hall-Beyer, 2007. GLCM Texture: A Tutorial
           https://prism.ucalgary.ca/handle/1880/51900
           DOI:`10.11575/PRISM/33280`
    .. [2] R.M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for
           image classification", IEEE Transactions on Systems, Man, and
           Cybernetics, vol. SMC-3, no. 6, pp. 610-621, Nov. 1973.
           :DOI:`10.1109/TSMC.1973.4309314`
    .. [3] M. Nadler and E.P. Smith, Pattern Recognition Engineering,
           Wiley-Interscience, 1993.
    .. [4] Wikipedia, https://en.wikipedia.org/wiki/Co-occurrence_matrix


    Examples
    --------
    Compute 2 GLCMs: One for a 1-pixel offset to the right, and one
    for a 1-pixel offset upwards.

    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> result = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
    ...                       levels=4)
    >>> result[:, :, 0, 0]
    array([[2, 2, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 1],
           [0, 0, 0, 1]], dtype=uint32)
    >>> result[:, :, 0, 1]
    array([[1, 1, 3, 0],
           [0, 1, 1, 0],
           [0, 0, 0, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 2]
    array([[3, 0, 2, 0],
           [0, 2, 2, 0],
           [0, 0, 1, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 3]
    array([[2, 0, 0, 0],
           [1, 1, 2, 0],
           [0, 0, 2, 1],
           [0, 0, 0, 0]], dtype=uint32)

    """
    check_nD(image, 2)
    check_nD(distances, 1, 'distances')
    check_nD(angles, 1, 'angles')

    image = np.ascontiguousarray(image)

    image_max = image.max()#序列最值和最小值

    if np.issubdtype(image.dtype, np.floating):
        raise ValueError(
            "Float images are not supported by graycomatrix. "
            "Convert the image to an unsigned integer type."
        )

    # for image type > 8bit, levels must be set.
    if image.dtype not in (np.uint8, np.int8) and levels is None:
        raise ValueError(
            "The levels argument is required for data types "
            "other than uint8. The resulting matrix will be at "
            "least levels ** 2 in size."
        )

    if np.issubdtype(image.dtype, np.signedinteger) and np.any(image < 0):
        raise ValueError("Negative-valued images are not supported.")

    if levels is None:
        levels = 256

    if image_max >= levels:
        raise ValueError(
            "The maximum grayscale value in the image should be "
            "smaller than the number of levels."
        )

    distances = np.ascontiguousarray(distances, dtype=np.float64)
    angles = np.ascontiguousarray(angles, dtype=np.float64)#在记忆里面连续生成数组

    P = np.zeros(
        (levels, levels, len(distances), len(angles)), dtype=np.uint32, order='C'
    )

    # count co-occurences
    _glcm_loop(image, distances, angles, levels, P)

    # make each GLMC symmetric对称矩阵
    if symmetric:
        Pt = np.transpose(P, (1, 0, 2, 3))
        P = P + Pt

    # normalize each GLCM
    if normed:
        P = P.astype(np.float64)
        glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums

    return P


def graycoprops(P, prop='contrast'):
    """Calculate texture properties of a GLCM.

    Compute a feature of a gray level co-occurrence matrix to serve as
    a compact summary of the matrix. The properties are computed as
    follows:

    - 'contrast': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-j)^2`
    - 'dissimilarity': :math:`\\sum_{i,j=0}^{levels-1}P_{i,j}|i-j|`
    - 'homogeneity': :math:`\\sum_{i,j=0}^{levels-1}\\frac{P_{i,j}}{1+(i-j)^2}`
    - 'ASM': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}^2`
    - 'energy': :math:`\\sqrt{ASM}`
    - 'correlation':
        .. math:: \\sum_{i,j=0}^{levels-1} P_{i,j}\\left[\\frac{(i-\\mu_i) \\
                  (j-\\mu_j)}{\\sqrt{(\\sigma_i^2)(\\sigma_j^2)}}\\right]

    Each GLCM is normalized to have a sum of 1 before the computation of
    texture properties.

    .. versionchanged:: 0.19
           `greycoprops` was renamed to `graycoprops` in 0.19.

    Parameters
    ----------
    P : ndarray
        Input array. `P` is the gray-level co-occurrence histogram
        for which to compute the specified property. The value
        `P[i,j,d,theta]` is the number of times that gray-level j
        occurs at a distance d and at an angle theta from
        gray-level i.
    prop : {'contrast', 'dissimilarity', 'homogeneity', 'energy', \
            'correlation', 'ASM'}, optional
    ##########################################
    plus:entropy,autocorrelation,cluster prominence,sum of square,sum average,sum variance,difference entropy,difference variance
        The property of the GLCM to compute. The default is 'contrast'.

    Returns
    -------
    results : 2-D ndarray
        2-dimensional array. `results[d, a]` is the property 'prop' for
        the d'th distance and the a'th angle.

    References
    ----------
    .. [1] M. Hall-Beyer, 2007. GLCM Texture: A Tutorial v. 1.0 through 3.0.
           The GLCM Tutorial Home Page,
           https://prism.ucalgary.ca/handle/1880/51900
           DOI:`10.11575/PRISM/33280`

    Examples
    --------
    Compute the contrast for GLCMs with distances [1, 2] and angles
    [0 degrees, 90 degrees]

    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> g = graycomatrix(image, [1, 2], [0, np.pi/2], levels=4,
    ...                  normed=True, symmetric=True)
    >>> contrast = graycoprops(g, 'contrast')
    >>> contrast
    array([[0.58333333, 1.        ],
           [1.25      , 2.75      ]])

    """
    check_nD(P, 4, 'P')

    (num_level, num_level2, num_dist, num_angle) = P.shape
    if num_level != num_level2:
        raise ValueError('num_level and num_level2 must be equal.')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive.')
    if num_angle <= 0:
        raise ValueError('num_angle must be positive.')

    # normalize each GLCM 归一化p
    P = P.astype(np.float64)
    glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums
    # create weights for specified property
    I, J = np.ogrid[0:num_level, 0:num_level]
    if prop == 'contrast':
        weights = (I - J) ** 2
    elif prop == 'dissimilarity':
        weights = np.abs(I - J)
    elif prop == 'homogeneity':
        weights = 1.0 / (1.0 + (I - J) ** 2)

#plus:entropy,autocorrelation,cluster prominence,sum of squares(joint variance),sum average,sum variance,difference entropy,difference variance


    elif prop == 'sum average':
        weights = I

    
    elif prop in ['entropy','ASM', 'energy', 'correlation','autocorrelation','difference variance','sum of squares','cluster prominence','sum variance','difference entropy']:
        pass
    else:
        raise ValueError(f'{prop} is an invalid property')

    # compute property for each GLCM
    if prop == 'energy':
        asm = np.sum(P**2, axis=(0, 1))
        results = np.sqrt(asm)
   #     
    elif prop == 'entropy':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        results = np.sum((-1) * P * np.log2(P + 1e-5))
    elif prop == 'ASM':
        results = np.sum(P**2, axis=(0, 1))
    elif prop == 'correlation':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.sum(I * P, axis=(0, 1))#I-myu
        diff_j = J - np.sum(J * P, axis=(0, 1))

        std_i = np.sqrt(np.sum(P * (diff_i) ** 2, axis=(0, 1)))#sigema
        std_j = np.sqrt(np.sum(P * (diff_j) ** 2, axis=(0, 1)))
        cov = np.sum(P * (diff_i * diff_j), axis=(0, 1))

        # handle the special case of standard deviations near zero
        mask_0 = std_i < 1e-15
        mask_0[std_j < 1e-15] = True
        results[mask_0] = 1

        # handle the standard case
        mask_1 = ~mask_0
        results[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])

    elif prop == 'autocorrelation':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        weights = I * J
        results = np.sum(P * weights, axis=(0, 1))
    elif prop == 'difference variance':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        weights = np.abs(I - J)
        myu_dv = np.sum(P * weights, axis=(0, 1))
        k = np.abs(I - J)
        weights = (k - myu_dv) ** 2
        results = np.sum(P * weights, axis=(0, 1))
        
    elif prop == 'sum of squares':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        myu_f = np.sum(P * I,axis=(0,1))#joint average
        weights = (I - myu_f) ** 2
        results = np.sum(P * weights, axis=(0, 1))
        
    elif prop == 'cluster prominence':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        myu_i = np.sum(I * P,axis=(0,1)) 
        myu_j = np.sum(J * P,axis=(0,1))
        results = np.sum(P * (I + J - myu_i - myu_j) ** 4)
        
    elif prop == 'sum variance':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        myu_i = np.sum(I * P,axis=(0,1))
        myu_j = np.sum(J * P,axis=(0,1))
        weights = (I + J - myu_i - myu_j) ** 2
        results = np.sum(P * weights,axis=(0,1))


    elif prop == 'sum average':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        results = np.sum(P * I, axis=(0, 1)) * 2

    elif prop == 'difference entropy':
        p_ij = np.zeros((num_level, num_level, 2, 4),dtype=np.float64)
        for i in range(num_level):
            for j in range(num_level):
                k = abs(i - j)
                p_ij[abs(i-j)][k] += P[i, j]
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        weights = np.log(p_ij + 1e-5)
        results = -np.sum(p_ij * weights,axis=(0,1))
        
        
        
    elif prop in ['contrast', 'dissimilarity', 'homogeneity']:
        weights = weights.reshape((num_level, num_level, 1, 1))#sum of squares?
        results = np.sum(P * weights, axis=(0, 1))

    return results


def local_binary_pattern(image, P, R, method='default'):
    """Compute the local binary patterns (LBP) of an image.

    LBP is a visual descriptor often used in texture classification.

    Parameters
    ----------
    image : (M, N) array
        2D grayscale image.
    P : int
        Number of circularly symmetric neighbor set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    method : str {'default', 'ror', 'uniform', 'nri_uniform', 'var'}, optional
        Method to determine the pattern:

        ``default``
            Original local binary pattern which is grayscale invariant but not
            rotation invariant.
        ``ror``
            Extension of default pattern which is grayscale invariant and
            rotation invariant.
        ``uniform``
            Uniform pattern which is grayscale invariant and rotation
            invariant, offering finer quantization of the angular space.
            For details, see [1]_.
        ``nri_uniform``
            Variant of uniform pattern which is grayscale invariant but not
            rotation invariant. For details, see [2]_ and [3]_.
        ``var``
            Variance of local image texture (related to contrast)
            which is rotation invariant but not grayscale invariant.

    Returns
    -------
    output : (M, N) array
        LBP image.

    References
    ----------
    .. [1] T. Ojala, M. Pietikainen, T. Maenpaa, "Multiresolution gray-scale
           and rotation invariant texture classification with local binary
           patterns", IEEE Transactions on Pattern Analysis and Machine
           Intelligence, vol. 24, no. 7, pp. 971-987, July 2002
           :DOI:`10.1109/TPAMI.2002.1017623`
    .. [2] T. Ahonen, A. Hadid and M. Pietikainen. "Face recognition with
           local binary patterns", in Proc. Eighth European Conf. Computer
           Vision, Prague, Czech Republic, May 11-14, 2004, pp. 469-481, 2004.
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.214.6851
           :DOI:`10.1007/978-3-540-24670-1_36`
    .. [3] T. Ahonen, A. Hadid and M. Pietikainen, "Face Description with
           Local Binary Patterns: Application to Face Recognition",
           IEEE Transactions on Pattern Analysis and Machine Intelligence,
           vol. 28, no. 12, pp. 2037-2041, Dec. 2006
           :DOI:`10.1109/TPAMI.2006.244`
    """
    check_nD(image, 2)

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V'),
    }
    if np.issubdtype(image.dtype, np.floating):
        warnings.warn(
            "Applying `local_binary_pattern` to floating-point images may "
            "give unexpected results when small numerical differences between "
            "adjacent pixels are present. It is recommended to use this "
            "function with images of integer dtype."
        )
    image = np.ascontiguousarray(image, dtype=np.float64)
    output = _local_binary_pattern(image, P, R, methods[method.lower()])
    return output


def multiblock_lbp(int_image, r, c, width, height):
    """Multi-block local binary pattern (MB-LBP).

    The features are calculated similarly to local binary patterns (LBPs),
    (See :py:meth:`local_binary_pattern`) except that summed blocks are
    used instead of individual pixel values.

    MB-LBP is an extension of LBP that can be computed on multiple scales
    in constant time using the integral image. Nine equally-sized rectangles
    are used to compute a feature. For each rectangle, the sum of the pixel
    intensities is computed. Comparisons of these sums to that of the central
    rectangle determine the feature, similarly to LBP.

    Parameters
    ----------
    int_image : (N, M) array
        Integral image.
    r : int
        Row-coordinate of top left corner of a rectangle containing feature.
    c : int
        Column-coordinate of top left corner of a rectangle containing feature.
    width : int
        Width of one of the 9 equal rectangles that will be used to compute
        a feature.
    height : int
        Height of one of the 9 equal rectangles that will be used to compute
        a feature.

    Returns
    -------
    output : int
        8-bit MB-LBP feature descriptor.

    References
    ----------
    .. [1] L. Zhang, R. Chu, S. Xiang, S. Liao, S.Z. Li. "Face Detection Based
           on Multi-Block LBP Representation", In Proceedings: Advances in
           Biometrics, International Conference, ICB 2007, Seoul, Korea.
           http://www.cbsr.ia.ac.cn/users/scliao/papers/Zhang-ICB07-MBLBP.pdf
           :DOI:`10.1007/978-3-540-74549-5_2`
    """

    int_image = np.ascontiguousarray(int_image, dtype=np.float32)
    lbp_code = _multiblock_lbp(int_image, r, c, width, height)
    return lbp_code


def draw_multiblock_lbp(
    image,
    r,
    c,
    width,
    height,
    lbp_code=0,
    color_greater_block=(1, 1, 1),
    color_less_block=(0, 0.69, 0.96),
    alpha=0.5,
):
    """Multi-block local binary pattern visualization.

    Blocks with higher sums are colored with alpha-blended white rectangles,
    whereas blocks with lower sums are colored alpha-blended cyan. Colors
    and the `alpha` parameter can be changed.

    Parameters
    ----------
    image : ndarray of float or uint
        Image on which to visualize the pattern.
    r : int
        Row-coordinate of top left corner of a rectangle containing feature.
    c : int
        Column-coordinate of top left corner of a rectangle containing feature.
    width : int
        Width of one of 9 equal rectangles that will be used to compute
        a feature.
    height : int
        Height of one of 9 equal rectangles that will be used to compute
        a feature.
    lbp_code : int
        The descriptor of feature to visualize. If not provided, the
        descriptor with 0 value will be used.
    color_greater_block : tuple of 3 floats
        Floats specifying the color for the block that has greater
        intensity value. They should be in the range [0, 1].
        Corresponding values define (R, G, B) values. Default value
        is white (1, 1, 1).
    color_greater_block : tuple of 3 floats
        Floats specifying the color for the block that has greater intensity
        value. They should be in the range [0, 1]. Corresponding values define
        (R, G, B) values. Default value is cyan (0, 0.69, 0.96).
    alpha : float
        Value in the range [0, 1] that specifies opacity of visualization.
        1 - fully transparent, 0 - opaque.

    Returns
    -------
    output : ndarray of float
        Image with MB-LBP visualization.

    References
    ----------
    .. [1] L. Zhang, R. Chu, S. Xiang, S. Liao, S.Z. Li. "Face Detection Based
           on Multi-Block LBP Representation", In Proceedings: Advances in
           Biometrics, International Conference, ICB 2007, Seoul, Korea.
           http://www.cbsr.ia.ac.cn/users/scliao/papers/Zhang-ICB07-MBLBP.pdf
           :DOI:`10.1007/978-3-540-74549-5_2`
    """

    # Default colors for regions.
    # White is for the blocks that are brighter.
    # Cyan is for the blocks that has less intensity.
    color_greater_block = np.asarray(color_greater_block, dtype=np.float64)
    color_less_block = np.asarray(color_less_block, dtype=np.float64)

    # Copy array to avoid the changes to the original one.
    output = np.copy(image)

    # As the visualization uses RGB color we need 3 bands.
    if len(image.shape) < 3:
        output = gray2rgb(image)

    # Colors are specified in floats.
    output = img_as_float(output)

    # Offsets of neighbor rectangles relative to central one.
    # It has order starting from top left and going clockwise.
    neighbor_rect_offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    )

    # Pre-multiply the offsets with width and height.
    neighbor_rect_offsets = np.array(neighbor_rect_offsets)
    neighbor_rect_offsets[:, 0] *= height
    neighbor_rect_offsets[:, 1] *= width

    # Top-left coordinates of central rectangle.
    central_rect_r = r + height
    central_rect_c = c + width

    for element_num, offset in enumerate(neighbor_rect_offsets):
        offset_r, offset_c = offset

        curr_r = central_rect_r + offset_r
        curr_c = central_rect_c + offset_c

        has_greater_value = lbp_code & (1 << (7 - element_num))

        # Mix-in the visualization colors.
        if has_greater_value:
            new_value = (1 - alpha) * output[
                curr_r : curr_r + height, curr_c : curr_c + width
            ] + alpha * color_greater_block
            output[curr_r : curr_r + height, curr_c : curr_c + width] = new_value
        else:
            new_value = (1 - alpha) * output[
                curr_r : curr_r + height, curr_c : curr_c + width
            ] + alpha * color_less_block
            output[curr_r : curr_r + height, curr_c : curr_c + width] = new_value

    return output
