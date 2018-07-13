"""Transforms on ndarray."""
from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
from skimage import transform, color
from numbers import Number
from scipy import ndimage as ndi


class NDTransform(object):
    """Base class for all numpy based transforms.

    This class achieves the following:

    * Abstract the transform into
        * Getting parameters to apply which is only run only once per __call__.
        * Applying transform given parameters
    * Check arguments passed to a transforms for consistency

    Abstraction is especially useful when there is randomness involved with the
    transform. You don't want to have different transforms applied to different
    members of a data point.
    """

    def _argcheck(self, data):
        """Check data for arguments."""
        if isinstance(data, np.ndarray):
            assert data.ndim in {2, 3}, \
                'Image should be a ndarray of shape H x W x C or H X W.'
            if data.ndim == 3:
                assert data.shape[2] < data.shape[0], \
                    'Is your color axis the last? Roll axes using np.rollaxis.'

            return data.shape[:2]
        elif isinstance(data, dict):
            for k, img in data.items():
                if isinstance(img, np.ndarray):
                    assert isinstance(k, str)

            shapes = {k: self._argcheck(img) for k, img in data.items()
                      if isinstance(img, np.ndarray)}
            assert len(set(shapes.values())) == 1, \
                'All member images must have same size. Instead got: {}'.format(shapes)
            return set(shapes.values()).pop()
        else:
            raise TypeError('ndarray or dict of ndarray can only be passed')

    def _get_params(self, h, w):
        """Get parameters of the transform to be applied for all member images.

        Implement this function if there are parameters to your transform which
        depend on the image size. Need not implement it if there are no such 
        parameters.

        Parameters
        ----------
        h: int
            Height of the image. i.e, img.shape[0].
        w: int
            Width of the image. i.e, img.shape[1].

        Returns
        -------
        params: dict
            Parameters of the transform in a dict with string keys.
            e.g. {'angle': 30}
        """
        return {}

    def _transform(self, img, is_label, **kwargs):
        """Apply the transform on an image.

        Use the parameters returned by _get_params and apply the transform on
        img. Be wary if the image is label or not.

        Parameters
        ----------
        img: ndarray
            Image to be transformed. Can be a color (H X W X C) or
            gray (H X W)image.
        is_label: bool
            True if image is to be considered as label, else False.
        **kwargs
            kwargs will be the dict returned by get_params

        Return
        ------
        img_transformed: ndarray
            Transformed image.
        """
        raise NotImplementedError

    def __call__(self, data):
        """
        Parameters
        ----------
        data: dict or ndarray
            Image ndarray or a dict of images. All ndarrays in the dict are 
            considered as images and should be of same size. If key for a
            image in dict has string `target` in it somewhere, it is
            considered as a target segmentation map.
        """
        h, w = self._argcheck(data)
        params = self._get_params(h, w)

        if isinstance(data, np.ndarray):
            return self._transform(data, is_label=False, **params)
        else:
            data = data.copy()
            for k, img in data.items():
                if isinstance(img, np.ndarray):
                    if isinstance(k, str) and 'target' in k:
                        is_label = True
                    else:
                        is_label = False

                    data[k] = self._transform(img, is_label, **params)
            return data


class Rescale(NDTransform):
    """Scale image by a certain factor.

    Parameters
    ----------
    scale : {float, tuple of floats}
        Scale factors. Separate scale factors for each axis can be defined
        as `(row_scale, col_scale)`.
    **kwargs: optional
        Other kwargs as described in skimage.transforms.rescale

    """

    def __init__(self, scale, **kwargs):
        self.scale = scale
        self.kwargs = kwargs

    def _transform(self, img, is_label):
        kwargs = self.kwargs.copy()
        if is_label:
            kwargs.update({'order': 0, 'preserve_range': True})

        return transform.rescale(img, self.scale, **kwargs)


class Resize(NDTransform):
    """Resize image to match a certain size.

    Parameters
    ----------
    output_shape : int or tuple
        Size of the generated output image `(rows, cols)`. If it is a
        number, aspect ratio of the image is preserved and smaller of the
        height and width is matched to it.
    **kwargs: optional
        Other params as described in skimage.transforms.resize

    """

    def __init__(self, output_shape, **kwargs):
        self.kwargs = kwargs
        self.output_shape = output_shape

    def _get_params(self, h, w):
        if isinstance(self.output_shape, Number):
            req = self.output_shape
            if h > w:
                output_shape = (int(h * req / w), req)
            else:
                output_shape = (req, int(w * req / h))
        else:
            output_shape = self.output_shape

        return {'output_shape': output_shape}

    def _transform(self, img, is_label, output_shape):
        kwargs = self.kwargs.copy()
        if is_label:
            kwargs.update({'order': 0, 'preserve_range': True})

        return transform.resize(img, output_shape, **kwargs)


class CenterCrop(NDTransform):
    """Crop to centered rectangle.

    Parameters
    ----------
    output_shape : int or tuple
        Size of the cropped image `(rows, cols)`. If it is a number,
        a square image with side of length `output_shape` will be cropped.
    """

    def __init__(self, output_shape):
        if isinstance(output_shape, Number):
            self.output_shape = (output_shape, output_shape)
        else:
            assert len(output_shape) == 2
            self.output_shape = output_shape

    def _get_params(self, h, w):
        new_h, new_w = self.output_shape

        assert (new_h <= h) and (new_w <= w), \
            'desired height/width larger than the image ({}, {})'.format(h, w)

        h1, w1 = (h - new_h) // 2, (w - new_w) // 2
        return {'h1': h1, 'w1': w1}

    def _transform(self, img, is_label, h1, w1):
        new_h, new_w = self.output_shape
        return img[h1: h1 + new_h, w1: w1 + new_w]


class RandomCrop(NDTransform):
    """Random crop from larger image.

    Parameters
    ----------
    output_shape : int or tuple
        Size of the cropped image `(rows, cols)`. If it is a number, a
        square image with side of length `output_shape` will be cropped.
    """

    def __init__(self, output_shape):
        if isinstance(output_shape, Number):
            self.output_shape = (output_shape, output_shape)
        else:
            assert len(output_shape) == 2
            self.output_shape = output_shape

    def _get_params(self, h, w):
        new_h, new_w = self.output_shape

        assert (new_h <= h) and (new_w <= w), \
            'desired height/width larger than the image ({}, {})'.format(h, w)

        h1 = 0 if h == new_h else np.random.randint(0, h - new_h)
        w1 = 0 if w == new_w else np.random.randint(0, w - new_w)

        return {'h1': h1, 'w1': w1}

    def _transform(self, img, is_label, h1, w1):
        new_h, new_w = self.output_shape
        return img[h1: h1 + new_h, w1: w1 + new_w]


class RandomScale(NDTransform):
    """Randomly resize the image.

    Resized so that aspect ratio is preserved and shorter side randomly
    sampled from [min_size, max_size).

    Parameters
    ----------
    min_size : int
        Lower bound of shorter side (inclusive)
    max_size: int
        Upper bound of shorter side (exclusive)
    **kwargs: optional
        Other kwargs as described in skimage.transforms.resize
    """

    def __init__(self, min_size, max_size, **kwargs):
        assert (min_size < max_size) and (min_size > 0)
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.kwargs = kwargs

    def _get_params(self, h, w):
        assert min(h, w) > self.max_size
        req = np.random.randint(self.min_size, self.max_size)
        if h > w:
            output_shape = (int(h * req / w), req)
        else:
            output_shape = (req, int(w * req / h))

        return {'output_shape': output_shape}

    def _transform(self, img, is_label, output_shape):
        kwargs = self.kwargs.copy()
        if is_label:
            kwargs.update({'order': 0, 'preserve_range': True})

        return transform.resize(img, output_shape, **kwargs)


class RandomSizedCrop(NDTransform):
    """Randomly sized crop within a specified size and aspect ratio range.

    An area fraction and a aspect ratio is sampled within frac_range and
    aspect_range respectively. Then sides of the crop are calculated using
    these two. Random crop of this size is finally resized to desired size.

    Parameters
    ----------
    output_shape: tuple
        `(rows, cols)` of output image.
    frac_range: sequence of length 2
        Range for fraction of the area to be sampled from.
    aspect_range: sequence of length 2
        Aspect ratio range to be sampled from.
    **kwargs: optional
        Other kwargs as described in skimage.transforms.resize
    """

    def __init__(self, output_shape, frac_range=[0.08, 1],
                 aspect_range=[3 / 4, 4 / 3], **kwargs):
        if isinstance(output_shape, Number):
            self.output_shape = (output_shape, output_shape)
        else:
            assert len(output_shape) == 2
            self.output_shape = output_shape

        self.frac_range = frac_range
        self.aspect_range = aspect_range
        self.kwargs = kwargs

    def _get_params(self, h, w):
        area = h * w

        attempts = 0
        while attempts < 10:
            try:
                targer_area = area * np.random.uniform(*self.frac_range)
                aspect_ratio = np.random.uniform(*self.aspect_range)

                new_h = int(np.sqrt(targer_area * aspect_ratio))
                new_w = int(np.sqrt(targer_area / aspect_ratio))

                if np.random.uniform() < 0.5:
                    new_h, new_w = new_w, new_h

                assert (new_h <= h) and (new_w <= w), 'Attempt failed'

                h1 = 0 if h == new_h else np.random.randint(0, h - new_h)
                w1 = 0 if w == new_w else np.random.randint(0, w - new_w)

                return {'h1': h1, 'w1': w1, 'new_h': new_h, 'new_w': new_w}
            except AssertionError:
                attempts += 1

        # fall back
        new_size = min(h, w)
        h1, w1 = (h - new_size) // 2, (w - new_size) // 2

        return {'h1': h1, 'w1': w1, 'new_h': new_size, 'new_w': new_size}

    def _transform(self, img, is_label, h1, w1, new_h, new_w):
        img = img[h1: h1 + new_h, w1: w1 + new_w]
        kwargs = self.kwargs.copy()
        if is_label:
            kwargs.update({'order': 0, 'preserve_range': True})

        return transform.resize(img, self.output_shape, **kwargs)


class RandomRotate(NDTransform):
    """Randomly rotate image.

    Parameters
    ----------
    angle_range: float or tuple
        Range of angles in degrees. If float, angle_range = (-theta, theta).
    kwargs: optional
        Other kwargs as described in skimage.transforms.rotate
    """

    def __init__(self, angle_range, **kwrgs):
        """Angle is in degrees."""
        if isinstance(angle_range, Number):
            assert angle_range > 0
            self.angle_range = (-angle_range, angle_range)
        else:
            self.angle_range = angle_range

        self.kwrgs = kwrgs

    def _get_params(self, h, w):
        angle = np.random.uniform(*self.angle_range)
        return {'angle': angle}

    def _transform(self, img, is_label, angle):
        kwrgs = self.kwrgs.copy()
        if is_label:
            kwrgs.update({'order': 0, 'preserve_range': True})

        return transform.rotate(img, angle, **kwrgs)


class RandomHorizontalFlip(NDTransform):
    """Flip horizontally with 0.5 probability."""

    def _get_params(self, h, w):
        return {'to_flip': np.random.uniform() < 0.5}

    def _transform(self, img, is_label, to_flip):
        if to_flip:
            return np.flip(img, 1)
        else:
            return img


class RandomVerticalFlip(NDTransform):
    """Flip vertically with 0.5 probability."""

    def _get_params(self, h, w):
        return {'to_flip': np.random.uniform() < 0.5}

    def _transform(self, img, is_label, to_flip):
        if to_flip:
            return np.flip(img, 0)
        else:
            return img


class Padding(NDTransform):
    """Pad images.

    Parameters
    ----------
    pad_width:
        See np.pad for exact documentation. For example, if pad_width is 20,
        20 pixels are padded before *and* after each dimension.
    params: dict
        Other params to be passed to `np.pad`. 'mode' is a important parameter.
        You can, for example, have reflections of image padded instead of
        zeros. Default mode is 'constant'.
    """

    def __init__(self, pad_width, mode='constant', **kwargs):
        if isinstance(pad_width, int):
            self.pad_width = ((pad_width, pad_width), (pad_width, pad_width))
        else:
            pad_width = np.array(pad_width)
            if pad_width.shape == (1,):
                pad_width = pad_width[0]
                self.pad_width = ((pad_width, pad_width),
                                  (pad_width, pad_width))
            elif pad_width.shape == (1, 2):
                self.pad_width = (pad_width[0], pad_width[0])
            elif pad_width.shape == (2, 2):
                self.pad_width = tuple(pad_width)
            else:
                raise TypeError('Unsupported pad_width')

        self.kwargs = kwargs
        self.kwargs['mode'] = mode

    def _transform(self, img, is_label):
        pad_width = self.pad_width
        if img.ndim == 3:
            pad_width = pad_width + ((0, 0),)

        return np.pad(img, pad_width, **self.kwargs)


class MinimumPadding(NDTransform):
    """Pad an image so that it is at least of a minimum size.

    Parameters
    ----------
    min_shape: tuple
        Sequence which looks like (H, W). Image is padded such that it's height
        and width is at least H and W respectively.
    params: dict
        Other params to be passed to `np.pad`. 'mode' is a important parameter.
        You can, for example, have reflections of image padded instead of
        zeros. Default mode is 'constant'.
    """

    def __init__(self, min_shape, mode='constant', **kwargs):
        self.min_shape = min_shape
        self.kwargs = kwargs
        self.kwargs['mode'] = mode

    def _get_params(self, h, w):
        total_padding = [max(0, self.min_shape[0] - h),
                         max(0, self.min_shape[1] - w)]
        pad_width = [(total_padding[0] // 2,
                      total_padding[0] - total_padding[0] // 2),
                     (total_padding[1] // 2,
                      total_padding[1] - total_padding[1] // 2)]

        return {'pad_width': pad_width}

    def _transform(self, img, is_label, pad_width):
        if img.ndim == 3:
            pad_width = pad_width + [(0, 0)]

        return np.pad(img, pad_width, **self.kwargs)


class ElasticTransform(NDTransform):
    """Elastic transform.

    Flow is sampled from independent uniform distribution for each pixel
    and then smoothed using a Gaussian smoothing kernel. This flow is used to
    transform the image.

    Parameters
    ----------
    alpha: float
        Intensity of unsmoothened flow.
    sigma: float
        Width of Gaussian smoothing kernel.
    **kwargs: optional
        Other kwargs as described in ndi.interpolation.map_coordinates
    """

    def __init__(self, alpha, sigma, **kwargs):
        self.alpha = alpha
        self.sigma = sigma
        self.kwargs = kwargs

    def _get_params(self, h, w):
        flow = self.alpha * (2 * np.random.rand(2, h, w) - 1)
        smooth_flow = np.array([
            ndi.filters.gaussian_filter(x, self.sigma, mode='constant', cval=0)
            for x in flow
        ])
        return {'flow': tuple(smooth_flow)}

    def _transform(self, img, is_label, flow):
        kwargs = self.kwargs.copy()
        if is_label:
            kwargs['order'] = 0

        mesh = np.meshgrid(*[np.arange(x) for x in img.shape],
                           indexing='ij')
        if img.ndim == 3:
            indices = (mesh[0] + flow[0][:, :, np.newaxis],
                       mesh[1] + flow[1][:, :, np.newaxis],
                       mesh[2])
        else:
            indices = (mesh[0] + flow[0], mesh[1] + flow[1])

        indices = [np.reshape(x, (-1, 1)) for x in indices]

        out = ndi.interpolation.map_coordinates(img, indices, **kwargs)
        return out.reshape(img.shape)


class Normalize(NDTransform):
    """Normalize image.


    Parameters
    ----------
    norm_params: sequence of length 2
        Normalization parameters based on mode
    mode: {'percentile', 'meanstd'}
        Normalization mode. If 'percentile',  norm_params is (p_low, p_high)
        where p_low and p_high are low and high percentiles respectively.
        if 'meanstd', norm_params is (mean, std).
    """

    def __init__(self, norm_params, mode='percentile'):
        self.mode = mode
        self.norm_params = norm_params
        assert self.mode in {'percentile', 'meanstd'}
        assert len(norm_params) == 2

    def _transform(self, img, is_label):
        if is_label:
            return img
        else:
            if self.mode == 'percentile':
                p_low, p_hi = self.norm_params
                low = np.percentile(img, p_low)
                hi = np.percentile(img, p_hi)
                return (img - low) / (hi - low)
            else:
                mean, std = self.norm_params
                return (img - mean) / std


class RandomIntensityJitter(NDTransform):
    """Random jitters in brightness, contrast and saturation.

    Parameters
    ----------
    brightness: float
        Intensity of brightness jitter. float in [0, 1].
    contrast: float
        Intensity of contrast jitter. float in [0, 1].
    saturation: float
        Intensity of saturation jitter. float in [0, 1].
    """

    def __init__(self, brightness=0, contrast=0, saturation=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def _blend(self, img1, img2, alpha):
        return img1 * alpha + img2 * (1 - alpha)

    def _gs(self, img):
        if img.ndim == 3:
            return color.rgb2gray(img)[:, :, np.newaxis]
        else:
            return img

    def _saturation(self, img, var):
        if var == 0 or img.ndim == 3:
            return img

        return self._blend(img, self._gs(img), var)

    def _brightness(self, img, var):
        if var == 0:
            return img

        return self._blend(img, np.zeros_like(img), var)

    def _contrast(self, img, var):
        if var == 0:
            return img

        mean_img = np.full_like(img, self._gs(img).mean())
        return self._blend(img, mean_img, var)

    def _get_params(self, h, w):
        vars = [1 + self.brightness * np.random.uniform(-1, 1),
                1 + self.contrast * np.random.uniform(-1, 1),
                1 + self.saturation * np.random.uniform(-1, 1)]

        return {'vars': vars, 'order': np.random.permutation(3)}

    def _transform(self, img, is_label, vars, order):
        if is_label:
            return img
        else:
            tsfrms = [self._brightness, self._contrast, self._saturation]
            for i in order:
                img = tsfrms[i](img, vars[i])

            return img


class Clip(NDTransform):
    """Clip numpy array by a low and high value.

    Linear transformation can be applied after clipping so that new desired
    high and lows can be set.

    Default behaviour is to clip with 0 and 1 as bounds.

    Parameters
    ----------
    inp_low: float, optional
        Minimum value of the input, default is 0.
    inp_high: float, optional
        Maximum value of the input, default is 1,
    out_low: float, optional
        New minimum value for the output, default is inp_low.
    out_high: float, optional
        New Maximum value for the output, default is inp_high.

    """

    def __init__(self, inp_low=0, inp_high=1, out_low=None, out_high=None):
        self.inp_low = inp_low
        self.inp_high = inp_high
        self.out_low = out_low if out_low is not None else inp_low
        self.out_high = out_high if out_high is not None else inp_high

    def _transform(self, img, is_label):
        if is_label:
            return img
        else:
            img = (img - self.inp_low) / (self.inp_high - self.inp_low)
            img = np.clip(img, 0, 1)
            img = self.out_low + (self.out_high - self.out_low) * img

            return img


class ToTensor(NDTransform):
    """Convert ndarrays to tensors.

    Following are taken care of when converting to tensors:

    * Axes are swapped so that color axis is in front of rows and columns
    * A color axis is added in case of gray images
    * Target images are left alone and are directly converted
    * Label images is set to LongTensor by default as expected by torch's loss
      functions.

    Parameters
    ----------
    dtype: torch dtype
        If you want to convert all tensors to cuda, you can directly
        set dtype=torch.cuda.FloatTensor. This is for non label images
    dtype_label: torch dtype
        Same as above but for label images.
    """

    import torch

    def __init__(self, dtype='torch.FloatTensor',
                 dtype_label='torch.LongTensor'):
        self.dtype = dtype
        self.dtype_label = dtype_label

    def _transform(self, img, is_label):
        img = np.ascontiguousarray(img)
        if not is_label:
            # put it from HWC to CHW format
            if img.ndim == 3:
                img = np.rollaxis(img, 2, 0)
            elif img.ndim == 2:
                img = img.reshape((1,) + img.shape)

        img = self.torch.from_numpy(img)

        dtype = self.dtype_label if is_label else self.dtype
        if dtype is not None:
            return img.type(dtype)
        else:
            return img
