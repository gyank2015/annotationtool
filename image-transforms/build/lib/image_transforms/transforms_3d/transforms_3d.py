"""Transforms an image."""
from __future__ import print_function, division
import numpy as np
import SimpleITK as sitk
from numbers import Number
import itertools


class SITKTransform(object):
    """Baseclass for all SimpleITK based transforms."""

    def _argcheck(self, data):
        if isinstance(data, sitk.Image):
            try:
                if self.ndim is not None:
                    assert data.GetDimension() == self.ndim, \
                        'Expected {} dim image, got {}'.format(self.ndim,
                            data.GetDimension())
            except AttributeError:
                pass

            return data
        elif isinstance(data, dict):
            sizes = {k: self._argcheck(img).GetSize()
                     for k, img in data.items()
                     if isinstance(img, sitk.Image)}
            assert len(set(sizes.values())) == 1, \
                'All images should have same size. Got {}'.format(sizes)

            k = sizes.popitem()[0]  # get one element
            return data[k]
        else:
            raise TypeError('dict or Image has to be passed. Got : {}'.format(
                type(data)))

    def _get_params(self, img):
        return {}

    def _transform(self, img, is_label, **kwargs):
        raise NotImplementedError

    def __call__(self, data):
        img = self._argcheck(data)
        params = self._get_params(img)

        if isinstance(data, dict):
            data = data.copy()
            for k, img in data.items():
                if isinstance(img, sitk.Image):
                    if isinstance(k, str) and 'target' in k:
                        is_label = True
                    else:
                        is_label = False

                    data[k] = self._transform(img, is_label, **params)
            return data
        else:
            return self._transform(data, is_label=False, **params)


class Resample(SITKTransform):
    """Resample image to given spacing and direction.

    Although resampling looks like trivial thing to do in SimpleITK, it is not.
    It is a problem because origin etc. need to be changed. Output image size
    is *not* same as input image size.


    Parameters
    ----------
    spacing: float or tuple
        Spacing of the output image. If float, spacing is same across all axes.
        If spacing in an axis is None, then that axis is not resampled.
    direction: list, optional
        Direction of the output image. By default, direction is not changed.
    """

    def __init__(self, spacing, direction=None):
        self.ndim = len(spacing)
        self.spacing = tuple(spacing)
        self.direction = direction

    def _get_params(self, img):
        if self.direction is not None:
            direction = [float(x) for x in np.array(self.direction).flatten()]
        else:
            direction = img.GetDirection()

        default_spacing = img.GetSpacing()
        spacing = [x if x is not None else default_spacing[i]
                   for i, x in enumerate(self.spacing)]

        new_affine_matrix = np.dot(np.resize(direction,
                                             (self.ndim, self.ndim)),
                                   np.diag(spacing))
        inv_affine_matrix = np.linalg.inv(new_affine_matrix)

        # Get corner points of current image
        corners = [img.TransformIndexToPhysicalPoint(point)
                   for point in itertools.product(*[(0, x)
                                                    for x in img.GetSize()])]
        corners = np.transpose(corners)
        # Find new corner indices after transformation
        new_corners_idx = np.dot(inv_affine_matrix, corners)

        # Get the minimum across each axis.
        # This will be our new origin in physical space
        new_origin_idx = new_corners_idx.min(axis=1)
        new_origin = np.dot(new_affine_matrix, new_origin_idx)

        # Get new size from new corner indices
        new_size = np.max(new_corners_idx, axis=1) - new_origin_idx
        new_size = [int(x) for x in new_size]

        # Refernece image
        reference = sitk.Image(new_size, sitk.sitkFloat32)
        reference.SetDirection(direction)
        reference.SetSpacing(spacing)
        reference.SetOrigin(tuple(new_origin))

        return {'reference_img': reference}

    def _transform(self, img, is_label, reference_img):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_img)
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        return resampler.Execute(img)


class CenterCrop(SITKTransform):
    """Crop from center of the image.

    Raises error if output size is greater than the input size. Use
    MinimumPadding as required.

    Parameters
    ----------
    output_size: int or tuple
        Output size after crop.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def _get_params(self, img):
        self.ndim = img.GetDimension()
        output_size = self.output_size
        if isinstance(output_size, int):
            output_size = [output_size] * self.ndim

        size = img.GetSize()
        for i in range(self.ndim):
            assert output_size[i] <= size[i]

        lower_boundary = [(size[i] - output_size[i]) // 2
                          for i in range(self.ndim)]
        upper_boundary = [size[i] - lower_boundary[i] - output_size[i]
                          for i in range(self.ndim)]

        return {'lower_boundary': lower_boundary,
                'upper_boundary': upper_boundary}

    def _transform(self, img, is_label, lower_boundary, upper_boundary):
        return sitk.Crop(img, lower_boundary, upper_boundary)


class RandomCrop(SITKTransform):
    """Randomly crop from image.

    Raises error if output size is greater than the input size. Use
    MinimumPadding as required.

    Parameters
    ----------
    output_size: int or tuple
        Output size after crop.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def _get_params(self, img):
        self.ndim = img.GetDimension()
        output_size = self.output_size
        if isinstance(output_size, int):
            output_size = [output_size] * self.ndim

        size = img.GetSize()
        for i in range(self.ndim):
            assert output_size[i] <= size[i]

        lower_boundary = [(size[i] - output_size[i]) // 2
                          for i in range(self.ndim)]
        upper_boundary = [size[i] - lower_boundary[i] - output_size[i]
                          for i in range(self.ndim)]

        return {'lower_boundary': lower_boundary,
                'upper_boundary': upper_boundary}

    def _transform(self, img, is_label, lower_boundary, upper_boundary):
        return sitk.Crop(img, lower_boundary, upper_boundary)


class Padding(SITKTransform):
    """Pad an image.

    Parameters
    ----------
    pad_width:
        Pad width like in np.pad.
    cval: float, optional
        Constant value to be padded with.
    """

    def __init__(self, pad_width, cval=0):
        if isinstance(pad_width, int):
            self.pad_width = ((pad_width, pad_width),)
        else:
            self.pad_width = pad_width

        self.cval = cval

    def _get_params(self, img):
        ndim = img.GetDimension()
        pad_width = np.array(self.pad_width)

        if pad_width.shape[0] == ndim:
            assert pad_width.shape == (ndim, 2)
        elif pad_width.shape in [(1, 2), (2,)]:
            pad_width = np.tile(pad_width, [ndim, 1])
        else:
            raise TypeError('Unsupported pad width')

        return {'upper_padding': list(map(int, pad_width[:, 0])),
                'lower_padding': list(map(int, pad_width[:, 1]))}

    def _transform(self, img, is_label, upper_padding, lower_padding):
        return sitk.ConstantPad(img, upper_padding, lower_padding, self.cval)


class MinimumPadding(SITKTransform):
    """Pad an image so that it is at least of a minumum size.

    Parameters
    ----------
    min_size: tuple
        Minimum size of the image.
    """

    def __init__(self, min_size, cval=0):
        self.min_size = min_size
        self.cval = cval

    def _get_params(self, img):
        img_size = img.GetSize()
        ndim = img.GetDimension()
        extra_padding = [max(0, self.min_size[i] - img_size[i])
                         for i in range(ndim)]

        upper_padding = [int(extra_padding[i] // 2) for i in range(ndim)]
        lower_padding = [int(extra_padding[i] - upper_padding[i])
                         for i in range(ndim)]

        return {'upper_padding': upper_padding, 'lower_padding': lower_padding}

    def _transform(self, img, is_label, upper_padding, lower_padding):
        return sitk.ConstantPad(img, upper_padding, lower_padding, self.cval)


class RandomAffineTransform(SITKTransform):
    """Random affine transform.

    One transform for all your augmentation needs. Output size is same as input
    size.

    Parameters
    ----------
    rotation: float or list
        Random rotation in degrees. Angle is sampled in [-rotation, rotation].
        A list can be passed to specify rotation along each axis.
    shear: float or list.
        Shear coefficient in [0, 1].
    scale: float or list
        Random scale is sampled from [1 - scale, 1 + scale].
    reflect: float or list
        Probability to reflect along a axis.
    swap: float or list
        Probability to swap two axes.
    """

    def __init__(self, rotation=0, shear=0, scale=0, reflect=0, swap=0):
        self.rotation = rotation
        self.shear = shear
        self.scale = scale
        self.reflect = reflect
        self.swap = swap

    def _rotate(self, affine, axis):
        rotation = (self.rotation if isinstance(self.rotation, Number)
                    else self.rotation[axis])
        if rotation == 0:
            return affine

        if self.ndim == 3:
            axes = [0, 1, 2]
            axes.pop(axis)
        else:
            axes = [0, 1]

        rotation_rad = np.pi / 180 * rotation
        return affine.Rotate(*axes, np.random.uniform(-1, 1) * rotation_rad)

    def _shear(self, affine, axis):
        shear = (self.shear if isinstance(self.shear, Number)
                 else self.shear[axis])
        if shear == 0:
            return affine

        if self.ndim == 3:
            axes = [0, 1, 2]
            axes.pop(axis)
        else:
            axes = [0, 1]

        return affine.Shear(*axes, np.random.uniform(-1, 1) * shear)

    def _scale(self, affine, axis):
        scale = (self.scale if isinstance(self.scale, Number)
                 else self.scale[axis])
        if scale == 0:
            return affine

        scale_axes = [1] * self.ndim
        scale_axes[axis] = np.random.uniform(1 - scale, 1 + scale)
        return affine.Scale(scale_axes)

    def _reflect(self, affine, axis):
        reflect = (self.reflect if isinstance(self.reflect, Number)
                   else self.reflect[axis])
        if np.random.uniform() > reflect:
            return affine
        else:
            affine_matrix = affine.GetMatrix()
            affine_matrix = np.array(affine_matrix).reshape(self.ndim,
                                                            self.ndim)
            diag = np.eye(self.ndim)
            diag[axis, axis] = -1

            affine_matrix = np.dot(affine_matrix, diag)

        return affine.SetMatrix(list(affine_matrix.flatten()))

    def _swap(self, affine, axis):
        swap = (self.swap if isinstance(self.swap, Number)
                else self.swap[axis])
        if np.random.uniform() > swap:
            return affine
        else:
            if self.ndim == 3:
                axes = [0, 1, 2]
                axes.pop(axis)
            else:
                axes = [0, 1]

            angle = -np.pi / 2 if np.random.uniform() < 0.5 else np.pi / 2

            return affine.Rotate(*axes, angle)

    def _random_order(self, affine):
        all_transforms = [self._rotate, self._shear, self._scale,
                          self._reflect, self._swap]
        all_axes_transforms = [(x, y) for x in all_transforms
                               for y in range(self.ndim)]
        np.random.shuffle(all_axes_transforms)

        for transform, axis in all_axes_transforms:
            transform(affine, axis)

    def _get_params(self, img):
        self.ndim = img.GetDimension()
        size = img.GetSize()
        img_center_idx = [x / 2 for x in size]
        img_center = img.TransformContinuousIndexToPhysicalPoint(img_center_idx)

        affine = sitk.AffineTransform(self.ndim)
        affine.SetCenter(img_center)
        # Apply desired transforms in random order
        self._random_order(affine)

        return {'affine': affine}

    def _transform(self, img, is_label, affine):
        if is_label:
            return sitk.Resample(img, affine, sitk.sitkNearestNeighbor)
        else:
            return sitk.Resample(img, affine)


class RandomIntensityJitter(SITKTransform):
    """Random jitters in brightness and contrast.

    Parameters
    ----------
    brightness: float
        Intensity of brightness jitter. float in [0, 1].
    contrast: float
        Intensity of contrast jitter. float in [0, 1].
    """
    def __init__(self, brightness=0, contrast=0):
        self.brightness = brightness
        self.contrast = contrast

    def _blend(self, img1, img2, alpha):
        return img1 * alpha + img2 * (1 - alpha)

    def _brightness(self, img, var):
        return self._blend(img, 0 * img, var)

    def _contrast(self, img, var):
        statFilter = sitk.StatisticsImageFilter()
        statFilter.Execute(img)
        mean = statFilter.GetMean()
        return self._blend(img, 0 * img + mean, var)

    def _get_params(self, img):
        vars = [1 + self.brightness * np.random.uniform(-1, 1),
                1 + self.contrast * np.random.uniform(-1, 1)]

        return {'vars': vars, 'order': np.random.permutation(2)}

    def _transform(self, img, is_label, vars, order):
        if is_label:
            return img
        else:
            tsfrms = [self._brightness, self._contrast]
            for i in order:
                img = tsfrms[i](img, vars[i])

            return img


class Clip(SITKTransform):
    """Clip an image by a low and high value.

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
            return sitk.IntensityWindowing(img, self.inp_low, self.inp_high,
                                           self.out_low, self.out_high)


class Normalize(SITKTransform):
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
                arr = sitk.GetArrayFromImage(img)
                low = np.percentile(arr, p_low)
                hi = np.percentile(arr, p_hi)
                return (img - low) / (hi - low)
            else:
                mean, std = self.norm_params
                return (img - mean) / std


class ToTensor(SITKTransform):
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
        img = sitk.GetArrayFromImage(img)
        img = np.ascontiguousarray(img)

        if not is_label:
            # put it from HWC to CHW format
            if img.ndim == 4:
                img = np.rollaxis(img, 3, 0)
            elif img.ndim == 3:
                img = img.reshape((1,) + img.shape)

        img = self.torch.from_numpy(img)

        dtype = self.dtype_label if is_label else self.dtype
        if dtype is not None:
            return img.type(dtype)
        else:
            return img
