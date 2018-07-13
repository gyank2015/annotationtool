from __future__ import print_function, division
import SimpleITK as sitk  
from skimage import io, util, color
from image_transforms.transforms import *
import pytest
import numpy as np
import torch
import os

filename = os.path.join(os.path.dirname(__file__), 'test.jpg')
img = util.img_as_float(io.imread(filename))


all_data = [
    img,
    {'input': img, 'target': np.random.randint(0, 4, size=img.shape),
     'target2': np.random.randint(0, 10, size=img.shape[:2]),
     2: 4,
     'lol': img, 'target_label': 3},
    {'input': color.rgb2gray(img),
     'target': np.random.randint(0, 4, size=img.shape)},
]

h, w = img.shape[:2]

all_transforms = {
    Rescale(0.5): (h / 2, w / 2),
    Rescale((0.5, 0.25)): (h / 2, w / 4),
    Rescale((0.5, 0.25), order=0): (h / 2, w / 4),

    Resize((100, 200)): (100, 200),
    Resize((100, 200), order=0): (100, 200),
    Resize(100): (100, w / h * 100) if h < w else (h / w * 100, w),

    CenterCrop((200, 250)): (200, 250),
    CenterCrop(200): (200, 200),

    RandomCrop((400, 200)): (400, 200),
    RandomCrop(400): (400, 400),

    RandomScale(200, 250): None,
    RandomSizedCrop((400, 300)): (400, 300),
    RandomSizedCrop((400, 300), order=0): (400, 300),

    # rotate
    RandomRotate(30): (h, w),
    RandomRotate([0, 45]): (h, w),
    RandomRotate([0, 45], order=0, preserve_range=True): (h, w),

    RandomHorizontalFlip(): (h, w),
    RandomVerticalFlip(): (h, w),

    # padding
    Padding(40): (h + 80, w + 80),
    Padding([(10, 20), (30, 20)]): (h + 30, w + 50),
    Padding([(10, 20)]): (h + 30, w + 30),

    MinimumPadding((800, 700)): (max(h, 800), max(w, 700)),
    MinimumPadding((40, 800)): (max(h, 40), max(w, 800)),
    MinimumPadding((40, 40)): (max(h, 40), max(w, 40)),

    Normalize((1, 99), 'percentile'): (h, w),
    Normalize((50, 100), 'meanstd'): (h, w),

    RandomIntensityJitter(0.3, 0.4, 0.2): (h, w),

    ElasticTransform(100, 10): (h, w)
}

all_parameters = [(transform, data, expected_shape)
                  for transform, expected_shape in all_transforms.items()
                  for data in all_data]


@pytest.mark.parametrize("transform,data,expected_shape", all_parameters)
def test_transform(transform, data, expected_shape):
    h, w = transform._argcheck(data)
    out = transform(data)

    # check if outputs are of same size
    out_shape = transform._argcheck(out)

    # check the expected size
    if expected_shape is not None:
        assert out_shape == expected_shape, \
            'Not expected shape. Expected: {}, got: {}'.format(
                out_shape, expected_shape)

    # check if input is modified
    assert transform._argcheck(data) == (h, w), 'transform modified input?'

    # if out should be dict as ndarry if data is so.
    assert isinstance(out, type(data))

    if isinstance(out, dict):
        assert len(out) == len(data)
        for k in data.keys():
            if isinstance(data[k], np.ndarray):
                if 'target' in k:
                    # check if images with target had same unique
                    assert (np.unique(out[k]) == np.unique(data[k])).all()
            else:
                assert data[k] == out[k]

dtypes = ['torch.DoubleTensor', 'torch.FloatTensor',
          torch.ByteTensor, torch.CharTensor, torch.DoubleTensor, None]


@pytest.mark.parametrize("data, dtype",
                         [(x, y) for x in all_data for y in dtypes])
def test_to_tensor(data, dtype):
    to_tensor = ToTensor(dtype)
    out = to_tensor(data)

    # check if type matches
    if isinstance(out, dict):
        assert len(out) == len(data)
        imgs = {k: out[k] for k in data if isinstance(data[k], np.ndarray)}
    else:
        imgs = {False: out}

    for k, img in imgs.items():
        if isinstance(k, str) and 'target' in k:
            continue

        # check if dtype matches
        if dtype is not None:
            if isinstance(dtype, str):
                assert img.type() == dtype
            else:
                assert isinstance(img, dtype)

        # check if hwc to chw conversion is made
        size = img.size()
        in_img = data[k] if k else data
        in_size = in_img.shape
        if len(in_size) == 3:
            assert size == (in_size[2], in_size[0], in_size[1])
        else:
            assert size == (1, in_size[0], in_size[1])


def test_elastic():
    # visual test
    from skimage.color import rgb2gray
    elastic = ElasticTransform(100, 10)
    data = {'input': img, 'target': (rgb2gray(img) * 5).astype('int')}

    out = elastic(data)
    out_img = np.clip(out['input'], 0, 1)

    io.imsave('in_target.png', data['target'] / 5)
    io.imsave('out.png', out_img)
    io.imsave('out_target.png', out['target'] / 5)


def manual_test():
    for data in all_data:
        print(type(data))
        for transform, expected_size in all_transforms.items():
            try:
                test_transform(transform, data, expected_size)
            except:
                print(transform)
                raise
        print('passed')

if __name__ == '__main__':
    manual_test()
