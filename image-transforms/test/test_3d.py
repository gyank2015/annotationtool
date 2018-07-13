import SimpleITK as sitk
from image_transforms.transforms_3d import *
import numpy as np
import pytest
import os

filename = os.path.join(os.path.dirname(__file__), 'test.nii.gz')

img = sitk.ReadImage(filename, sitk.sitkFloat32)
label_img = sitk.Cast(5 * img / 600., sitk.sitkInt32)
h, w, d = img.GetSize()

print(h,w,d)
all_data = [
    img,
    {'input': img, 'target': label_img,
     2: 4,
     'lol': img, 'target_label': 3},
]

all_transforms = {
    Resample([0.5, 0.5, 0.5]): (2 * h, 2 * w, 2 * d),
    Resample([1, 1, 1], np.eye(3)): None,
    Resample([1, 1, None], np.eye(3)): None,

    CenterCrop(150): (150, 150, 150),
    CenterCrop([200, 150, 100]): (200, 150, 100),

    RandomCrop(145): (145, 145, 145),
    RandomCrop([240, 89, 100]): (240, 89, 100),

    Padding(10): (h + 20, w + 20, d + 20),
    Padding([[10, 40]]): (h + 50, w + 50, d + 50),
    Padding([12, 32]): (h + 44, w + 44, d + 44),
    Padding([[5, 7], [3, 4], [4, 7]]): (h + 12, w + 7, d + 11),

    MinimumPadding((200, 300, 400)): (max(h, 200), max(w, 300), max(d, 400)),
    MinimumPadding((400, 100, 40)): (max(h, 400), max(w, 100), max(d, 40)),

    RandomAffineTransform(30, 0.1, 0.1, 0.9, 0.9): (h, w, d),
    RandomAffineTransform(0, [0.1, 0, 0.1], 0.1, 0.1, [0, 0.1, 0.1]): (h, w, d),
    RandomAffineTransform(0, 0.1, [0.1, 0, 0.1], [0, 0.1, 0.1], 0.2): (h, w, d),

    Clip(0, 100): (h, w, d),
    Clip(0, 100, 0, 1): (h, w, d),

    Normalize((1, 99), 'percentile'): (h, w, d),
    Normalize((50, 100), 'meanstd'): (h, w, d)
}

all_parameters = [(transform, data, expected_size)
                  for transform, expected_size in all_transforms.items()
                  for data in all_data]


@pytest.mark.parametrize("transform,data,expected_size", all_parameters)
def test_transform(transform, data, expected_size, save=None):
    in_size = transform._argcheck(data).GetSize()
    out = transform(data)

    # check if outputs are of same size
    out_size = transform._argcheck(out).GetSize()

    # check the expected (size
    if expected_size is not None:
        assert out_size == expected_size, \
            'Not expected shape. Expected: {}, got: {}'.format(
                out_size, expected_size)

    # check if input is modified
    assert transform._argcheck(data).GetSize() == in_size, \
        'transform modified input?'

    # if out should be dict as ndarry if data is so.
    assert isinstance(out, type(data))

    if isinstance(out, dict):
        assert len(out) == len(data)
        for k in data.keys():
            if isinstance(data[k], sitk.Image):
                if 'target' in k:
                    # check if images with target had same unique
                    in_img = sitk.GetArrayFromImage(data[k])
                    out_img = sitk.GetArrayFromImage(out[k])
                    try:
                        assert (np.unique(in_img) == np.unique(out_img)).all()
                    except AttributeError:
                        print('unique in: {} out: {}'.format(np.unique(in_img),
                                                             np.unique(out_img)))
                        raise

                    continue
            else:
                assert data[k] == out[k]

    if save is not None:
        if save is True:
            sitk.WriteImage(out, 'out.nii.gz')
        else:
            sitk.WriteImage(out[save], 'out.nii.gz')


def manual_test():
    for data in all_data:
        print(type(data))
        for transform, expected_size in all_transforms.items():
            try:
                test_transform(transform, data, expected_size)
            except Exception as e:
                print(transform)
                raise e
        print('passed')

if __name__ == '__main__':
    manual_test()
