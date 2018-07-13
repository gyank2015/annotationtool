Transforms (3d)
===============

.. currentmodule:: image_transforms.transforms_3d

Transforms described in this document are meant to be applied on 3d images. 
Each transform can accept an image or a dict of images. Dict of images is
especially useful when you are doing segmentation.

Let's see an example ::
    
    import SimpleITK as sitk
    from torchvision.transforms import Compose
    from image_transforms import transforms_3d as tsfms 

    img = sitk.ReadImage('test.nii.gz', sitk.sitkFloat32)
    label_img = sitk.Cast(5 * img / 600., sitk.sitkInt32)

    img = io.imread('cut.jpg')
    transform = Compose([tsfms.RandomAffineTransform(rotation=30),
                         tsfms.RandomCrop([256, 256, 20])]) 

    # option1: Pass img directly 
    out1 = transform(img)
    assert out1.GetSize() == (256, 256, 20)

    # option2: pass a dict of images
    data = {'input': img, 'target': label_img,
            'target_label': 3,
            2: 4, 'lol': img}

    out2 = transform(data2)
    assert out2['input'].shape == (400, 400, 3)
    assert (out2['target'].max() == 3) and (out2['label'] == 30)
    

Following is assumed when you pass a dict:

    1. All ndarrays in the dict are considered as images and should be of the same size.
    2. dict can have values which are not arrays, like ``label`` in the above example
    3. If key for a image in dict has string ``target`` in it somewhere, it is
       considered as a target segmentation map. For example in rotation, nearest
       neighbor interpolation is applied to target.


.. note::
    ``transforms_3d`` uses SimpleITK size notation wherever required.
    Remember that this is reverse of numpy/torch notation. For example ::

        sitk_size = [3, 4, 5]
        img = sitk.Image(sitk_size, sitk.sitkFloat32)
        assert sitk.GetArrayFromImage(img).shape == (5, 4, 3)



.. contents:: Available transforms
    :local:

All transforms are classes with a ``__call__``. 
SimpleITK based transforms inherit the following base class:

Structural transforms
---------------------

Resample
~~~~~~~~
.. autoclass:: Resample

CenterCrop
~~~~~~~~~~
.. autoclass:: CenterCrop

RandomCrop
~~~~~~~~~~
.. autoclass:: RandomCrop

Padding
~~~~~~~
.. autoclass:: Padding

MinimumPadding
~~~~~~~~~~~~~~
.. autoclass:: MinimumPadding

RandomAffineTransform
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandomAffineTransform


Intensity transforms
--------------------

Clip
~~~~
.. autoclass:: Clip

Normalize
~~~~~~~~~
.. autoclass:: Normalize

RandomIntensityJitter
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandomIntensityJitter

Others
------

ToTensor
~~~~~~~~
.. autoclass:: ToTensor

BaseClass
~~~~~~~~~

.. autoclass:: SITKTransform
    :members: _get_params, _transform
    :private-members: