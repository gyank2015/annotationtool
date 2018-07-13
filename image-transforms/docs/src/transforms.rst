Transforms (2d)
===============

.. currentmodule:: image_transforms.transforms

Transforms described in this document are meant to be applied on 2d images. 
Each transform can accept an image or a dict of images. Dict of images is
especially useful when you are doing segmentation.


Let's see an example ::
    
    from skimage import io
    from torchvision.transforms import Compose
    from image_transforms import transforms as tsfms 

    img = io.imread('cut.jpg')
    transform = Compose([tsfms.RandomRotate(30),
                         tsfms.RandomCrop(400)]) 

    # option1: Pass img directly 
    data1 = img
    out1 = transform(data2)
    assert out1.shape == (400, 400, 3)

    # option2: pass a dict of images
    data2 = {'input': img, 
             'target': np.random.randint(0, 4, size=img.shape[:2]),
             'label': 30}
    out2 = transform(data2)
    assert out2['input'].shape == (400, 400, 3)
    assert (out2['target'].max() == 3) and (out2['label'] == 30)
    

Following is assumed when you pass a dict:

    1. All ndarrays in the dict are considered as images and should be of the same size.
    2. dict can have values which are not arrays, like ``label`` in the above example
    3. If key for a image in dict has string ``target`` in it somewhere, it is
       considered as a target segmentation map. For example in rotation, nearest
       neighbor interpolation is applied to target.


.. contents:: Available transforms
    :local:

All transforms are classes with a ``__call__``. 
numpy based transforms inherit the a base class ``NDTransform``.

Structural transforms
---------------------

Rescale
~~~~~~~
.. autoclass:: Rescale

Resize
~~~~~~
.. autoclass:: Resize

CenterCrop
~~~~~~~~~~
.. autoclass:: CenterCrop

RandomCrop
~~~~~~~~~~
.. autoclass:: RandomCrop

RandomScale
~~~~~~~~~~~
.. autoclass:: RandomScale

RandomSizedCrop
~~~~~~~~~~~~~~~
.. autoclass:: RandomSizedCrop

RandomRotate
~~~~~~~~~~~~
.. autoclass:: RandomRotate

RandomHorizontalFlip
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandomHorizontalFlip

RandomVerticalFlip
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandomVerticalFlip

Padding
~~~~~~~
.. autoclass:: Padding

ElasticTransform
~~~~~~~~~~~~~~~~
.. autoclass:: ElasticTransform

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
.. autoclass:: NDTransform
    :members: _get_params, _transform
    :private-members:
