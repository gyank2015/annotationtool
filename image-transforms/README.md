# Image Transforms

These are transforms based on scikit-image and SimpleITK for data preprocessing and augmentation.

### Why is this needed?

You ask why not use [`torchvision.transforms`](http://pytorch.org/docs/torchvision/transforms.html). Suppose you are working on a segmentation problem, your random crop transform have to crop the both input and target exactly the same way. This is not possible with `torchvision.transforms`.

### Great, how to use it?

Prerequisite:

```
pip install SimpleITK
```

Install using 

```
$ pip install git+https://bitbucket.org/aiinnovation/image-transforms.git
```

Alternatively,

```
$ git clone git@bitbucket.org:aiinnovation/image-transforms.git
$ cd image-transforms
$ python setup.py install
```
Here's how API works:

```python
from skimage import io, util
from image_transforms.transforms import RandomCrop

img = util.img_as_float(io.imread('cut.jpg'))
transform = RandomCrop(400)

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
```

For complete documentation, see [docs/](docs).

### Recommended way to use in your repo

I recommend you to install this as package as described above and use it like just another package.

### Note

Do not unnecessarily copy the code into your repo and create copies of this. If you want to change something raise an issue or create a PR to this repo.
This repo is fairly well tested. See `tests/`. Let's keep the data preprocessing code clean.