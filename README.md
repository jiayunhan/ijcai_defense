## Docker Image

**registry.cn-shanghai.aliyuncs.com/aliseccompetition/tensorflow:1.1.0-devel-gpu** 

- python2.7 
- cuda8.0
- tensorflow:1.1.0

---

## Data PreProcess


```python
def preprocessor(im, model_name):
    """
    :param im:  the raw image array [height, width, channels]
    :param model_name: the network model name 
    :return: 
    """
    if 'inception' in model_name.lower():
        image = imresize(im, [FLAGS.image_height, FLAGS.image_width]).astype(np.float)
        image = ( image / 255.0 ) * 2.0 - 1.0
        return  image
    if 'resnet' in model_name.lower() or 'vgg' in model_name.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        image = imresize(im, [FLAGS.image_height, FLAGS.image_width]).astype(np.float)
        image[:, :, 0] = image[:, :, 0] - _R_MEAN
        image[:, :, 1] = image[:, :, 1] - _G_MEAN
        image[:, :, 2] = image[:, :, 2] - _B_MEAN
        return image


def load_images(input_dir, batch_shape):
    """ Read images from input directory 
    
    :param input_dir:  input directory
    :param batch_shape: shape of batch array, i.e. [batch_size, height, width, 3]
    :return: yields:
            filenames: list file names of each image
            images: array of each images, images are all preprocessed
    """

    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with open(filepath) as f:
            raw_image = imread(f, mode='RGB')
            image = preprocessor(raw_image, FLAGS.model_name)
        images[idx, :, :, :] = image
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

```

