import numpy as np 
from PIL import Image
from utils.rngs import nprng

import pdb

def batch_image(image, params):
    resize_factors, noise_epsilons = params
    images_out = []
    for resize_factor in resize_factors:
        for noise_epsilon in noise_epsilons:
            noise = nprng.uniform(-1 * noise_epsilon, noise_epsilon, size=image.shape)
            image_noise = image + noise_epsilon * noise
            image_noise = np.clip(image_noise, 0, 1)
            img_np = (image_noise * 255.).astype(np.uint8)
            img = Image.fromarray(img_np)
            img = img.resize((int(image.shape[0] * resize_factor), int(image.shape[1] * resize_factor)), resample=3)
            img = img.resize((image.shape[0], image.shape[1]), resample=3)
            image_out = np.array(img).astype(float) / 255. 
            images_out.append(image_out)
    return np.array(images_out)

def vote_label(labels_batch, label_ori):
    label_ori = label_ori[0]
    counts = np.bincount(labels_batch)
    counts_sort = np.copy(counts)
    counts_sort.sort()
    if counts_sort.shape[0] < 2 or counts_sort[-1] == counts_sort[-2]:
        return label_ori
    else:
        return np.argmax(counts)
