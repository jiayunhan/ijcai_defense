import numpy as np
import os
from PIL import Image
import torch
from utils.models import resnet101_ori, resnet50_ori, NASlarge_ori
from utils.rngs import nprng
from utils.image_transpose import batch_image, vote_label
import pdb

def load_resize_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    image_height = batch_shape[1]
    image_width = batch_shape[2]
    for filename in os.listdir(input_dir):
        if(filename.endswith(".png")):
            filepath = os.path.join(input_dir, filename)
            with Image.open(filepath) as img:
                img = img.resize((image_height // 2, image_width // 2), resample=3)
                img = img.resize((image_height, image_width), resample=3)
                image = np.asarray(img, dtype=np.float) / 255.
                # image = (image / 255.0) * 2.0 - 1.0
        images[idx, :, :, :] = image
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images.astype(np.float32)
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images.astype(np.float32)

def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    image_height = batch_shape[1]
    image_width = batch_shape[2]
    for filename in os.listdir(input_dir):
        if(filename.endswith(".png")):
            filepath = os.path.join(input_dir, filename)
            with Image.open(filepath) as img:
                img = img.resize((image_height, image_width))
                image = np.asarray(img, dtype=np.float) / 255.
                # image = (image / 255.0) * 2.0 - 1.0
        images[idx, :, :, :] = image
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images.astype(np.float32)
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images.astype(np.float32)

def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with open(os.path.join(output_dir, filename), 'w') as f:
            # img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            img = (images[i, :, :, :] * 255.).astype(np.uint8)
            Image.fromarray(img).resize((299, 299)).save(f, format='PNG')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_file')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--image_height', default=224)
    parser.add_argument('--image_width', default=224)
    args = parser.parse_args()
    batch_shape = [args.batch_size, args.image_height, args.image_width, 3]

    net = resnet101_ori(n_channels=3, num_classes=110, fe_branch=True, isPretrain=False)
    net = torch.nn.DataParallel(net).cuda()
    net.eval()
    pre_weight_path = args.checkpoint_path
    checkpoint = torch.load(pre_weight_path)
    net.load_state_dict(checkpoint['state_dict'])
    asdasd = 0       
    with open(args.output_file, 'w') as out_file:
        for filenames, images in load_images(args.input_dir, batch_shape):
            resize_factors = np.linspace(0.5, 1, num=6)[:-1]
            noise_epsilon = np.linspace(0, 0.05, num=6)[1:]
            images_var = torch.from_numpy(np.transpose(images, (0, 3, 1, 2))).cuda()
            outputs, _ = net(images_var)
            labels_ori = np.argmax(outputs.data.cpu().numpy(), axis=1)
            for filename, image in zip(filenames, images):
                
                image_ori = np.transpose(image, (2, 0, 1))
                image_ori_var = torch.from_numpy(np.expand_dims(image_ori, axis=0)).float().cuda()
                output, _ = net(image_ori_var)
                label_ori = np.argmax(output.data.cpu().numpy(), axis=1)
                
                images_batch = batch_image(image, (resize_factors, noise_epsilon))
                '''
                for adversarial in images_batch:
                    adversarial = (adversarial * 255).astype(np.uint8)
                    from PIL import Image
                    img = Image.fromarray(adversarial).save('../temp/test_out_' + str(asdasd) + '.png')
                    asdasd += 1
                '''
                images_batch_var = torch.from_numpy(np.transpose(images_batch, (0, 3, 1, 2))).float().cuda()
                outputs_batch, _ = net(images_batch_var)
                labels_batch = np.argmax(outputs_batch.data.cpu().numpy(), axis=1)
                label_vote = vote_label(labels_batch, label_ori)

                out_file.write('{0},{1}\n'.format(filename, label_vote))

if __name__ == '__main__':
    main()