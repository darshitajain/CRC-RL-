import numpy as np
from skimage.util.shape import view_as_windows

# import kornia
import torch
import torch.nn.functional as F


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones
    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    return windows[np.arange(n), w1, h1]


def center_crop_image(image, output_size):
    h, w = image.shape[-2], image.shape[-1]
    #h, w = 100, 100
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top : top + new_h, left : left + new_w]
    return image

def center_crop_image_batch(image, output_size):
    h, w = image.shape[2], image.shape[3]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, :, top : top + new_h, left : left + new_w]
    return image


def center_crop_image_batch_nw(imgs, out_h, out_w=None):
    '''
    args: 
        image: input image of shape (c, h, w) or (b, c, h, w)
        out_h: output height
        out_w: output width
    returns:
        cropped image of shape (-1, h, w, c)
    '''
    img_array = np.asarray(imgs)
    
    if len(img_array.shape) == 3:   # (c, h, w)
        img_h = img_array.shape[1]
        img_w = img_array.shape[2]
    elif len(img_array.shape) == 4:     # (B, C, H, W)
        img_h = img_array.shape[2]
        img_w = img_array.shape[3]

    if out_w is None:
        out_w = out_h
        
    top = (img_h - out_h) // 2
    left = (img_w - out_w) // 2
    cropped_images = img_array[..., top:top+out_h, left:left+out_w, :]
    return cropped_images

def random_conv(imgs, out_size):
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    imgs = torch.from_numpy(imgs).float()
    n, c, h, w = imgs.shape
    print(h, w)
    for i in range(n):
        weights = torch.randn(3, 3, 3, 3).to(imgs.device)
        temp_imgs = imgs[i : i + 1].reshape(-1, 3, h, w) / 255.0
        temp_imgs = F.pad(temp_imgs, pad=[1] * 4, mode="replicate")
        out = torch.sigmoid(F.conv2d(temp_imgs, weights)) * 255.0
        total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
    return np.array(total_out.reshape(n, c, h, w))


# Need to fix the kornia import error. This was working fine till 18 April 2022
'''
def random_shift(imgs, out_size, pad=4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = imgs.shape
    # print('h,w', h, w)
    # print(type(imgs))
    imgs = torch.from_numpy(imgs).float()
    imgs = F.pad(imgs, (pad, pad, pad, pad), mode="replicate")
    _, _, h, w = imgs.shape
    # print('after padding',h,w)
    # print('inside random_shift', imgs.shape, type(imgs))
    crop = h - out_size
    return np.array(kornia.augmentation.RandomCrop((h - crop, w - crop))(imgs))
'''

if __name__ == "__main__":
    out_size = 84
    crop_size = 100 - 84
    img = np.ones([128, 9, 100, 100])
    out = random_conv(img, out_size)
    print(out.shape)
    print(type(out))
