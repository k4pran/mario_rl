import torch
import matplotlib.pyplot as plt


def display_img(img):
    img_to_show = img
    if isinstance(img, torch.Tensor) and img.ndim > 2:
        img_to_show = img.clone()
        img_to_show = img_to_show.squeeze()
    plt.imshow(img_to_show, cmap='gray')
    plt.show()
