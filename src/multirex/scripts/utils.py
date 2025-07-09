import numpy as np

def gammaCorrect(img, dim=-1):
    if dim == -1:
        dim = len(img.shape) - 1
    assert (img.shape[dim] == 3)
    gamma, black, color_scale = 2.0, 3.0 / 255.0, [1.4, 1.1, 1.6]
    scale = np.array(color_scale).reshape([3 if i == dim else 1 for i in range(img.ndim)])
    img = img * scale / 1.1
    correct_img = np.clip(
        (((1.0 / (1 - black)) * 0.95 * np.clip(img - black, 0, 2)) ** (1.0 / gamma)) - 15.0 / 255.0, 0, 2, )

    return correct_img