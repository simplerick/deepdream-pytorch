from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.misc import imresize
import numpy as np
import torch
import math
import random
import PIL.Image
from scipy.ndimage.filters import gaussian_filter



def get_gradient(self, layer, x):
    """
    Get gradient of some layer mean by image.
    Important: make sure that list of children of your model gives the desired computational graph!
    """
    x.grad = None
    activation = x
    for l in self.children():
        activation = l(activation)
        if l == layer:
            break
    loss = activation.mean()
    loss.backward()
    gradient = x.grad.data
    x.grad = None
    return gradient


def to_tensor(tile, pad, device):
    #Convert to a torch tensor that can be fed to the input model
    tile = torch.tensor(tile.transpose([2,0,1]),dtype=torch.float32,device=device)
    tile = torch.nn.functional.pad(tile.unsqueeze(0), (pad, pad, pad, pad), 'reflect')
    tile.requires_grad = True
    return tile


def to_numpy(grad,pad):
    #Convert torch tensor of gradient back to a ndarray
    return (np.squeeze(grad.cpu().numpy())[:,pad:-pad,pad:-pad]).transpose([1,2,0])


def plot_image(image):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)
    # Convert pixels to bytes.
    image = image.astype(np.uint8)
    plt.imshow(image)
    plt.show()


def plot_grad(grad):
    g_min = grad.min()
    g_max = grad.max()
    grad_normalized = (grad-g_min)/(g_max-g_min)
    # Plot the normalized gradient.
    plt.imshow(grad_normalized, interpolation='bilinear')
    plt.show()


def resize_image(image, size=None, factor=None):
    # If a rescaling-factor is provided then use it.
    if factor is not None:
        # Scale the numpy array's shape for height and width.
        size = np.array(image.shape[0:2]) * factor
        # The size is floating-point because it was scaled.
        # PIL requires the size to be integers.
        size = size.astype(int)
    else:
        # Ensure the size has length 2.
        size = size[0:2]
    # The height and width is reversed in numpy vs. PIL.
    size = tuple(reversed(size))
    # Ensure the pixel-values are between 0 and 255.
    img = np.clip(image, 0.0, 255.0)
    # Convert the pixels to 8-bit bytes.
    img = img.astype(np.uint8)
    # Create PIL-object from numpy array.
    img = PIL.Image.fromarray(img)
    # Resize the image.
    img_resized = img.resize(size, PIL.Image.LANCZOS)
    # Convert 8-bit pixel values back to floating-point.
    img_resized = np.float32(img_resized)
    return img_resized


def get_tile_size(num_pixels, tile_size=500):
    """
    num_pixels is the number of pixels in a dimension of the image.
    tile_size is the desired tile-size.
    """
    # How many times can we repeat a tile of the desired size.
    num_tiles = int(round(num_pixels / tile_size))
    # Ensure that there is at least 1 tile.
    num_tiles = max(1, num_tiles)
    # The actual tile-size.
    actual_tile_size = math.ceil(num_pixels / num_tiles)
    return actual_tile_size


def tiled_gradient(model,layer, image, tile_size=400):
    pad = 10
    # Allocate an array for the gradient of the entire image.
    grad = np.zeros_like(image)
    # Number of pixels for the x- and y-axes.
    x_max, y_max, _ = image.shape
    # Tile-size for the x-axis.
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    # 1/4 of the tile-size.
    x_tile_size4 = x_tile_size // 4
    # Tile-size for the y-axis.
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    # 1/4 of the tile-size
    y_tile_size4 = y_tile_size // 4
    # Random start-position for the tiles on the x-axis.
    # The random value is between -3/4 and -1/4 of the tile-size.
    # This is so the border-tiles are at least 1/4 of the tile-size,
    # otherwise the tiles may be too small which creates noisy gradients.
    x_start = random.randint(-3*x_tile_size4, -x_tile_size4)
    while x_start < x_max:
        # End-position for the current tile.
        x_end = x_start + x_tile_size
        # Ensure the tile's start- and end-positions are valid.
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)
        # Random start-position for the tiles on the y-axis.
        # The random value is between -3/4 and -1/4 of the tile-size.
        y_start = random.randint(-3*y_tile_size4, -y_tile_size4)
        while y_start < y_max:
            # End-position for the current tile.
            y_end = y_start + y_tile_size
            # Ensure the tile's start- and end-positions are valid.
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)
            # Get the image-tile.
            img_tile = image[x_start_lim:x_end_lim,
                             y_start_lim:y_end_lim, :]
            # Create a tensor with the image-tile.
            img = to_tensor(img_tile,pad,model.device)
            # Use PyTorch to calculate the gradient-value.
            g = to_numpy(model.get_gradient(layer,img),pad)
            # Normalize the gradient for the tile. This is
            # necessary because the tiles may have very different
            # values. Normalizing gives a more coherent gradient.
            g /= (np.mean(g) + 1e-8)
            # Store the tile's gradient at the appropriate location.
            grad[x_start_lim:x_end_lim,
                 y_start_lim:y_end_lim, :] = g
            # Advance the start-position for the y-axis.
            y_start = y_end
        # Advance the start-position for the x-axis.
        x_start = x_end
    return grad


def optimize_image(model,layer, image,
                   num_iterations=10, step_size=3.0, tile_size=400,
                   show_gradient=False):
    """
    Use gradient ascent to optimize an image so it maximizes the
    mean value of the given layer_tensor.

    Parameters:
    model: Network model.
    layer: Reference to a tensor that will be maximized.
    image: Input image used as the starting point.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    show_gradient: Plot the gradient in each iteration.
    """
    # Copy the image so we don't overwrite the original image.
    img = image.copy()
    print("Image before:")
    plot_image(image)
    print("Processing image: ", end="")
    for i in range(num_iterations):
        # Calculate the value of the gradient.
        # This tells us how to change the image so as to
        # maximize the mean of the given layer-tensor.
        grad = tiled_gradient(model,layer, img)
        # Blur the gradient with different amounts and add
        # them together. The blur amount is also increased
        # during the optimization. This was found to give
        # nice, smooth images. You can try and change the formulas.
        # The blur-amount is called sigma (0=no blur, 1=low blur, etc.)
        # We could call gaussian_filter(grad, sigma=(sigma, sigma, 0.0))
        # which would not blur the colour-channel. This tends to
        # give psychadelic / pastel colours in the resulting images.
        # When the colour-channel is also blurred the colours of the
        # input image are mostly retained in the output image.
        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma*2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma*0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)
        # Scale the step-size according to the gradient-values.
        # This may not be necessary because the tiled-gradient
        # is already normalized.
        step_size_scaled = step_size / (np.std(grad) + 1e-8)
        # Update the image by following the gradient.
        img += grad * step_size_scaled
        plot_image(img)
        if show_gradient:
            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size_scaled))
            # Plot the gradient.
            plot_gradient(grad)
        else:
            # Otherwise show a little progress-indicator.
            print(". ", end="")
    print("\nImage after:")
    plot_image(img)
    return img
