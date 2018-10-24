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
    self.zero_grad()
    x.grad = None
    return gradient



def to_tensor(image,device):
    return torch.tensor(image.transpose([2,0,1]),dtype=torch.float32,device=device).unsqueeze(0)



def to_numpy(image_tensor):
    return image_tensor.squeeze().cpu().numpy().transpose([1,2,0])



def plot_grad(grad):
    g_min = grad.min()
    g_max = grad.max()
    grad_normalized = (grad-g_min)/(g_max-g_min)
    # Plot the normalized gradient.
    plt.imshow(grad_normalized, interpolation='bilinear')
    plt.show()



def plot_image(image):
    if not isinstance(image,np.ndarray):
        plot_image(to_numpy(image))
        return
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)
    # Convert pixels to bytes.
    image = image.astype(np.uint8)
    plt.imshow(image)
    plt.show()


def make_grid(image_tensor, tile_size, overlap):
    """
    Make grad grid and image grid with extended tiles.
    Coordinate system in the case of the grad_grid corresponds to image_tensor
    but in the case of the image_grid it corresponds to image_tensor with padding=overlap.
    """
    grad_grid = []
    image_grid = []
    n, c, x_max, y_max = image_tensor.shape
    num_tiles_x, num_tiles_y = int(round(x_max / tile_size)), int(round(y_max / tile_size))
    # How many times can we repeat a tile of the desired size.
    # Ensure that there is at least 1 tile.
    num_tiles_x, num_tiles_y = max(1, num_tiles_x), max(1,num_tiles_y)
    # Boundary coordinates
    xs = np.linspace(0,x_max,num_tiles_x+1,dtype=np.int16)
    ys = np.linspace(0,y_max,num_tiles_y+1,dtype=np.int16)
    # Random shifts for the tiles on x,y-axes.
    # The random value is between -1/4 and 1/4 of the tile-size.
    # This is so the border-tiles are roughly at least 1/4 of the tile-size,
    # otherwise the tiles may be too small which creates noisy gradients.
    x_shift = random.uniform(-0.75, -0.25)
    y_shift = random.uniform(-0.75, -0.25)
    xs = np.append(xs,[2*xs[-1]-xs[-2]]) + int(x_shift*(xs[1]-xs[0]))
    ys = np.append(ys,[2*ys[-1]-ys[-2]]) + int(y_shift*(ys[1]-ys[0]))
    # Crop to the image size
    xs[0], ys[0], xs[-1], ys[-1]  =  0, 0, x_max, y_max
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            grad_grid.append((xs[i],xs[i+1],ys[j],ys[j+1]))
    # Add padding
    xs += overlap
    ys += overlap
    for i in range(len(xs)-1):
        for j in range(len(ys)-1):
            image_grid.append((xs[i]-overlap,xs[i+1]+overlap,ys[j]-overlap,ys[j+1]+overlap))
    return image_grid, grad_grid



def tiled_gradient(model,layer, image_tensor, overlap=10, tile_size=400):
    # Allocate an array for the gradient of the entire image.
    grad = torch.zeros_like(image_tensor)
    # Make grid
    # To avoid visible seams problem which may be due to the smaller image gradients at the edges
    # we will feed extended tiles to the model input
    image_grid, grad_grid = make_grid(image_tensor, tile_size, overlap)
    #Also we should add padding to image. For smooth result we will use 'reflect' mode.
    img = torch.nn.functional.pad(image_tensor, [overlap]*4, 'reflect')

    for i in range(len(image_grid)):
        x_left,x_right,y_top,y_bot = image_grid[i]
        tile = img[:,:,x_left:x_right,y_top:y_bot]
        tile.requires_grad = True
        g = model.get_gradient(layer,tile)
        if overlap != 0:
            g = g[:,:,overlap:-overlap,overlap:-overlap]
        g /= (g.mean()+ 1e-8)
        x_left,x_right,y_top,y_bot = grad_grid[i]
        grad[:,:,x_left:x_right,y_top:y_bot] = g
    return grad/(grad.std()+1e-8)



def optimize_image(model,layer, image,
                   num_iterations=10, step_size=3.0, overlap=10, tile_size=400,
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
    image_tensor = to_tensor(image, model.device)
    print("Image before:")
    plot_image(image_tensor)
    print("Processing image: ", end="")
    for i in range(num_iterations):
        # Calculate the value of the gradient.
        # This tells us how to change the image so as to
        # maximize the mean of the given layer-tensor.
        grad = to_numpy(tiled_gradient(model,layer, image_tensor, overlap, tile_size))
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

        image_tensor += to_tensor(grad,model.device) * step_size_scaled

        if show_gradient:
            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size_scaled))
            # Plot the gradient.
            plot_grad(grad)
        else:
            # Otherwise show a little progress-indicator.
            print(". ", end="")
    print("\nImage after:")
    plot_image(image_tensor)
    return to_numpy(image_tensor)
