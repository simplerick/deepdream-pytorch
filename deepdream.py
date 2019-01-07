from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import random
import PIL.Image
from PIL.ImageFilter import GaussianBlur
from PIL import ImageChops
from scipy.ndimage.filters import gaussian_filter


def to_tensor(image,device):
    return transforms.ToTensor()(image).unsqueeze(0).to(device)

def to_pil(image_tensor):
    image = image_tensor.cpu().squeeze()
    return transforms.ToPILImage()(image)


def plot_grad(grad):
    # g_min = grad.min()
    # g_max = grad.max()
    # grad_normalized = (grad-g_min)/(g_max-g_min)
    grad_normalized = (grad - grad.mean())/grad.std()*0.4+0.5
    # Plot the normalized gradient.
    plt.imshow(grad_normalized, interpolation='bilinear')
    plt.show()


def resize_image(image, size=None):
    size = int(size[0]),int(size[1])
    # Resize the image.
    img_resized = image.resize(size, PIL.Image.LANCZOS)
    # Convert 8-bit pixel values back to floating-point.
    return img_resized


def make_grid(image_tensor, tile_size, overlap):
    """
    Make: 1) grad grid, 2) image grid with extended tiles.
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




class DeepDream():
    """
    Pytorch implementation of the DeepDream algorithm.
    Attrs: model,device,module,loss
    """
    def __init__(self,model=None,device=None):
        self.model = model
        self.device = device
        self.module = None
        self.loss = None


    # TODO: think about about early exit from the forward pass
    def get_gradient(self, x):
        """
        Get gradient of scalar function "loss" of some "module" output by image "x".
        If you get message about size mismatch it doesn't necessarily mean that grad computation failed
        It will be a problem only if size mismatch happens before target module.
        """
        # Create hook
        loss = self.loss
        def h(self,input,output):
            loss(output).backward()
        # Clear gradient
        x.grad = None
        error = None
        # Put handle
        handle = self.module.register_forward_hook(h)
        try:
            self.model(x)
        except RuntimeError as err:
            error = err
        if x.grad is None: #if x.grad wasn't computed by some reason
            raise error
        gradient = x.grad.detach()
        # Clear gradients and remove handle
        handle.remove()
        self.model.zero_grad()
        x.grad = None
        return gradient


    def tiled_gradient(self, image_tensor, overlap=10, tile_size=400):
        # Allocate an array for the gradient of the entire image.
        grad = torch.zeros_like(image_tensor)
        # Make grid
        # To avoid visible seams problem which may be due to the smaller image gradients at the edges
        # we will feed extended tiles to the model input
        image_grid, grad_grid = make_grid(image_tensor, tile_size, overlap)
        #Also we should add padding to the whole image. For smooth result we will use 'reflect' mode.
        img = torch.nn.functional.pad(image_tensor, [overlap]*4, 'reflect')

        for i in range(len(image_grid)):
            x_left,x_right,y_top,y_bot = image_grid[i]
            tile = img[:,:,x_left:x_right,y_top:y_bot]
            tile.requires_grad = True
            g = self.get_gradient(tile)
            if overlap != 0:
                g = g[:,:,overlap:-overlap,overlap:-overlap]
            g /= (g.std()+1e-8)
            x_left,x_right,y_top,y_bot = grad_grid[i]
            grad[:,:,x_left:x_right,y_top:y_bot] = g
        return grad/(grad.std()+1e-8)


    def render_image(self, image, sigma, iter_n=10, step=0.05, pad=10, tile_size=400,show_gradient=False,show_images=False):
        """
        Use gradient ascent to optimize an image so it maximizes the
        mean value of the given layer_tensor.

        Parameters:
        image: Input image used as the starting point.
        iter_n: Number of optimization iterations to perform.
        step: Scale for each step of the gradient ascent.
        tile_size: Size of the tiles when calculating the gradient.
        show_gradient: Plot the gradient in each iteration.
        """
        # Copy the image so we don't overwrite the original image.
        image_tensor = image.clone().detach()
        print("Processing image: ", end="")
        for i in range(iter_n):
            # Calculate the value of the gradient.
            grad = self.tiled_gradient(image_tensor, pad, tile_size)
            np_grad = grad.cpu().numpy()
            sigma_i = (i*sigma)/iter_n+sigma
            np_grad = gaussian_filter(np_grad, sigma=sigma_i)+gaussian_filter(np_grad, sigma=2*sigma_i)
            grad = torch.tensor(np_grad, device=self.device)
            # Update the image by following the gradient.
            image_tensor += grad * step
            image_tensor = torch.clamp(image_tensor,0.0,1.0)
            if show_gradient:
                # Print statistics for the gradient.
                msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}"
                print(msg.format(np_grad.min(), np_grad.max()))
                plot_grad(np_grad.squeeze().transpose(1,2,0))
            else:
                print(". ", end="")
        if show_images:
            pil_image = to_pil(image_tensor)
            plt.imshow(pil_image)
            plt.show()
        return image_tensor


    def render_multiscale(self, image, octave_n=3, octave_scale=0.7, blend=0.1, blur_radius=1, sigma=0.5, iter_n=10, step=0.01, pad=10,tile_size=400):
        # octaves contains downscaled versions of the original image
        octaves = [image.copy()]
        for _ in range(octave_n-1):
            img_blur = octaves[-1].filter(GaussianBlur(blur_radius))
            img = resize_image(image=img_blur,size=np.int32(np.float32(img_blur.size)*octave_scale))
            octaves.append(img)
        # generate details; at this step img is smallest octave
        for octave in range(octave_n):
            if octave>0:
                img_orig = octaves[-octave]
                img = resize_image(image=img,size=img_orig.size)
                img = ImageChops.blend(img, img_orig, blend)
            img = to_pil(self.render_image(to_tensor(img,self.device),sigma,iter_n, step, pad, tile_size))
            plt.imshow(img)
            plt.show()
        return img
