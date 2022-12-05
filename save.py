import cv2
import numpy as np
import imageio


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian(source[neighbour_x][neighbour_y] - source[x][y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = i_filtered


def bilateral_filter_own(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    return filtered_image
try:
    with open("saves/ltc.btc", "r") as f:
        a = np.fromfile(f, dtype='f4')
        a_ = a.reshape(-1,4)
        ltc = np.flipud(a_.reshape(-1,1024, 4))[..., :3]
        imageio.imwrite('saves/ltc.exr', ltc)
except:
    pass

try:
    with open("saves/stoDirect.btc", "r") as f:
        a = np.fromfile(f, dtype='f4')
        a_ = a.reshape(-1,4)
        stoDirect = np.flipud(a_.reshape(-1,1024, 4))[..., :3]
        imageio.imwrite('saves/stoDirect.exr', stoDirect)
except:
    pass

try:
    with open("saves/stoNoVis.btc", "r") as f:
        a = np.fromfile(f, dtype='f4')
        a_ = a.reshape(-1,4)
        stoNoVis = np.flipud(a_.reshape(-1,1024, 4))[..., :3]
        imageio.imwrite('saves/stoNoVis.exr', stoNoVis)
except:
    pass

try:
    with open("saves/normal.btc", "r") as f:
        a = np.fromfile(f, dtype='f4')
        a_ = a.reshape(-1,4)
        stoNoVis = np.flipud(a_.reshape(-1,1024, 4))[..., :3]
        imageio.imwrite('saves/normal.exr', stoNoVis)
except:
    pass

try:
    with open("saves/ltc_baseline.btc", "r") as f:
        a = np.fromfile(f, dtype='f4')
        a_ = a.reshape(-1,4)
        stoNoVis = np.flipud(a_.reshape(-1,1024, 4))[..., :3]
        imageio.imwrite('saves/ltc_baseline.exr', stoNoVis)
except:
    pass

try:
    with open("saves/albedo.btc", "r") as f:
        a = np.fromfile(f, dtype='f4')
        a_ = a.reshape(-1,4)
        stoNoVis = np.flipud(a_.reshape(-1,1024, 4))[..., :3]
        imageio.imwrite('saves/albedo.exr', stoNoVis)
except:
    pass

try:
    with open("saves/direct.btc", "r") as f:
        a = np.fromfile(f, dtype='f4')
        a_ = a.reshape(-1,4)
        stoNoVis = np.flipud(a_.reshape(-1,1024, 4))[..., :3]
        imageio.imwrite('saves/direct.exr', stoNoVis)
except:
    pass

try:
    with open("saves/path.btc", "r") as f:
        a = np.fromfile(f, dtype='f4')
        a_ = a.reshape(-1,4)
        stoNoVis = np.flipud(a_.reshape(-1,1024, 4))[..., :3]
        imageio.imwrite('saves/path.exr', stoNoVis)
except:
    pass

r"""
python C:\Users\dhawals\repos\build_binaries\optix-toolkit\PyOptiX\examples\denoiser.py -t 32 32 C:\Users\dhawals\repos\old_working\optix_renderer\saves\stoNoVis.exr -o  C:\Users\dhawals\repos\old_working\optix_renderer\saves\stoNovisDenoise.exr
python C:\Users\dhawals\repos\build_binaries\optix-toolkit\PyOptiX\examples\denoiser.py  -t 32 32 C:\Users\dhawals\repos\old_working\optix_renderer\saves\stoDirect.exr -o  C:\Users\dhawals\repos\old_working\optix_renderer\saves\stoDirectDenoise.exr
python .\ltc_ratio_estimator.py
"""
# from scipy.ndimage.filters import gaussian_filter

# stoDirectBlurred = gaussian_filter(stoDirect, sigma=2)
# imageio.imwrite('stoDirect_blured.exr', stoDirectBlurred)

# stoNoVis = gaussian_filter(stoNoVis, sigma=2)
# imageio.imwrite('stoNoVis_blured.exr', stoNoVis)


# finatl_color = ltc * ( stoDirectBlurred / stoNoVis)

# imageio.imwrite("final_img.exr", finatl_color)