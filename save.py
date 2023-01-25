import cv2
import numpy as np
import imageio

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
    with open("saves/normalBuffer.btc", "r") as f:
        a = np.fromfile(f, dtype='f4')
        a_ = a.reshape(-1,4)
        stoNoVis = np.flipud(a_.reshape(-1,1024, 4))[..., :3]
        imageio.imwrite('saves/normal.exr', stoNoVis)
except:
    pass


try:
    with open("saves/materialIDBuffer.btc", "r") as f:
        a = np.fromfile(f, dtype='f4')
        a_ = a.reshape(-1,4)
        stoNoVis = np.flipud(a_.reshape(-1,1024, 4))[..., :3]
        imageio.imwrite('saves/uv.exr', stoNoVis)
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
    print('albedo')
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