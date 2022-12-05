import imageio
import numpy as np

ltc = np.array(imageio.imread('saves/ltc.exr'))
stoDirect = np.array(imageio.imread('saves/stodirectDenoise.exr'))
stoNoVis = np.array(imageio.imread('saves/stoNoVisDenoise.exr'))

colors = ltc * (stoDirect / stoNoVis)

imageio.imwrite('saves/final_ratio_estimator.exr', colors)