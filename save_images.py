import os
import imageio
import numpy as np
from pathlib import Path
from PIL import Image


list_imgs = os.listdir('./saves/')
list_imgs = [i for i in list_imgs if 'exr' in i]


for im_name in list_imgs:
    im = np.array(imageio.imread(f'./saves/{im_name}'))
    im_gamma_correct = np.clip(np.power(im, 1/2.2), 0, 1)
    img_name = str(Path(im_name).name).split('.')[0]
    im_fixed = Image.fromarray(np.uint8(im_gamma_correct*255))
    im_fixed.save(f'./saves/pngs/{img_name}.png')
    # imageio.imwrite(f'./saves/pngs/{img_name}.png', im)
