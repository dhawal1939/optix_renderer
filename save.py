import os
import numpy as np
import imageio

btc_files = os.listdir('saves/')
btc_files = [b for b in btc_files if '.btc' in b ]
for btc_file in btc_files:
    try:
        with open(f"saves/{btc_file}", "r") as f:
            a = np.fromfile(f, dtype='f4')
            a_ = a.reshape(-1,4)
            ltc = np.flipud(a_.reshape(-1,1024, 4))[..., :3]
            imageio.imwrite(f'saves/{btc_file.split(".")[0]}.exr', ltc)
    except:
        print(btc_file)
