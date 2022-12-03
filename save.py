import cv2
import numpy as np


f = open("rendered_output.btc", "r")
a = np.fromfile(f, dtype='f4')

a_ = a.reshape(-1,4)
a_ = np.flipud(a_.reshape(-1,1920, 4))
print("test", a_.shape)
cv2.imwrite('test.png', cv2.cvtColor((a_ * 255.).astype('uint8'), cv2.COLOR_BGR2RGB))