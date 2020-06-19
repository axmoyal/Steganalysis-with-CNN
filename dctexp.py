import numpy as np
import matplotlib.pyplot as plt
import scipy

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc
import matplotlib.pylab as pylab

import imageio
im = np.array(imageio.imread("./data" + "/JMiPOD/" + "00000" + ".jpg") )
imog = np.array(imageio.imread("./data" + "/Cover/" + "00000" + ".jpg") )

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def rgb2ycb(im): 
    out = np.zeros(im.shape) 
    out[:,:,0] = 0.299*im[:,:,0] + 0.587*im[:,:,1] + 0.114*im[:,:,2]
    out[:,:,1] = 128 -  0.168736*im[:,:,0] - 0.331264*im[:,:,1] + 0.5*im[:,:,2]
    out[:,:,2] = 128 + 0.5*im[:,:,0] - 0.418688*im[:,:,1] - 0.081312*im[:,:,2]
    return out

im = rgb2ycb(im)
imog = rgb2ycb(imog)
imsize = im.shape
dct = np.zeros(imsize)
dctog = np.zeros(imsize)

for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dct[i:(i+8),j:(j+8)] = dct2( im[i:(i+8),j:(j+8)]  )
        dctog[i:(i+8),j:(j+8)] = dct2( imog[i:(i+8),j:(j+8)] )
        plt.subplot(331)
        print(dct[i:(i+8),j:(j+8), 0])
        print(dct[i:(i+8),j:(j+8)].shape)
        plt.imshow(dct[i:(i+8),j:(j+8),0]+ 128)
        plt.subplot(332)
        plt.imshow(dctog[i:(i+8),j:(j+8),0]+128)
        plt.subplot(333)
        plt.imshow(np.abs(dctog[i:(i+8),j:(j+8),0]- dct[i:(i+8),j:(j+8),0]))
        plt.subplot(334)
        plt.imshow(dct[i:(i+8),j:(j+8),1]+ 128)
        plt.subplot(335)
        plt.imshow(dctog[i:(i+8),j:(j+8),1]+128)
        plt.subplot(336)
        plt.imshow(np.abs(dctog[i:(i+8),j:(j+8),1]- dct[i:(i+8),j:(j+8),1]))
        plt.subplot(337)
        plt.imshow(dct[i:(i+8),j:(j+8),2]+ 128)
        plt.subplot(338)
        plt.imshow(dctog[i:(i+8),j:(j+8),2]+128)
        plt.subplot(339)
        plt.imshow(np.abs(dctog[i:(i+8),j:(j+8),2]- dct[i:(i+8),j:(j+8),2]))
        plt.show()
