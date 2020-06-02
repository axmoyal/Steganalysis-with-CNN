#Script to create a dataset of fourier transforms images
from utils import *
import imageio
import os


if __name__ == "__main__":
    folders = ["Cover", "JMiPOD", "JUNIWARD", "UERD", "Test"]
    for f in folders: 
        print("Parsing through ", f)
        relpth =  "./data/" + f + "/"
        svpth = "./data/" + f +"_fourier" + "/"
        os.mkdir("./data/" + f +"_fourier") 
        for i in range(75000):
            im = imageio.imread(relpth + str(i).rjust(5, '0') + ".jpg")
            fourier = dct2(im)
            torch.save(fourier, svpth + str(i).rjust(5, '0') + ".pt")



