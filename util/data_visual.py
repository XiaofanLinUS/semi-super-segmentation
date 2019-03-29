import matplotlib.pyplot as plt
from .data_process import normalize
import numpy as np
from PIL import Image
# for saving gif animation
from packaging import version

import os
# check if directory exists


def play_img_sequence(numpy_imgs, delay=.2):
    img_buff = None
    for np_mask in numpy_imgs:
        # print(np_mask.max())
        if img_buff is None:
            img_buff = plt.imshow(np_mask)
        else:
            img_buff.set_data(np_mask)
        plt.pause(delay)
        plt.draw()

    plt.show()


def save_img_sequence(numpy_imgs, f_name, gif=False, delay=0.2):
    frames = []
    for np_img in numpy_imgs:
        np_img = (normalize(np_img)*255).astype('uint8')
        frames.append(Image.fromarray(np_img))

    if gif:
        if version.parse(Image.PILLOW_VERSION) < version.parse("3.4"):
            print("Pillow in version not supporting making animated gifs")
            print("you need to upgrade library version")
            print("see release notes in")
            print(
                "https://pillow.readthedocs.io/en/latest/releasenotes/3.4.0.html#append-images-to-gif")
        else:
            frames[0].save(f_name+'.gif', format='GIF', append_images=frames[1:],
                           save_all=True, duration=int(1000*delay), loop=0)
    else:
        if not os.path.isdir(f_name):
            os.mkdir(f_name)
        for idx, frame in enumerate(frames):
            frame.save(f_name+'/'+str(idx)+'.png')
