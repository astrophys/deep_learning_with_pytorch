# Author : Ali Snedden
# Date   : 09/18/24
# Goals (ranked by priority) :
#
# Refs :
#   a)
#   #) https://www.nltk.org/book/ch06.html
#
# Copyright (C) 2024 Ali Snedden
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import sys
import pyheif
import argparse
import numpy as np
from PIL import Image
#from pillow_heif import register_heif_opener


def main():
    """Exercise 4.1 : Take pictures with a digital camera.
        a) Load each image, and convert it to a tensor
        b) For each image tensor, use the .mean() method to get a sense of how
           bright the image is
        c) Take the mean of each channel of your images. Can you identify the red,
           green and blue items from only the channel averages?

    Args

        N/A

    Returns

    Raises

    """
    parser = argparse.ArgumentParser(
                    description="This does exercise 4.1")
    parser.add_argument('--path', metavar='path_to_image', type=str,
                        help='Path to the image')
    args = parser.parse_args()
    path = args.path

    im = Image.open(path)
    size = im.size
    rgb  = im.convert("RGB")
    r,g,b = rgb.split()
    rM = np.array(r)
    gM = np.array(g)
    bM = np.array(b)
    #heif_file = pyheif.read(path)
    #image = Image.frombytes(
    #    heif_file.mode,
    #    heif_file.size,
    #    heif_file.data,
    #    "raw",
    #    heif_file.mode,
    #    heif_file.stride,
    #    )
    print("red [min,max,mean]     = [{:<.2f},{:<.2f},{:<.2f}] ".format(np.min(rM),
           np.max(rM), np.mean(rM)))
    print("green [min,max,mean]   = [{:<.2f},{:<.2f},{:<.2f}] ".format(np.min(gM),
           np.max(gM), np.mean(gM)))
    print("blue [min,max,mean]    = [{:<.2f},{:<.2f},{:<.2f}] ".format(np.min(bM),
           np.max(bM), np.mean(bM)))

    # data/exercises/ch4/20260207_tiger_at_zoo.jpg
    # red [min,max,mean]     = [0.00,255.00,97.62]
    # green [min,max,mean]   = [0.00,255.00,100.19]
    # blue [min,max,mean]    = [0.00,255.00,104.61]
    # --> This is a mostly white image, which makes sense while these are all
    #     so close in value

    return sys.exit(0)


if __name__ == "__main__":
    main()
