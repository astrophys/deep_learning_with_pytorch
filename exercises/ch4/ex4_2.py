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
import re
import sys
import pyheif
import argparse
import numpy as np
from PIL import Image
#from pillow_heif import register_heif_opener


def main():
    """Exercise 4.2 : Select a relatively large file containing Python source code
        a) Build an index of all the words in the source file (feel free to make
           your tokenization as simple or as complex as you like; we suggest
           starting with replacing r"[^a-zA-Z0-9_]+" with spaces).
            --> DONE
        b) Compare your index with the one we made for Pride and Prejudice. Which
           is larger?
            --> Pride and prejudice (data/p1ch4/jane-austen/1342-0.txt) : 7246
            --> gadget4-ali/src/mergertree/halotrees.cc : 815
        c) Create the one-hot encoding for the source code file.
        d) What information is lost with this encoding? How does that
           information compare to what’s lost in the Pride and Prejudice encoding?
            --> You lose context in both cases.

    Args

        N/A

    Returns

    Raises

    """
    parser = argparse.ArgumentParser(
                    description="This does exercise 4.2")
    parser.add_argument('--path', metavar='path_to_image', type=str,
                        help='Path to the image')
    args = parser.parse_args()
    path = args.path

    fin = open(path, 'r')
    wordD    = dict()
    wordidxD = dict()
    idx = 0
    nword = 0
    wordL = []
    for line in fin:
        line = line.strip()
        line = re.sub(r"[^a-zA-Z0-9_]+", " ", line)
        line = line.split()
        nword += len(line)
        for word in line:
            wordL.append(word)
            if word not in wordD.keys():
                wordD[word] = 1
                wordidxD[word] = idx
                idx += 1
            else :
                wordD[word] += 1
    fin.close()

    # Now create one hot encoding
    onehotM = np.zeros([nword, len(wordidxD.keys())])
    for i in range(len(wordL)):
        word = wordL[i]
        idx = wordidxD[word]
        onehotM[i,idx]  = 1


    print("Number of unique words : {}".format(len(wordD.keys())))
    print("Number of words : {}".format(nword))
    return sys.exit(0)


if __name__ == "__main__":
    main()
