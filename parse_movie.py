import os
import cv2
from pathlib import Path
import sys
import re
import albumentations as A
from utils import parse_movie

#  usage :
#  python parse_movie.py <movie file name>

if __name__ == "__main__":
    ROOT = Path('.')
    target_file = sys.argv[1]
    parse_movie(target_file, ROOT)