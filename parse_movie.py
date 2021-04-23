import os
import cv2
from pathlib import Path
import sys
import re
import albumentations as A
"""
ROOT = Path(".")
input_dir = ROOT / "movies"
out_dir = ROOT / "personal"
files = os.listdir(input_dir)
os.makedirs(out_dir, exist_ok=True)
ext = "jpg"

for file in files:
    base_fname = file.replace(".MOV", "")
    out_mov_dir = out_dir / base_fname
    os.makedirs(out_mov_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(input_dir / file))
    print(str(input_dir / file))
    
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    n = 0
    while True:
        ret, frame = cap.read()
        if ret:
            save_fname = out_mov_dir / f"{str(n).zfill(digit)}.{ext}"
            cv2.imwrite(str(save_fname), frame)
            n += 1
        else:
            break
"""

def parse(file):
    file_base_name = file.split('.')[0]
    ROOT = Path(".")
    input_dir = ROOT / "movies"
    out_dir = ROOT / "back_ground" / file_base_name
    os.makedirs(out_dir, exist_ok=True)
    ext = "jpg"

    cap = cv2.VideoCapture(str(input_dir / file))    
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    n = 0
    while True:
        ret, frame = cap.read()
        if ret:
            # 最大1:1.5までに収める
            W, H = frame.shape[1], frame.shape[0]
            if W*1.5 < H:
                H = int(W*1.5)
            elif H*1.5 < W:
                W = int(H*1.5)
            tr = A.CenterCrop(H, W)
            frame = tr(image=frame)["image"]
            #frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)
            save_fname = out_dir / f"{str(n).zfill(digit)}.{ext}"
            cv2.imwrite(str(save_fname), frame)
            n += 1
        else:
            break






if __name__ == "__main__":
    target_file = sys.argv[1]
    parse(target_file)