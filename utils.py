import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from openvino.inference_engine import IECore
from pathlib import Path
import asyncio
import time
import sys, os
import numpy as np
import pickle
import re


def read_image(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def center_crop(img, height, width):
    H, W = img.shape[0], img.shape[1]
    mid_x, mid_y = int(W/2), int(H/2)
    cw2, ch2 = int(width/2), int(height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def centerCrop_resize(img, H, W): # 720, 1280
    ratio = W / H
    width, height = img.shape[1], img.shape[0] # 画像サイズ
    if width / ratio < H:
        height = int(width / ratio)
    elif height * ratio < W:
        width = int(height * ratio)
    img = center_crop(img, height, width)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    return img

    
def get_transforms(img_size):
    transforms =  A.Compose([
                    A.Resize(img_size, img_size),
                    A.Normalize(),
                    ToTensorV2(),
                ])
    return transforms


def get_model(ROOT):
    model_path = str(ROOT / "model" / "model.xml")
    ie = IECore()
    ie_net = ie.read_network(model=model_path, weights=model_path.replace("xml", "bin"))
    exec_net = ie.load_network(network=ie_net, num_requests=1, device_name="CPU")
    return exec_net


class asyncObject:
    def __init__(self, exec_net, cap, transform, img_path, mask_path, save_path, W, H, IMG_SIZE, Q1, Q2):
        self.exec_net = exec_net
        self.cap = cap
        self.transform = transform
        self.img_path = img_path
        self.mask_path = mask_path
        self.save_path = save_path
        self.W = W
        self.H = H
        self.IMG_SIZE = IMG_SIZE
        self.BG_path = None
        self.Q1 = Q1
        self.Q2 = Q2

        self.draw_img = None
        self.prepro_time = None
        self.infer_time = None
        self.output_time = None

    def get_BG(self, BG_path):
        self.BG_path = BG_path


    async def run(self, stage, loop):
        if stage == "preprocess":
            await loop.run_in_executor(None, self.preprocess, self.cap, self.transform, self.img_path)
        elif stage == "inference":
            await loop.run_in_executor(None, self.inference, self.exec_net, self.img_path, self.mask_path)
        elif stage == "output":
            await loop.run_in_executor(None, self.output, self.W, self.H, self.BG_path, self.mask_path, self.save_path, self.IMG_SIZE)

    def preprocess(self, cap, transform, img_path):
        t = time.time()
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        mask = transform(image=img)["image"].unsqueeze(0)
        if os.path.exists(img_path): os.remove(img_path)
        obj = {
            "img" : img,
            "mask" : mask
        }
        self.Q1.append(obj)
        self.prepro_time = time.time() - t

    def inference(self, exec_net, img_path, mask_path):
        t = time.time()
        try:
            obj = self.Q1.popleft()
            img = obj["img"]
            mask = obj["mask"]
            mask = np.array(exec_net.infer({"input" : mask})["output"]).argmax(axis=1)[0].astype("uint8")
            
            if os.path.exists(mask_path): os.remove(mask_path)
            obj = {
                "img" : img,
                "mask" : mask
            }
            self.Q2.append(obj)

        except:
            pass
        self.infer_time = time.time() - t


    def output(self, W, H, BG_path, mask_path, save_path, IMG_SIZE):
        t = time.time()
        try:
            obj = self.Q2.popleft()
            img = obj["img"]
            mask = obj["mask"]
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, 2)

            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            BG = read_image(BG_path)
            #BG = cv2.resize(BG, (W, H), interpolation=cv2.INTER_NEAREST)
            BG = centerCrop_resize(BG, H, W)

            img = img * mask + BG*(-mask+1)
            self.draw_img = img

        except:
            pass
        self.output_time = time.time() - t



class asyncObject_rtc:
    def __init__(self, exec_net, transform, W, H, IMG_SIZE, Q1, Q2):
        self.exec_net = exec_net
        self.transform = transform
        self.W = W
        self.H = H
        self.IMG_SIZE = IMG_SIZE
        self.BG_path = None
        self.Q1 = Q1
        self.Q2 = Q2

        self.frame = None
        self.draw_img = None
        self.prepro_time = None
        self.infer_time = None
        self.output_time = None

    def set_BG_path(self, BG_path):
        self.BG_path = BG_path

    def set_frame(self, frame):
        self.frame = frame


    async def run(self, stage, loop):
        if stage == "preprocess":
            await loop.run_in_executor(None, self.preprocess, self.transform)
        elif stage == "inference":
            await loop.run_in_executor(None, self.inference, self.exec_net)
        elif stage == "output":
            await loop.run_in_executor(None, self.output, self.W, self.H, self.BG_path, self.IMG_SIZE)

    def preprocess(self, transform):
        t = time.time()
        frame = self.frame
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        mask = transform(image=img)["image"].unsqueeze(0)
        obj = {
            "img" : img,
            "mask" : mask
        }
        self.Q1.append(obj)
        self.prepro_time = time.time() - t

    def inference(self, exec_net):
        t = time.time()
        try:
            obj = self.Q1.popleft()
            img = obj["img"]
            mask = obj["mask"]
            mask = np.array(exec_net.infer({"input" : mask})["output"]).argmax(axis=1)[0].astype("uint8")
            
            obj = {
                "img" : img,
                "mask" : mask
            }
            self.Q2.append(obj)

        except:
            pass
        self.infer_time = time.time() - t


    def output(self, W, H, BG_path, IMG_SIZE):
        t = time.time()
        try:
            obj = self.Q2.popleft()
            img = obj["img"]
            mask = obj["mask"]
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, 2)

            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #BG = read_image(BG_path)
            BG = cv2.imread(str(BG_path))
            #BG = cv2.resize(BG, (W, H), interpolation=cv2.INTER_NEAREST)
            BG = centerCrop_resize(BG, H, W)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = img * mask + BG*(-mask+1)
            self.draw_img = img

        except:
            pass
        self.output_time = time.time() - t


def parse_movie(file, ROOT):
    file_base_name = file.split('.')[0]
    input_dir = ROOT / "movies"
    out_dir = ROOT / "back_ground" / file_base_name
    if os.path.exists(out_dir): 
        raise Exception("movie is already parsed, please confirm.")
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