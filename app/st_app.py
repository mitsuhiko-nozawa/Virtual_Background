import streamlit as st
import sys, os
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import cv2
import time
import numpy as np
import psutil
from utils import *
import asyncio
import gc
from collections import deque


######################## config ##############################
ROOT = Path(os.getcwd()) # ~/11_demospace
BG_PATH = ROOT / "back_ground"
img_path = str(ROOT / "tmp" / "temp1.pkl")
mask_path = str(ROOT / "tmp" / "temp2.pkl")
save_path = str(ROOT / "tmp" / "temp3.pkl")

process = psutil.Process(os.getpid())



######################## side bar settings ##############################

# background selector
st.sidebar.markdown("# settings")
BG_dirs = os.listdir(BG_PATH)
BG_fname = st.sidebar.selectbox("select background image", BG_dirs)
BG_path = BG_PATH / BG_fname
BG_files = sorted(os.listdir(BG_path))
SUM = len(BG_files)
cnt = 0 

# running sec
running_time = st.sidebar.slider('running time (sec)',  min_value=0, max_value=60, step=1, value=1)

# start toggle
run = st.sidebar.button('Run')

# progress bar
pbar = st.sidebar.progress(1.)

# monitoring value
text_mem = "[memused] : %f MB"
text_time = "[fps] : %f"
mem_disp = st.sidebar.empty()
time_disp = st.sidebar.empty()
fps = 7

######################## app contents #################################

st.title('Virtual Background Application')
st.subheader(f"[back ground type] {BG_fname}")
show = st.image(Image.open(str(BG_path / BG_files[0])))
#show = st.image([])

#######################################################################

if run:
    # camera settings##############################
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # (1280, 720)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    IMG_SIZE = 512

    # model load ##############################
    exec_net = get_model(ROOT)
    transform = get_transforms(IMG_SIZE)

    # data queue ##############################
    Q1 = deque(maxlen=2)
    Q2 = deque(maxlen=2)

    # async object #############################
    AO = asyncObject(
        exec_net,
        cap,
        transform,
        img_path,
        mask_path,
        save_path,
        W,
        H,
        IMG_SIZE,
        Q1,
        Q2,
    )

    # loop #####################################
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


    start_time = time.time()
    while time.time() - start_time < running_time:
        pbar.progress( (running_time - (time.time() - start_time)) / running_time)
        file = BG_files[cnt]
        AO.get_BG(str(BG_path / file))
        cnt += int(30/fps)
        cnt = cnt % SUM

        s = time.time()
            
        gather = asyncio.gather(
            AO.run("preprocess", loop),
            AO.run("inference", loop),
            AO.run("output", loop),
        )
        loop.run_until_complete(gather)
        try:
            img = AO.draw_img
            show.image(img)
        except:
            pass

        mem = process.memory_info().rss / 2**20
        fps = 1 / (time.time() - s)
        mem_disp.text(text_mem % mem)
        time_disp.text(text_time % fps)

        print(f"[prepro time] {AO.prepro_time}")
        print(f"[infer  time] {AO.infer_time}")
        print(f"[output time] {AO.output_time}")
        print()
        
    
