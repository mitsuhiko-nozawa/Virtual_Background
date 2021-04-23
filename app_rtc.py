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
import av
from collections import deque

from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
        ]},
    media_stream_constraints={"video": True, "audio": False},
)

######################## config ##############################

ROOT = Path(os.getcwd()) # ~/11_demospace
BG_PATH = ROOT / "back_ground"
process = psutil.Process(os.getpid())
IMG_SIZE = 512
W = 640
H = 480


######################## side bar settings ##############################

# background selector
st.sidebar.markdown("# settings")
BG_dirs = os.listdir(BG_PATH)
BG_dirs.remove(".DS_Store")
BG_fname = st.sidebar.selectbox("select background image", BG_dirs)
BG_path = BG_PATH / BG_fname
BG_files = sorted(os.listdir(BG_path))
SUM = len(BG_files)
CNT = 0 

# running sec
#running_time = st.sidebar.slider('running time (sec)',  min_value=0, max_value=60, step=1, value=1)

# start toggle
#run = st.sidebar.button('Run')

# progress bar
#pbar = st.sidebar.progress(1.)

# monitoring value
#mem_disp = st.sidebar.empty()
#time_disp = st.sidebar.empty()
fps = 7

# movie upload 
import io
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload file")
if uploaded_file is not None:
    with open( ROOT / "movies" / uploaded_file.name ,"wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.text("upload is successed!")







# movie parse
st.sidebar.markdown("---")
mov_files = os.listdir("movies")
#mov_files = [file for file in mov_files if "MOV" or "mp4" in file]
mov_files = [file for file in mov_files if re.match(".+\..+", file)]
mov_file_selector = st.sidebar.selectbox("select movie file you want to parse", mov_files)
parse_result = st.sidebar.empty()
parse_toggle =  st.sidebar.button('parse')

if parse_toggle:
    try:
        parse_result.text("parsing ...")
        parse_movie(mov_file_selector, ROOT)
        parse_result.text("finished!")
    except:
        parse_result.text("movie is already parsed, please confirm...", )
    st.experimental_rerun()
        
        

######################## app contents #################################

st.title('Virtual Background Application')
st.subheader(f"[back ground type] {BG_fname}")
#show = st.image(Image.open(str(BG_path / BG_files[0])))
#show = st.image([])



class VBG_VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.exec_net = get_model(ROOT)
        self.transforms = get_transforms(IMG_SIZE)
        self.CNT = CNT
        self.BG_PATH = None

        
    def initialize_loop(self):
        file = BG_files[self.CNT]
        self.BG_PATH = str(BG_path / file)
        self.CNT += int(30/fps)
        self.CNT = self.CNT % SUM
        self.s = time.time()


    def terminalize_loop(self):
        mem = process.memory_info().rss / 2**20
        fps = 1 / (time.time() - self.s)
        print(fps)
        print(mem)



    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.initialize_loop()
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        mask = self.transforms(image=img)["image"].unsqueeze(0)
        mask = np.array(self.exec_net.infer({"input" : mask})["output"]).argmax(axis=1)[0].astype("uint8")
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, 2)

        BG = cv2.imread(str(self.BG_PATH))
        BG = cv2.resize(BG, (W, H), interpolation=cv2.INTER_NEAREST)
        BG = centerCrop_resize(BG, H, W)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img * mask + BG*(-mask+1)
        self.terminalize_loop()
        return img


webrtc_streamer(
    key="loopback",
    mode=WebRtcMode.SENDRECV,
    client_settings=WEBRTC_CLIENT_SETTINGS,
    video_transformer_factory=VBG_VideoTransformer, 
    async_transform=True,
)