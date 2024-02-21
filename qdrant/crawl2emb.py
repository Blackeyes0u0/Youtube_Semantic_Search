import argparse
from pytube import YouTube
from decord import VideoReader, cpu
import torch
import numpy as np
from PIL import Image
np.random.seed(0)

# Import Library
import requests
from bs4 import BeautifulSoup
import time
import urllib.request
from selenium.webdriver import Chrome
import re
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import datetime as dt
from selenium.webdriver.common.by import By
from urllib import parse
from selenium import webdriver
import pandas as pd
from pytube import YouTube
import os

import av
import torch
import numpy as np
from transformers import CLIPModel,CLIPProcessor


# get video title,image embeddings.
import models
def Get_emb(documents,get_num):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    path ='../clip_lora_youtube_16.pt'
    # path ='../nocse_lora_32_1e-05_10.pt' # 잘못 학습시킴. 성능 안좋음.
    model = torch.load(path,map_location=torch.device('cpu'))
    models.Num_params(model)     
    
    t_encoder   = model.text_model
    v_encoder   = model.vision_model
    t_projector = model.text_projection
    v_projector = model.visual_projection
    
    cache_dir = os.path.join('./cache')
    
    result = []
    first = True
    documents = documents[:get_num]
    
    texts,imgs,urls = [],[],[]
    for doc in documents:
        texts.append(doc['title'])
        imgs.append(doc['img'])
        urls.append(doc['url'])
        
    img_inputs = torch.tensor([])
    for url in imgs:
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = processor(images=image, return_tensors="pt", padding=True,truncation=True)
        img_inputs=torch.concat((img_inputs,inputs.pixel_values),dim=0)
    v_out = v_projector(v_encoder(img_inputs).pooler_output)
    img_embs = v_out/torch.norm(v_out,dim=1,keepdim=True)
    
    ## text embeddings  # maximize max_length
    inputs = processor(text=texts, return_tensors="pt",padding=True, max_length=77,truncation=True)
    t_out = t_projector(t_encoder(inputs.input_ids).pooler_output)
    text_embs = t_out/torch.norm(t_out,dim=1,keepdim=True)
    return documents,text_embs.detach().cpu().numpy(),img_embs.detach().cpu().numpy()
