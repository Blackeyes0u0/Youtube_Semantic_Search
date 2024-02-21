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


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    # start_idx = 0
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64) #min보다 작은 값을 min으로 바꿔주고 큰것은 max값으로 바꿔줌.
    return indices

def Get_shorts(youtube_url):
    documents = []
    driver = Chrome()
    url = youtube_url
    driver.get(url)

    time.sleep(2)
    
    body = driver.find_element(By.CSS_SELECTOR, 'body')
    for _ in range(5): # 1번 반복.
        for _ in range(7): # 7번 내리고 1.5초 쉬고,
            body.send_keys(Keys.PAGE_DOWN)
        time.sleep(1.5)
    try:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        imgs_list = soup.select('ytd-thumbnail')
        for img in imgs_list:
            try:
                # 썸네일 이미지 URL 추출
                img_url = img.find('img').get('src').split('?')[0]

                # 동영상 링크 추출
                a_tag = img.find('a', id='thumbnail')  # 'id=thumbnail'은 YouTube 썸네일 링크에 일반적으로 사용됩니다.
                url = parse.urljoin(url, a_tag.get('href'))

                # 동영상 제목 추출 (선택자와 속성은 페이지 구조에 따라 다를 수 있음)
                title = a_tag.get('video-title')  # 'title' 속성을 사용하거나 다른 방법을 찾아야 할 수 있습니다.
                
                json_data = {}
                json_data['url'] = url
                json_data['title'] =title
                json_data['img'] = img_url
                documents.append(json_data)
            except Exception as e:
                print('image or link error:', e)
    except Exception as e:
        print('error exists:', e)

    driver.quit()
    return documents
