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

def Get_videos(url):
    documents = []
    driver = Chrome()
    # url = 'https://www.youtube.com/@marvel/videos'
    driver.get(url)

    time.sleep(2)
    #################################################################################
    ### youtube page scroll   : java script로 동적 구성되어있기때문에, 아래로 내려줘서 정보를 로딩
    #################################################################################

    body = driver.find_element(By.CSS_SELECTOR, 'body')
    for _ in range(5): # 5번 반복.
        for _ in range(7): # 7번 내리고 1초 쉬고,
            body.send_keys(Keys.PAGE_DOWN)
        time.sleep(1.5)

    #################################################################################
    ### get image,title,url - selenium이나 bs4를 이용해서 크롤링.
    #################################################################################

    time.sleep(3)
    try:
        soup = BeautifulSoup(driver.page_source)
        
        imgs_list = soup.select('div#contents div#content') # tag이름#ID이름
        img_urls = []
        result = []
        for img in imgs_list:
            try:
                
                xx =img.find('img') # img tag를 찾는다.
                ## 마찬가지로 xx도 bs4.element.Tag이다.
                print(xx['src'].split('?')[0]) ### img 존재하는거 하나 가져오기.
                img_url = xx['src'].split('?')[0] 

                a_tag = img.select('a#video-title-link')[0] ## soup에서 가져온 정보인데, list형태라서 안에 str을 가져와야해서 풀어줘야함.
                # type은 bs4.element.Tag이다.
                link = parse.urljoin(url,a_tag.get('href'))
                print('link ',link)
                
                title = a_tag['aria-label']
                print('title',title)

                documents.append({'url':link,'title':title,'img':img_url})
                
            except Exception as e:
                print(f'{e} error')
    except:
        print('error exists')

    driver.quit()
    return documents

# if __name__=="__main__":
#     # url = 'https://www.youtube.com/@NewJeans_official/videos' # 안댐
#     url = 'https://www.youtube.com/@BBCNews/videos'
#     # url = 'https://www.youtube.com/@BBCNews/featured' # 안댐
#     docs = Get_videos(url)
#     print()
#     print('docs')
#     print(docs)
#     breakpoint()
#     print(3)