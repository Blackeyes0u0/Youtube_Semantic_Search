## get documents and vectors
from crawl2emb import Get_emb
import argparse

parser = argparse.ArgumentParser(description='Download a YouTube video.')
parser.add_argument('--url', type=str,help='The URL of the YouTube video.')
parser.add_argument('--num', type=int,default=16,help='how many get videos?')
args = parser.parse_args()
youtube_url = args.url
num = args.num

import crawling
# youtube_url = 'https://www.youtube.com/@NewJeans_official/shorts'
# documents = crawling.Get_shorts(youtube_url)
from crawling import Get_videos
# youtube_url = 'https://www.youtube.com/@BBCNews/videos'
documents = Get_videos(youtube_url)

docs,text_embeddings,img_embeddings =Get_emb(documents,num)

### server start
import numpy as np
import torch

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct

client = QdrantClient(
    url="https://www.gcp.cloud.qdrant.io", 
    api_key="enter your token",
)

print('Qdrant cloud connect success!')
print(client)
print()

collections = client.get_collections()
collection_exists = False

for collection in collections.collections:
    if collection.name == 'clip_collection':
        print(collection.name,'Exist!!')
        collection_exists = True
        break

if not collection_exists:
    print('collection does not exist')
    print('so create collection')
    print()
    # 컬랙션이 존재하지않으면 생성
    client.recreate_collection(
        collection_name="clip_collection",
        # vectors_config=VectorParams(
        #     size=512, # Vector size is defined by used model
        #     # distance=Distance.COSINE
        #     distance=Distance.DOT
        # )
        vectors_config={
            "text": VectorParams(
                size=512,
                distance=Distance.DOT,
            ),
            "image": VectorParams(
                size=512,
                distance=Distance.DOT,
            ),
        },
    )
from datetime import datetime
now = datetime.now()
date =int(f'{now.year}{0}{now.month}{now.day}{now.hour}{now.minute}{0}{0}{0}')

client.upsert(
    collection_name="clip_collection",
    points=[
        PointStruct(
            id=idx+date,
            # vector=vector.tolist(),
            vector={
                "text": text_e,
                "image": img_e,
            },
            payload={"link": doc['url'], "title":doc['title'],"img": doc['img']}
        )
        for idx, (text_e,img_e,doc) in enumerate(zip(text_embeddings,img_embeddings,docs))
    ]
)

### client upsert style version 2
"""
from qdrant_client.http.models import PointStruct

for index, row in sample_df.iterrows():
    client.upsert(
        collection_name="ms-coco-2017",
        points=[
            PointStruct(
                id=index,
                vector={
                    "text": row["text_vector"],
                    "image": row["image_vector"],
                },
                payload={
                    "url": row["URL"],
                    "text": row["TEXT"],
                }
            )
        ]
    )
"""