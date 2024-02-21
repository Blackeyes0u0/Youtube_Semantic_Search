# get query embedding
import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
# from dotenv import load_dotenv, find_dotenv
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os

from transformers import CLIPProcessor
# t_out = t_projector(t_encoder(inputs.input_ids).pooler_output)
# t_emb = t_out/torch.norm(t_out,dim=1,keepdim=True)
model_url = 'https://huggingface.co/Soran/XCLIP_custom/resolve/main/clip_lora_text_enc.pt'
proj_url = 'https://huggingface.co/Soran/XCLIP_custom/resolve/main/clip_lora_t_proj.pt'

# PyTorch를 사용하여 원격에서 가중치를 다운로드하고 로드합니다.
tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = torch.hub.load_state_dict_from_url(model_url)
text_projector = torch.hub.load_state_dict_from_url(proj_url)

# import time
# with st.spinner(text='In progress'):
#     st.balloons()
    
@st.cache_data
def query2emb(texts,like_hate,power):
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    print(like_hate)
    
    text_encoder.eval()
    text_projector.eval()
    with torch.no_grad():
        text_emb = text_encoder(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
        text_emb = text_projector(text_emb.pooler_output)
        logit_scale   =100 #4.6052
        query_vectors = logit_scale * text_emb /torch.norm(text_emb,dim=1,keepdim=True)

    # try:
    like = like_hate[0]
    hate = like_hate[1]
    if like!='':
        inputs = tokenizer(like, padding=True, return_tensors="pt")
        with torch.no_grad():
            text_emb = text_encoder(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
            text_emb = text_projector(text_emb.pooler_output)
            logit_scale   =100 #4.6052
            like_vector = logit_scale * text_emb /torch.norm(text_emb,dim=1,keepdim=True)
            print('#########')
            print(like_vector.shape)
    else:
        like_vector = torch.zeros((1,512),dtype=torch.float32)
    
    if hate!='':
        inputs = tokenizer(hate, padding=True, return_tensors="pt")
        with torch.no_grad():
            text_emb = text_encoder(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
            text_emb = text_projector(text_emb.pooler_output)
            logit_scale   =100 #4.6052
            hate_vector = logit_scale * text_emb /torch.norm(text_emb,dim=1,keepdim=True)
    else:
        hate_vector = torch.zeros((1,512),dtype=torch.float32)
    mean = (like_vector-hate_vector)/2
    ans = query_vectors+mean*power/5
    return ans.detach().cpu().numpy()
    # except:
        # print('like hate vector error!')
    # return query_vectors.detach().cpu().numpy()
# query2emb.clear()
# st.cache_data.clear()

from qdrant_client.http.models import NamedVector
def search(query_vectors,top_rank,API_KEY):
    client = QdrantClient(
        url="https://39e3480c-9ca1-4c0c-acf0-fa3311eab2bc.us-east4-0.gcp.cloud.qdrant.io:6333", 
        api_key=API_KEY
    )
    print('Qdrant cloud connect success!')
    print()
    # run Query - which of our stored vectors are most similar to the query?
    search_result = client.search(
        collection_name = 'clip_collection',
        # query_vector = query_vectors,
        # query_filter = Filter(
        #     must=[FieldCondition(key="city", match=MatchValue(value="London"))]
        # ),
        query_vector=NamedVector(
            name="image", # image
            vector=query_vectors,
        ),
        limit=top_rank,
        with_payload=True,
        with_vectors=False,
    )
    return search_result

# 사이드바 설정 함수
def configure_sidebar():
    # 오디오 파일 불러오기
    audio_file_path = 'music/Stroke.mp3'
    
    with open(audio_file_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()
    # Streamlit을 사용하여 웹 페이지에 오디오 재생기 삽입
    st.sidebar.audio(audio_bytes, format='audio/mp3')
    st.sidebar.caption('Enjoy my song')
    
    st.sidebar.title("Input Query")
    value_text = 'Newjeans with 5 members, cute animals at home, delicious food with meet, t1 Faker match game'
    value_text = """Newjeans with 5 members, 
cute animals at home, 
delicious food with meet"""
    texts = st.sidebar.text_area("Enter a YouTube video description", value=value_text, help="Please enter comma-separated queries")
    top_rank = st.sidebar.slider("Ranking number per Query", 1, 7, 4)
    API_KEY = st.secrets["db_password"]
    # 환경 변수에서 API 키 로드
    # API_KEY = st.sidebar.text_input("Qdrant API Key", key="vecotrDB_api_key", type="password")
    
    like_prompt = st.sidebar.text_area("Type a YouTube video prompt you like", value="", help="like prompt")
    hate_prompt = st.sidebar.text_area("Type a YouTube video prompt you don't like", value="", help="dislike prompt")
    power = st.sidebar.slider("Prompt Power", 1, 10, 5)
    if st.sidebar.button("Submit Query"):
        st.session_state.texts = texts.split(',')
        st.session_state.top_rank = top_rank
        st.session_state.API_KEY = API_KEY  # save api key session state!!!!
        st.session_state.like_hate = [like_prompt,hate_prompt]
        st.session_state.power = power
        
# 캐싱된 검색 함수
@st.cache_data
def retrieval(texts,top_rank,API_KEY,like_hate,power):
    query_vectors = query2emb(texts,like_hate,power)
    result = {}
    for i,text in enumerate(texts):
        search_results = []
        search_result = search(query_vectors[i],top_rank,API_KEY)
        for j in range(top_rank):
            link    = search_result[j].payload['link']
            title   = search_result[j].payload['title']
            img     = search_result[j].payload['img']
            id_name = search_result[j].id
            score   = search_result[j].score
            dict1 = {'link':link,'title':title,'img':img,'id':id_name,'score':score}
            search_results.append(dict1)
        result[text] = search_results
    return result
# 중앙 이미지 자르기 함수
def crop_center(img, crop_width, crop_height):
    img_width, img_height = img.size
    left = (img_width - crop_width) / 2
    top = (img_height - crop_height) / 2
    right = (img_width + crop_width) / 2
    bottom = (img_height + crop_height) / 2
    return img.crop((left, top, right, bottom))
# retrieval.clear()
# st.cache_data.clear()

# 메인 UI 구성 함수
def main_ui():
    st.title("YouTube Video Semantic Search")
    st.caption('made by :blue[ _Joonghyun_] :sunglasses:')
    st.link_button("Joonghyun Github", "https://github.com/Blackeyes0u0")
    # st.divider()
    st.markdown('---')
    
    if 'texts' in st.session_state and 'top_rank' in st.session_state:
        result = retrieval(st.session_state.texts, st.session_state.top_rank,st.session_state.API_KEY,st.session_state.like_hate,st.session_state.power)
        display_results(result)
    else:
        # intro = '''
        #     Create your own :red[Youtube Algorithm]
        # '''
        # a1 = st.subheader(intro)
        description = '''
            Enter your Query, and click the :blue[**"Submit Query"**] button.
        '''
        a3 = st.subheader(description)
        a2 = st.image("image/youtube3.jpg", width=400)  # Adjust width as needed
            
# 결과 표시 함수
def display_results(result):
    if result:
        for text in st.session_state.texts:
            # st.header(f"Query: {text}")
            st.markdown(f"""
                <div style='background-color: #ffffff; padding: 10px; border-radius: 10px;'>
                    <h1 style='text-align: center; color: #333; font-size: 24px;'>
                        Query: <span style='color: orange;'>{text}</span>
                    </h1>
                </div>
                """, unsafe_allow_html=True)
            col = st.columns(st.session_state.top_rank)
            for idx, (info, c) in enumerate(zip(result[text], col)):
                with c:
                    response = requests.get(info['img'])
                    img = Image.open(BytesIO(response.content))
                    # img = crop_center(img, 1600, 1000)
                    st.image(img, width=200)
                    display_text,meta_data = info['title'].split('게시자:')
                    creator_name, meta_text = meta_data.split('조회수')
                    # st.link_button(info['title'], info['link'])
                    st.link_button(display_text, info['link'])
                    score = info['score']
                    st.caption(f'Ranking score: {score}')
                    st.caption(f'creator name: {creator_name}')
                    st.caption(f'YouTube views & date : {meta_text}')
                    # st.caption(f'Ranking score: {info['rank']}')
    else:
        st.warning("no existing search query. try another querys")
        
# 사이드바 구성
configure_sidebar()
# 메인 UI 구성
main_ui()

# Stop execution immediately:
# st.stop()
# # Rerun script immediately:
# st.experimental_rerun()

# # Group multiple widgets:
# with st.form(key='my_form'):
#     username = st.text_input('Username')
#     password = st.text_input('Password')
#     st.form_submit_button('Login')

# st.button('Hit me')
# st.checkbox('Check me out')
# st.radio('Pick one:', ['nose','ear'])
# st.selectbox('Select', [1,2,3])
# st.multiselect('Multiselect', [1,2,3])
# st.slider('Slide me', min_value=0, max_value=10)
# st.select_slider('Slide to select', options=[1,'2'])
# st.text_input('Enter some text')
# st.number_input('Enter a number')
# st.text_area('Area for textual entry')
# st.date_input('Date input')
# st.time_input('Time entry')
# st.file_uploader('File uploader')
# st.camera_input("一二三,茄子!")
# st.color_picker('Pick a color')

# # Insert containers separated into tabs:
# tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
# tab1.write("this is tab 1")
# tab2.write("this is tab 2")

# # You can also use "with" notation:
# with tab1:
#     st.radio('Select one:', [1, 2])

# # E.g. Dataframe computation, storing downloaded data, etc.
# @st.cache_data
# def foo(bar):
#   # Do something expensive and return data
#   return data
# # Executes foo

# d1 = foo(ref1)
# # Does not execute foo
# # Returns cached item by value, d1 == d2
# d2 = foo(ref1)
# # Different arg, so function foo executes
# d3 = foo(ref2)
# # Clear all cached entries for this function
# foo.clear()
# # Clear values from *all* in-memory or on-disk cached functions
# st.cache_data.clear()

# Insert a chat message container.
# with st.chat_message("user"):
    # st.write("Hello 👋")
    # st.line_chart(np.random.randn(30, 3))

# Display a chat input widget.
# x = st.chat_input("Say something")
# print(x)

# st.echo()
# with st.echo():
#     st.write('Code will be executed and printed')
    
    
# # Replace any single element.
# element = st.empty()
# # element.line_chart(...)
# # element.text_input(...)  # Replaces previous.

# # Insert out of order.
# elements = st.container()
# # elements.line_chart(...)
# st.write("Hello")
# # elements.text_input(...)  # Appears above "Hello".

# import pandas as pd
# st.help(pd.DataFrame)
# st.get_option(key)
# st.set_option(key, value)
# st.set_page_config(layout='wide')
# st.experimental_show(objects)
# st.experimental_get_query_params()
# st.experimental_set_query_params(**params)

# Show a spinner during a process
# import time
# with st.spinner(text='In progress'):
#     time.sleep(1)
#     # st.success('Done')

# Show and update progress bar
# bar = st.progress(50)
# time.sleep(3)
# bar.progress(100)

# st.balloons()
# st.snow()
# st.toast('Mr Stay-Puft')
# st.error('Error message')
# st.warning('Warning message')
# st.info('Info message')
# st.success('Success message')