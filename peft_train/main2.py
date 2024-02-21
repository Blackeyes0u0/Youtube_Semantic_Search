import os
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import requests
import warnings
warnings.filterwarnings(action='ignore')

class image_title_dataset(Dataset):
    def __init__(self, list_image_path,list_txt,transforms,tokenizer,processor):
        # Initialize image paths and corresponding title
        self.image_path = list_image_path

        # Tokenize text using CLIP's tokenizera
        self.title = tokenizer(text=list_txt, padding=True, return_tensors="pt")
        #return_tensors='np' 하면 numpy array and pt는 pytorch tensor.
        self.transform = transforms
        self.processor = processor
        
    def __len__(self): #split 할때 여기를 보고 나눠줌.
        return len(self.image_path)

    def __getitem__(self, idx): # iterate하게 만들어주는것.

        # tokenize text token make as (batch_size,77)
        total_length = self.title['input_ids'].shape[0]
        rest_token_num = 77-self.title['input_ids'].shape[1]
        dummy = torch.ones(total_length,rest_token_num)*49407
        #여기서 49407 is PAD token이다. padding을 넣어주어서 길이를 맞춰주는것.

        text_token_tensor = torch.concat((self.title['input_ids'],dummy),dim=1)

        # torch.int64가 원하는 형식이다!!! 이걸로 맞춰줄것!
        text_token_tensor = text_token_tensor.type(torch.int64)
        # dummy와 합쳐서 전체길이,77(최대 토큰수)의 인풋으로 만들어줌.

        # Preprocess image using CLIP's preprocessing function
        image = Image.open(self.image_path[idx])
        image = self.processor(images=image, return_tensors="pt")['pixel_values']
        image = torch.tensor(image)
        if self.transform != None:
            image = self.transform(image).squeeze() # torch.tensor([3,224,224])

        title = text_token_tensor[idx] #.unsqueeze(dim=0) # torch.tensor([77])
        return image,title

def train(model, device, train_loader, optimizer, epoch, cfg):
    model.train()  # 모델을 훈련 모드로 설정
    total_loss = 0
    for batch_idx, (images, titles) in enumerate(tqdm(train_loader, desc="Training")):
        images, titles = images.to(device), titles.to(device)

        optimizer.zero_grad()  # 그래디언트 초기화
        output = model(images, titles)  # 모델의 forward pass
        loss = F.cross_entropy(output, titles)  # Loss 계산
        loss.backward()  # 그래디언트 계산
        optimizer.step()  # 매개변수 업데이트

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}')


def validate(model, device, validation_loader, cfg):
    model.eval()  # 모델을 평가 모드로 설정
    total_loss = 0
    with torch.no_grad():  # 그래디언트 계산을 비활성화
        for images, titles in tqdm(validation_loader, desc="Validation"):
            images, titles = images.to(device), titles.to(device)
            output = model(images, titles)
            loss = F.cross_entropy(output, titles)  # Loss 계산
            total_loss += loss.item()
    avg_loss = total_loss / len(validation_loader)
    print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Randomness 설정
    if cfg.training.RANDOMNESS:
        torch.manual_seed(cfg.training.random_seed)
        np.random.seed(cfg.training.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(cfg.training.RANDOMNESS)
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 로드 및 전처리
    data = pd.read_csv(cfg.dataset.metadata_csv)
    
    # 이미지 및 텍스트 리스트 생성 로직
    img_dir = cfg.dataset.img_dir
    
    img_list = []
    id_list = []
    for file in os.listdir(img_dir):
        ch_dir = os.path.join(img_dir,file)
        for images in os.listdir(ch_dir):
            # print(images)
            if images.endswith('.jpg'):
                img_list.append(os.path.join(ch_dir, images))
                img_id = images.split('.jpg')[0]
                id_list.append(img_id)

    text_list = []
    cat_list = []
    ch_list = []
    
    for i in id_list:
        try:
            text_list.append(data[data["Id"]==i]['Title'].values[0])
            cat_list.append(data[data["Id"]==i]['Category'].values[0])
            ch_list.append(data[data["Id"]==i]['Channel'].values[0])
        except:
            i = i.split(' (1)')[0]
            text_list.append(data[data["Id"]==i]['Title'].values[0])
            cat_list.append(data[data["Id"]==i]['Category'].values[0])
            ch_list.append(data[data["Id"]==i]['Channel'].values[0])

    new_list = []
    for cat,text in zip(cat_list,text_list):
        new_list.append(cat+', '+text)

    print(len(new_list),len(text_list))
    
    # 모델 및 토크나이저 초기화
    model = CLIPModel.from_pretrained(cfg.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    processor = AutoProcessor.from_pretrained(cfg.model.model_name)

    # 데이터셋 및 DataLoader 설정
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ColorJitter(
            brightness=cfg.dataset.CJ_value,
            contrast=cfg.dataset.CJ_value,
            saturation=cfg.dataset.CJ_value,
            hue=cfg.dataset.CJ_value
        ),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 및 DataLoader 인스턴스 생성
    dataset = image_title_dataset(img_list,new_list,transform,tokenizer,processor)
    
    print(dataset)
    print(len(dataset))
    # print(dataset[0][1].shape)
    # print(dataset[0][0].shape)

    
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    print(len(train_dataset),len(valid_dataset))

    print()
    print('split dataset : complete')
    batch_size = cfg.training.batch_size
    # drop last해서 constrastive learning의 배치 개수별로 학습되게 맞춰줌.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=cfg.training.SHUFFLE,num_workers=6, collate_fn=None, pin_memory=True,drop_last=True,persistent_workers=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=cfg.training.SHUFFLE,num_workers=2, collate_fn=None, pin_memory=True,drop_last=True,persistent_workers=True)
    
    from scheduler import CosineAnnealingWarmUpRestarts

    from lora import LoRA_Config,LoRALayer,print_trainable_parameters
    _,basic_model_params_num =  print_trainable_parameters(model)

    def apply_lora_to_model(model, config):
        for name, module in model.named_modules():
            hierarchy = name.split('.')
            if len(hierarchy) > 1:  # Ensure the module is not the top-level module
                parent_module = model
                for submodule_name in hierarchy[:-1]:  # Navigate to the parent module
                    parent_module = getattr(parent_module, submodule_name)
                
                layer_name = hierarchy[-1]
                for target_module in config.target_modules:
                    if target_module in layer_name:
                        original_layer = getattr(parent_module, layer_name)
                        if isinstance(original_layer, nn.Linear):
                            setattr(parent_module, layer_name, LoRALayer(original_layer, config))
                            # print(f"Replaced {name} with LoRALayer")
        return model

    lora_config = LoRA_Config(
        r=cfg.lora.r, 
        lora_alpha=cfg.lora.alpha, 
        lora_dropout=cfg.lora.dropout, 
        merge_weights=cfg.lora.merge_weights, 
        target_modules=cfg.lora.target_modules,
    )
    # Apply LoRA to the model
    model = apply_lora_to_model(model, lora_config)

    def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
        for n, p in model.named_parameters():
            if 'lora_' not in n:
                p.requires_grad = False
        if bias == 'none':
            return
        elif bias == 'all':
            for n, p in model.named_parameters():
                if 'bias' in n:
                    p.requires_grad = True
        elif bias == 'lora_only':
            for m in model.modules():
                if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                        m.bias.requires_grad = True
        else:
            raise NotImplementedError
        
    mark_only_lora_as_trainable(model,bias='lora_only')

    _,basic_model_params_num =  print_trainable_parameters(model)

    from Loss import SimLoss


    from train import train
    from validation import validation_test
        
    
    print('gooooodddd!!!')

    for step,(name,p) in enumerate(model.named_parameters()):
        if p.requires_grad:
            print(name)
            if step>10:
                break
            
    import wandb
    wandb.login()
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts , CyclicLR, ExponentialLR,StepLR, CosineAnnealingLR
    import torch.optim as optim

    LR = cfg.training.LR #7e-7
    EPOCHS = cfg.training.EPOCHS
    LR_max = cfg.training.LR_MAX

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    # loss_alignment = nn.MSELoss()

    # cyclic LR
    # optimizer = optim.AdamW(model.parameters(), lr=2e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    optimizer = optim.AdamW(model.parameters(), lr = LR)#,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=70, T_mult=1, eta_max=LR_max,  T_up=17, gamma=0.9)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=13, T_mult=1, eta_max=LR_max,  T_up=7, gamma=0.9)
    # scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-4, step_size_up=5, step_size_down=5, mode='triangular2', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.95, last_epoch=-1)

    import datetime
    rank = cfg.lora.r
    # 현재 날짜와 시간을 포함한 실행 이름 생성
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{rank}_LR_{LR_max}_Epochs_{EPOCHS}_{current_time}"

    wandb.init(
        project="Youtube_CLIP",
        name=run_name,  # 동적으로 생성된 실행 이름 사용
        config={
            "learning_rate": LR,
            "architecture": "CLIP+LoRA+SimCSELR",
            "dataset": "youtube thumbnails",
            "epochs": EPOCHS,
        }
    )


    temp = 2.7 # clip paper logit scaling value.

    earlystopping = 6
    paitent = 0
    now = float('inf')
    model.to(device)
    for epoch in tqdm(range(EPOCHS)):
        print("Epoch : ",epoch)
        model.train()
        train(model,device,train_dataloader,temp,optimizer)
        total_valid=validation_test(model,device,valid_dataloader,temp,optimizer,batch_size)
        
        scheduler.step()
            
        if now>total_valid:
            now = total_valid
            paitent = 0
            if epoch%3==1:
                
                torch.save(model,f'model_save/nocse_lora_{rank}_{LR_max}_{epoch}.pt')    
        else:
            paitent+=1
            if paitent>earlystopping:
                break
    wandb.finish()
    print('train Fin!!!!!')
    # torch.save(model,f'model_save2/clora_{rank}_{LR_max}_{epoch}.pt')    
if __name__ == "__main__":
    main()
