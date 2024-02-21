from Loss import SimLoss
import torch
import wandb
import torch.nn as nn
def train(model,device,train_dataloader,temp,optimizer):
    for step,(img,texts)in enumerate(train_dataloader):
        optimizer.zero_grad()
        ## logging
        lr = optimizer.param_groups[0]['lr']
        
        dict1 = {}
        dict1['input_ids'] = texts.to(device)
        dict1['pixel_values'] = img.to(device)

        y1 = model(**dict1)

        image_embeddings = y1.image_embeds
        text_embeddings = y1.text_embeds
        
        ####################
        ## simCSE dropout ##
        ####################      
        
        # e1 = torch.nn.Dropout(p=0.1)
        # e2 = torch.nn.Dropout(p=0.1)
        
        # y1 = e1(text_embeddings)
        # y2 = e2(text_embeddings)

        # texts_loss = SimLoss(
        #     hi=y1,
        #     ht=y2,
        #     temp=0.05) # simCSE paper best temperatures.
        # simcse_loss,cse_alignment,cse_uniformity = texts_loss.Li()
        
        
        # print(image_embeddings.shape)
        # print(text_embeddings)
        
        
        # loss function
        loss = SimLoss(
            hi=image_embeddings,
            ht=text_embeddings,
            temp=temp)
        
        # train_loss,alignment_image,uniformity_image,alignment_text,uniformity_text = loss.Li()
        train_loss,alignment,uniformity = loss.Li()

        # train_loss = torch.sum(alignment+anisotropy)/batch_size
        train_loss = torch.sum(train_loss)
        train_loss.backward()
        optimizer.step()

        # alignment = (alignment_image+alignment_text)/2
        # uniformity = (uniformity_image+uniformity_text)/2
        wandb.log({"Learning rate":lr,"train_loss": train_loss.item(), "alignment": alignment.item(),"uniformity":uniformity.item()})
        # wandb.log({"Learning rate":lr,"train_loss": train_loss.item()})
        if step%10==0:
            print("Learning rate",lr)
            print("train_loss: ", train_loss.item(), "alignment: ", alignment.item(),"uniformity: ",uniformity.item())
            # print(step,"'s batch  ",'  &loss :',round(train_loss.item(),5),'alignment : ',round(alignment.mean().item(),4),' anisotropy :',round(anisotropy.mean().item(),4))
print('define train function')