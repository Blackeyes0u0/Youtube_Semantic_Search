from Loss import SimLoss
import torch
import wandb

# test validation set
def validation_test(model,device,valid_dataloader,temp,optimizer,batch_size):
    model.eval()
    print('validation!!')
    valid_loss = 0
    valid_alignment =0
    valid_uniformity =0

    with torch.no_grad():
        total_valid = 0
        for step,(img,texts)in enumerate(valid_dataloader):
            images= img.to(device)
            texts = texts.to(device)

            dict1 = {}
            dict1['input_ids'] = texts
            dict1['pixel_values'] = images


            y1 = model(**dict1)

            image_embeddings = y1.image_embeds
            text_embeddings = y1.text_embeds
            
            
            # loss function
            loss = SimLoss(
                hi=image_embeddings,
                ht=text_embeddings,
                temp=temp)
            
            # valid_loss,alignment_image,uniformity_image,alignment_text,uniformity_text = loss.Li()
            valid_loss,valid_alignment,valid_uniformity = loss.Li()

            # valid_alignment = (alignment_image+alignment_text)/2
            # valid_uniformity = (uniformity_image+uniformity_text)/2
            valid_loss = torch.sum(valid_loss)
            wandb.log({"valid_loss": valid_loss.item(), "valid_alignment_loss": valid_alignment.item(),"valid_uniformity_loss":valid_uniformity.item()})
            # wandb.log({"valid_loss": valid_loss, "valid_alignment_loss": valid_alignment,"valid_uniformity_loss":valid_uniformity})
            if step==(int(461/batch_size)-1):
                step_num=int(461/batch_size)
                print('valid_loss: ',valid_loss.item(),'valid_alignment: ',valid_alignment.item(),'valid_uniformity: ',valid_uniformity.item())
            total_valid+=valid_loss.item()
        return total_valid
print('compelete validation class')
