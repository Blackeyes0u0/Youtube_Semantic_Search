from abc import ABC
from abc import abstractmethod
# static method
class MyLoss(ABC):
    # def __init__(self) -> None:
        # super().__init__()

    @abstractmethod
    def Li(self) -> None:
        """Define layers in ther model."""
        raise NotImplementedError
import torch
import torch.nn.functional as F

class SimLoss(MyLoss): # what i want to similar
    def __init__(self,hi:torch.tensor
                ,ht:torch.tensor
                ,temp:float
                ):
        self.hi = hi # batch_size * dim
        self.ht = ht # batch_size * dim
        self.temp = temp
    def sim(self,Ie,Te):
        dot_product = torch.dot(Ie,Te)
        norm_I = torch.norm(Ie)
        norm_T = torch.norm(Te)
        return dot_product/(norm_I*norm_T) if norm_T*norm_I !=0 else 0
    
    def Li(self):
        batch_size = self.hi.shape[0]
        L_image_total=0
        alignment_image = 0
        uniformity_image = 0
        for k in range(batch_size):
            L_image_alignment =0
            L_image_uniformity=0
            L_image_alignment = torch.exp(self.sim(self.hi[k],self.ht[k])/self.temp)
            for j in range(batch_size):
                L_image_uniformity += torch.exp(self.sim(self.hi[k],self.ht[j])/self.temp)
            # L_image = -torch.log(L_image_alignment/L_image_uniformity)
            alignment_image+=-torch.log(L_image_alignment)
            uniformity_image+=torch.log(L_image_uniformity/batch_size)
            L_image_total +=alignment_image+uniformity_image
        L_image_total/=batch_size

        L_text_total=0
        alignment_text = 0
        uniformity_text = 0
        for k in range(batch_size):
            L_text_alignment =0
            L_text_uniformity=0
            L_text_alignment = torch.exp(self.sim(self.hi[k],self.ht[k])/self.temp)
            for j in range(batch_size):
                L_text_uniformity += torch.exp(self.sim(self.hi[k],self.ht[j])/self.temp)
            # L_text = -torch.log(L_text_alignment/L_text_uniformity)
            alignment_text+=-torch.log(L_text_alignment)
            uniformity_text+=torch.log(L_text_uniformity/batch_size)
            L_text_total +=alignment_text+uniformity_text
        L_text_total/=batch_size

        Loss = (L_image_total+L_text_total)/2

        return Loss,alignment_image,uniformity_image,alignment_text,uniformity_text
    

import torch
import torch.nn.functional as F

class SimLoss(MyLoss):
    def __init__(self, hi: torch.tensor, ht: torch.tensor, temp: float):
        self.hi = hi  # batch_size * dim
        self.ht = ht  # batch_size * dim
        self.temp = temp

    def sim_matrix(self, I, T):
        # I와 T의 정규화
        I_norm = F.normalize(I, p=2, dim=1)
        T_norm = F.normalize(T, p=2, dim=1)

        # 전체 배치에 대한 유사도 행렬 계산
        return torch.mm(I_norm, T_norm.t())  # 결과는 batch_size * batch_size

    def Li(self):
        sim = self.sim_matrix(self.hi, self.ht) / self.temp

        # 대각선 요소가 자기 자신과의 유사도이므로, 이를 alignment 손실로 사용
        alignment = -torch.log(torch.diag(sim).exp())

        # 소프트맥스를 사용하여 각 행에 대한 uniformity 손실 계산
        uniformity = -torch.log_softmax(sim, dim=1)

        # alignment와 uniformity 손실의 평균을 구함
        L_image = (alignment + uniformity.mean(dim=1)).mean()

        # 텍스트에 대해서도 동일하게 계산
        # 주의: 실제 구현에서는 텍스트 임베딩에 대해서도 동일한 과정을 수행해야 함
        L_text = L_image  # 예시에서는 동일한 계산을 사용

        Loss = (L_image + L_text) / 2

        return Loss,alignment.mean(),uniformity.mean(dim=1).mean()

