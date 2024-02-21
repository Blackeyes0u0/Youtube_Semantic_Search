## lora.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRA_Config:
    def __init__(self, r, lora_alpha, lora_dropout, merge_weights, target_modules):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_weights = merge_weights
        self.target_modules = target_modules

class LoRALayer(nn.Module):
    def __init__(self, original_layer, config: LoRA_Config):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        input_dim = original_layer.weight.size(1)
        output_dim = original_layer.weight.size(0)

        # Initialize and then apply kaiming_uniform_
        lora_A_tensor = torch.empty(input_dim, config.r)
        torch.nn.init.kaiming_uniform_(lora_A_tensor)
        self.lora_A = nn.Parameter(lora_A_tensor)

        self.lora_B = nn.Parameter(torch.zeros(config.r, output_dim))
        self.scaling = config.lora_alpha/config.r
        if config.lora_dropout > 0:
            self.dropout = nn.Dropout(p=config.lora_dropout)
        else:
            self.dropout = lambda x: x  # No-op

    def forward(self, x):
        # Apply dropout before the matrix multiplication
        A_dropout = self.dropout(self.lora_A)
        B_dropout = self.dropout(self.lora_B)
        W_prime = self.original_layer.weight + self.scaling*A_dropout @ B_dropout
        return F.linear(x, W_prime, self.original_layer.bias)

    # def forward(self,x):
    #     delta_W = self.dropout(self.lora_B(self.lora_A(x)))
    #     W = self.original_layer(x)
    #     return self.scaling*delta_W + W
    
    def __repr__(self):
        return f'{self.__class__.__name__}(\n  (original_layer): {self.original_layer},\n  (lora_A): Parameter of size {self.lora_A.size()},\n  (lora_B): Parameter of size {self.lora_B.size()}\n)'
    
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    #for param in model.parameters():
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad: # True이면 learnable parameter에 추가
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param:.2f} %"
    )
    return trainable_params, all_param