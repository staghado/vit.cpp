import torch
import timm

def save_timm_model():
    # Load the pretrained model
    model_name = 'vit_base_patch8_224.augreg2_in21k_ft_in1k'
    timm_model = timm.create_model(model_name, pretrained=True)

    # Save the model to disk
    torch.save(timm_model.state_dict(), 'vit_base.pth')

save_timm_model()