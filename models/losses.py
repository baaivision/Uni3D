import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

class Uni3d_Text_Image_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs, masks):
        pc_embed = outputs['pc_embed']
        text_embed = outputs['text_embed']
        image_embed = outputs['image_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=pc_embed.device
            )
            self.last_local_batch_size = local_batch_size

        masks = masks.to(pc_embed.device)

        # normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # gather features from all GPUs
        pc_embed_all, text_embed_all, image_embed_all, masks_all = \
            utils.all_gather_batch([pc_embed, text_embed, image_embed, masks])

        # cosine similarity as logits
        logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
        logits_per_pc_image = logit_scale * pc_embed @ image_embed_all.t()
        logits_per_image_pc = logit_scale * image_embed @ pc_embed_all.t()

        loss_text = (F.cross_entropy(logits_per_pc_text, self.labels) + \
                F.cross_entropy(logits_per_text_pc, self.labels)) / 2 
        
        masks = masks.bool()
        masks = ~masks

        self.labels_c = self.labels.clone()
        self.labels_c[masks] = -100

        loss_image = (F.cross_entropy(logits_per_pc_image, self.labels_c, ignore_index=-100) +\
                        F.cross_entropy(logits_per_image_pc, self.labels_c, ignore_index=-100)) / 2
        
        loss = loss_text + loss_image



        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_pc_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_text_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_pc_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_image_acc = 100 * correct / local_batch_size

        return {'loss': loss, 'uni3d_loss': loss, 'pc_image_acc': pc_image_acc, 'pc_text_acc': pc_text_acc}

  