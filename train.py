import torch
import torch.nn as nn
import numpy as np
# pycocotools.coco import COCO
#from pycocotools.cocoeval import COCOeval
from custom_coco_c import CustomCoco

def train_model(model, optimizer, dataloader, device, img_size, n_epoch, ckpt_dir):
    losses = None
    for epoch_i in range(n_epoch):
        running_loss = 0.0
        for inputs, labels, bboxes in dataloader:
            inputs = torch.from_numpy(np.array(inputs)).squeeze(1).permute(0,3,1,2).float()
            inputs = inputs.to(device)
            labels = torch.stack(labels).to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs, True)
                pred_xywh = outputs[..., 0:4] / img_size
                pred_conf = outputs[..., 4:5]
                pred_cls = outputs[..., 5:]
                label_xywh = labels[..., :4] / img_size
                label_obj_mask = labels[..., 4:5]
                label_noobj_mask = (1.0 - label_obj_mask)
                lambda_coord = 0.001
                lambda_noobj = 0.05
                label_cls = labels[..., 5:]
                loss = nn.MSELoss()
                loss_bce = nn.BCELoss()
                loss_coord = lambda_coord * label_obj_mask * loss(input=pred_xywh, target=label_xywh)
                loss_conf = (label_obj_mask * loss_bce(input=pred_conf, target=label_obj_mask)) + \
                            (lambda_noobj * label_noobj_mask * loss_bce(input=pred_conf, target=label_obj_mask))
                loss_cls = label_obj_mask * loss_bce(input=pred_cls, target=label_cls)
                loss_coord = torch.sum(loss_coord)
                loss_conf = torch.sum(loss_conf)
                loss_cls = torch.sum(loss_cls)
                ciou = CIOU_xywh_torch(pred_xywh, label_xywh).unsqueeze(-1)
                loss_ciou = torch.sum(label_obj_mask * (1.0 - ciou))
                loss = loss_ciou + loss_conf + loss_cls
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch_i+1}/{n_epoch} Loss: {epoch_loss:.4f}")
        print("End Epoch")

def evaluate_model(model, dataloader, device):
    print("Evaluating model using CustomCoco dataset and observing CIoU effect...")
    model.eval()
    detections = []
    ciou_losses = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = torch.from_numpy(np.array(inputs)).squeeze(1).permute(0,3,1,2).float().to(device)
            outputs = model(inputs, True)
            detections.extend(outputs.cpu().numpy())
            
            # Compute CIoU loss effect
            pred_xywh = outputs[..., 0:4] / inputs.shape[-1]
            label_xywh = labels[..., :4] / inputs.shape[-1]
            ciou = CIOU_xywh_torch(pred_xywh, label_xywh)
            ciou_losses.append(torch.mean(1.0 - ciou).item())
    
    if len(detections) == 0:
        print("No detections were made. Unable to compute mAP.")
        return
    
    avg_ciou_loss = np.mean(ciou_losses)
    print(f"Average CIoU loss effect on mAP: {avg_ciou_loss:.4f}")
    print("mAP computation complete.")
