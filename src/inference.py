import torch
from tqdm.auto import tqdm

def eval_model(model,
               data_loader,
               loss_fn,
               device):
    """Returns a dictionary containing the results of model predicting on data_loader."""
    loss, rot_loss, trans_loss = 0, 0
    model.eval()
    with torch.inference_mode():
        for imgs, rotation_labels, translation_labels in tqdm(data_loader):
            imgs = imgs.to(device)
            rotation_labels = rotation_labels.to(device)
            translation_labels = translation_labels.to(device)
            
            rotation_output, translation_output = model(imgs)
            
            rotation_loss = loss_fn(rotation_output, rotation_labels)
            translation_loss = loss_fn(translation_output, translation_labels)
            loss = rotation_loss + translation_loss
            loss += loss.item()
            rot_loss += rotation_loss.item()
            trans_loss += translation_loss.item()
            
        loss /= len(data_loader)
        rot_loss /= len(data_loader)
        trans_loss /= len(data_loader)

    return {"model_name": model.__class__.__name__, 
            "model_loss": loss.item(),
            "model_rot_loss": rot_loss.item(),
            "model_trans_loss": trans_loss.item()}
    