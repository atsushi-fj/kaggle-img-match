import torch
from tqdm.auto import tqdm
import wandb


def train_step(model, 
               dataloader, 
               loss_fn, 
               optimizer,
               device):
    model.train()
    train_loss, rot_loss, trans_loss = 0, 0, 0
    for batch, (imgs, rotation_labels, translation_labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        rotation_labels = rotation_labels.to(device)
        translation_labels = translation_labels.to(device)
        
        rotation_output, translation_output = model(imgs)
        
        rotation_loss = loss_fn(rotation_output, rotation_labels)
        translation_loss = loss_fn(translation_output, translation_labels)
        loss = rotation_loss + translation_loss
        train_loss += loss.item()
        rot_loss += rotation_loss.item()
        trans_loss += translation_loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss = train_loss / len(dataloader)
    rot_loss = rot_loss / len(dataloader)
    trans_loss = rot_loss / len(dataloader)
    
    return train_loss, rot_loss, trans_loss


def val_step(model, 
              dataloader, 
              loss_fn,
              device):
    model.eval() 
    val_loss, val_rot_loss, val_trans_loss = 0, 0, 0
    
    with torch.inference_mode():
        for batch, (imgs, rotation_labels, translation_labels) in enumerate(dataloader):
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

    loss = loss / len(dataloader)
    rot_loss = rot_loss / len(dataloader)
    trans_loss = trans_loss / len(dataloader)
    return loss, rot_loss, trans_loss


def train(model, 
          train_dataloader, 
          test_dataloader, 
          optimizer,
          loss_fn,
          epochs,
          earlystopping,
          model_name,
          device):
 
    results = {"train_loss": [],
               "train_rot_loss": [],
               "train_trans_loss": [],
               "val_loss": [],
               "val_rot_loss": [],
               "val_trans_loss": []
    }
    
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_rot_loss, train_trans_loss = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        val_loss, val_rot_loss, val_trans_loss = val_step(model=model,
                                                          dataloader=test_dataloader,
                                                          loss_fn=loss_fn,
                                                          device=device)
        wandb.log({"Epoch": epoch+1,
                   "train_loss": train_loss,
                   "train_rot_loss": train_rot_loss,
                   "train_trans_loss": train_trans_loss,
                   "val_loss": val_loss,
                   "val_rot_loss": val_rot_loss,
                   "val_trans_loss": val_trans_loss})
        
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_rot_loss: {train_rot_loss:.4f} | "
          f"test_trans_loss: {train_trans_loss:.4f} | "
          f"val_loss: {val_loss:.4f} | "
          f"val_rot_loss: {val_rot_loss:.4f} | "
          f"val_trans_loss: {train_trans_loss:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_rot_loss"].append(train_rot_loss)
        results["train_trans_loss"].append(train_trans_loss)
        results["val_loss"].append(val_loss)
        results["val_rot_loss"].append(val_rot_loss)
        results["val_trans_loss"].append(val_trans_loss)
        
        
        earlystopping(val_loss, model, model_name)
        
        if earlystopping.early_stop: 
            print("Early Stopping!")
            break

      