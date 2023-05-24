import numpy as np
import torch
import wandb
import argparse

from engine import train
from utils import load_config, create_display_name, seed_everything, EarlyStopping
from dataset import get_datasets, get_train_validation_set, get_train_dataloader, get_validation_dataloader, get_train_dataset, get_validation_dataset
from model.baseline import BaseCNN
from inference import eval_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kaggle image matching contest")
    parser.add_argument("-config", type=str, default="config.yaml",
                        help="Set config file")
    parser.add_argument("-model", type=object, default=BaseCNN(),
                        help="Set model")
    parser.add_argument("-extra", type=str, default=None,
                        help="Set extra info")
    args = parser.parse_args()
    
    cfg = load_config(file=args.config)
    name = create_display_name(experiment_name=cfg.experiment_name,
                               model_name=cfg.model_name,
                               extra=args.extra)
    
    run = wandb.init(project=cfg.project,
                     name=name,
                     config=cfg)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    seed_everything(seed=cfg.seed)
    
    train_data, val_data = get_train_validation_set(
        get_datasets(cfg.train_path, cfg.train_label_file))
    
    train_dataloader = get_train_dataloader(
        get_train_dataset(cfg.train_path, train_data))
    
    val_dataloader = get_validation_dataloader(
        get_validation_dataloader(cfg.train_path, val_data))
    
    
    # Training model
    args.model.to(device)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(args.model.parameters(), lr=cfg.lr)
    earlystopping = EarlyStopping(patience=cfg.patience, verbose=True)
    
    
    train(args.model, train_dataloader, val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=cfg.epochs,
        earlystopping=earlystopping,
        model_name=cfg.model_path,
        device=device)
    
    args.model.load_state_dict(torch.load(f=cfg.load_model_path))
    args.model.to(device)
    
    result = eval_model(model=args.model,
                        data_loader=val_dataloader,
                        loss_fn=loss_fn,
                        device=device)
    
    print(f"\n{result}")

