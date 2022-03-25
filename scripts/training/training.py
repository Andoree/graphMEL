import logging
import os

import torch

from graphmel.scripts.utils.io import update_log_file


def train_model(model, train_epoch_fn, val_epoch_fn, chkpnt_path: str, train_loader, val_loader, learning_rate: float, num_epochs: int,
                output_dir: str, save_chkpnt_epoch_interval: int, device: torch.device):
    if chkpnt_path is not None:
        logging.info(f"Successfully loaded checkpoint from: {chkpnt_path}")
        checkpoint = torch.load(chkpnt_path)
        optimizer = checkpoint["optimizer"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state"])
    else:
        start_epoch = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log_file_path = os.path.join(output_dir, "training_log.txt")

    train_loss_history = []
    val_loss_history = []
    logging.info("Starting training process....")
    global_num_steps = 0
    for i in range(start_epoch, start_epoch + num_epochs):
        epoch_train_loss, num_steps = train_epoch_fn(model=model, train_loader=train_loader, optimizer=optimizer,
                                                            device=device)
        global_num_steps += num_steps
        epoch_val_loss_1 = val_epoch_fn(model=model, val_loader=val_loader, device=device)
        epoch_val_loss_2 = val_epoch_fn(model=model, val_loader=val_loader, device=device)
        # assert epoch_val_loss_1 == epoch_val_loss_2
        log_dict = {"epoch": i, "train loss": epoch_train_loss, "val loss 1": epoch_val_loss_1,
                    "val loss 2": epoch_val_loss_2}
        logging.info(', '.join((f"{k}: {v}" for k, v in log_dict.items())))

        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss_1)
        if i % save_chkpnt_epoch_interval == 0:
            checkpoint = {
                'epoch': i + 1,
                'model_state': model.state_dict(),
                'optimizer': optimizer,
            }

            chkpnt_path = os.path.join(output_dir, f"checkpoint_e_{i}_steps_{global_num_steps}.pth")
            torch.save(checkpoint, chkpnt_path)

        update_log_file(path=log_file_path, dict_to_log=log_dict)