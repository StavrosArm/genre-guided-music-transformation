import torch
import wandb
from tqdm import tqdm

from model.model import MusicModel
from dataset.dataset import FMADataset
from factories.dataloader import get_dataloader
from factories.loss import get_loss_function
from factories.optimizer import get_optimizer, get_scheduler

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MusicModel(config).to(device)

    train_data = FMADataset(config)
    val_data = FMADataset(config, mode = "val")

    train_loader = get_dataloader(train_data, batch_size=config.training.batch_size, num_workers=1, shuffle=True)
    val_loader = get_dataloader(val_data, batch_size=config.training.batch_size, num_workers=1, shuffle=False)

    criterion = get_loss_function(config.loss)
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_scheduler(config, optimizer)

    use_wandb = config.experiment.log == "wandb"
    if use_wandb:
        wandb.init(
            project=config.experiment.project,
            group=config.experiment.experiment_group,
            name=config.experiment.experiment_name,
            config=config.__dict__
        )

    best_val_loss = float('inf')
    global_step = 0
    epochs = config.training.epochs

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0

        train_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, batch in enumerate(train_bar):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

            if use_wandb and global_step % config.experiment.log_every_n_steps == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/step": global_step,
                    "epoch": epoch + 1
                })
            global_step += 1

        avg_train_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        val_bar = tqdm(val_loader, desc="Validating", leave=False)
        for batch in val_bar:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_bar.set_postfix(loss=loss.item())

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total


        if use_wandb:
            wandb.log({
                "val/loss": avg_val_loss,
                "val/accuracy": val_acc,
                "epoch": epoch + 1
            })

        if scheduler:
            scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{config.training.checkpoint_dir}/best_model.pt")


    if use_wandb:
        wandb.finish()
