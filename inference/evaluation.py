import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import wandb

from model.model import MusicModel
from dataset.dataset import FMADataset
from factories.dataloader import get_dataloader
from utils.seed import set_seed
from utils.fad_distance import calculate_fad_distance
from model.maest import Maest


def evaluate_music_quality(config):
    """
    Takes as input a csv files containing the paths of distorted songs and real rock genre songs.
    Inferences MAEST to produce music embeddings and calculates the Frechet Audio Distance between them.

    :param config: The configuration file.
    :return: The FAD of real music embeddings and distorted music embeddings.
    """
    embeddings_model = Maest(config)
    real_embeddings, distorted_embeddings = embeddings_model.get_embeddings_for_fad_distance("fma_small/040/040845.mp3", "distortion/040845.mp3")
    fad_distance = calculate_fad_distance(real_embeddings, distorted_embeddings)
    with open(config.distortion.fad_txt, "a") as f:
        f.write(f"FAD Distance: {fad_distance:.6f}\n")


def test(config):
    """
    Initiates testing procedure.

    :param config: The configuration file.
    :return: All metrics together with the predictions of the model.
    """
    set_seed(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = FMADataset(config, mode="val")
    test_loader = get_dataloader(test_data, batch_size=config.training.batch_size, num_workers=1, shuffle=False)

    model = MusicModel(config).to(device)
    checkpoint_path = f"{config.training.checkpoint_dir}/best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    test_bar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for batch in test_bar:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)

    if config.experiment.log == "wandb":
        wandb.init(
            project=config.experiment.project,
            group=config.experiment.experiment_group,
            name=f"{config.experiment.experiment_name}_test",
            config=config.__dict__,
        )
        wandb.log({
            "test/accuracy": acc,
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1,
            "test/cls_report": report
        })
        wandb.finish()

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "y_true": all_labels,
        "y_pred": all_preds,
    }
