import click
from config import Config
from trainer.train import train


@click.group()
def cli():
    pass

@cli.command(name = "train")
@click.option('--config_path', required=True, help='Path to the configuration file.')
def train(path: str):
    """ Initiates training procedure given the configuration file """
    click.echo("Loading configuration ...")
    config = Config(path)
    click.echo("Initiating training procedure ...")
    train(config)
    click.echo("Training procedure complete.")

@cli.command(name = "test")
@click.option('--config_path', required=True, help='Path to the configuration file.')
def test(path: str):
    """ Inferences the model on the test set and logs them to wandb. """
    click.echo("Loading configuration ...")
    config = Config(path)
    click.echo("Initiating testing procedure ...")
    train(config)
    click.echo("Testing procedure complete.")

@cli.command(name = "Inference sample")
@click.option('--model_path', required=True, help='Path to the model')
@click.option('--song', required=True, help='Path to the song sample.')
def predict(model, sample):
    """Make a prediction with the trained model."""
    click.echo("Initiating inference procedure ...")
    click.echo("Inference on song sample complete.")


if __name__ == '__main__':
    cli()
