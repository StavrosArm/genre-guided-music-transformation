import click
from config import Config
from trainer.train import train
from inference.test import test
from inference.song_distortion import distort_songs
from inference.predict import predict_song


@click.group()
def cli():
    pass

@cli.command(name = "initiate_training")
@click.option('--config_path', required=True, help='Path to the configuration file.', default="config.yaml")
def initiate_training(config_path: str):
    """ Initiates training procedure given the configuration file """
    click.echo("Loading configuration ...")
    config = Config(config_path)
    click.echo("Initiating training procedure ...")
    train(config)
    click.echo("Training procedure complete.")

@cli.command(name = "initiate_testing")
@click.option('--config_path', required=True, help='Path to the configuration file.')
def initiate_testing(path: str):
    """ Inferences the model on the test set and logs them to wandb. """
    click.echo("Loading configuration ...")
    config = Config(path)
    click.echo("Initiating testing procedure ...")
    test(config)
    click.echo("Testing procedure complete.")

@cli.command(name = "inference_song")
@click.option('--song', required=True, help='Path to the song sample.')
def predict(song: str):
    """Make a prediction with the trained model."""
    click.echo("Initiating inference procedure ...")
    prediction = predict_song(song)
    click.echo(prediction)
    click.echo("Inference on song sample complete.")

@cli.command(name = "distort_songs")
@click.option('--songs', required=True, help='Path to the song samples.')
def distort_songs(songs: str):
    """Distort a song to match rock genre."""
    click.echo("Initiating song's genre distortion procedure ...")
    distort_songs(songs)
    click.echo("Songs genre distortion complete.")


if __name__ == '__main__':
    cli()
