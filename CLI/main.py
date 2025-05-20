import typer
from src.train import train_model
from src.inference import run_inference
from src.evaluate import evaluate_model
from src.logger import get_logger
import pandas as pd

app = typer.Typer()
logger = get_logger()

@app.command()
def train(
    data_path: str = typer.Argument(..., help="Path to training CSV file"),
    model_out: str = typer.Option("models/lstm.pt", help="Where to save trained model")
):
    """Train the LSTM model."""
    logger.info("Starting training...")
    train_model(data_path, model_out)
    logger.info(f"Model saved to {model_out}")

@app.command()
def predict(
    track_path: str = typer.Argument(..., help="Path to input track CSV"),
    model_path: str = typer.Option("models/lstm.pt", help="Path to trained model"),
    scaler_path: str = typer.Option("models/scaler.pkl", help="Scaler file"),
):
    """Run inference on a missile track."""
    df = pd.read_csv(track_path)
    prediction = run_inference(df, model_path, scaler_path)
    logger.info("Prediction complete.")
    typer.echo(prediction.tolist())

@app.command()
def evaluate(
    data_path: str = typer.Argument(..., help="Path to validation set"),
    model_path: str = typer.Option("models/lstm.pt", help="Model file"),
    scaler_path: str = typer.Option("models/scaler.pkl", help="Scaler file"),
):
    """Evaluate model on validation set."""
    logger.info("Evaluating model...")
    metrics = evaluate_model(data_path, model_path, scaler_path)
    typer.echo(metrics)

@app.command()
def version():
    """Show CLI version."""
    typer.echo("Missile Classifier CLI v0.1")

if __name__ == "__main__":
    app()
