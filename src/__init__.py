from data import ContrastiveDataModule, ForecastingDataModule
from src.modules.contrastive import ContrastiveTrainingModule
from src.modules.forecasting import ForecastingTrainModule
from models import TimeSeriesTransformerEncoder, ModelClass

from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
CONFIG_DIR = PROJECT_DIR / "configs"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
