from .encoders import TimeSeriesTransformerEncoder
from .forecasting import (
    LSTM,
    PersistenceBaseline,
    SeasonalBaseline,
    TransformerForecast,
    LinearRegressionBaseline,
    MLP,
)

ModelClass = {
    "LSTM": LSTM,
    "PersistenceBaseline": PersistenceBaseline,
    "SeasonalBaseline": SeasonalBaseline,
    "TransformerForecast": TransformerForecast,
    "LinearRegressionBaseline": LinearRegressionBaseline,
    "MLP": MLP,
}
