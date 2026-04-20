from .encoders import TimeSeriesTransformerEncoder

# from .contrasting import ContrastiveTransformer
from .forecasting import (
    ContextFormerForecast,
    LSTM,
    PersistenceBaseline,
    SeasonalBaseline,
    TransformerBaseline,
    LinearRegressionBaseline,
    MLP,
)

ModelClass = {
    "LSTM": LSTM,
    "PersistenceBaseline": PersistenceBaseline,
    "SeasonalBaseline": SeasonalBaseline,
    "TransformerBaseline": TransformerBaseline,
    "ContextFormerForecast": ContextFormerForecast,
    "LinearRegressionBaseline": LinearRegressionBaseline,
    "MLP": MLP,
    # "TransformerContrastive": ContrastiveTransformer,
}
