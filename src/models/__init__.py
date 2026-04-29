from .encoders import (
    TimeSeriesTransformerEncoder,
    ContrastiveTransformer,
    ContrastiveHead,
)

from .forecasting import (
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
    "LinearRegressionBaseline": LinearRegressionBaseline,
    "MLP": MLP,
    # "TransformerContrastive": ContrastiveTransformer,
}
