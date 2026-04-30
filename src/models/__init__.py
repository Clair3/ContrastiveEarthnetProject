from .encoders import TimeSeriesTransformerEncoder

from .forecasting import (
    LSTM,
    TransformerBaseline,
    LinearRegressionBaseline,
    MLP,
)

from .probing import RegressionHead, CLSHead

ModelClass = {
    "LSTM": LSTM,
    "TransformerBaseline": TransformerBaseline,
    "LinearRegressionBaseline": LinearRegressionBaseline,
    "MLP": MLP,
    "RegressionHead": RegressionHead,
    "CLSHead": CLSHead,
}
