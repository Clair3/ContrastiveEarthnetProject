from .encoders import TimeSeriesTransformerEncoder

from .forecasting import (
    LSTM,
    TransformerBaseline,
    LinearRegressionBaseline,
    MLP,
    TransformerDecoder,
    TransformerTimeMixer
)

from .probing import RegressionHead, CLSHead

ModelClass = {
    "LSTM": LSTM,
    "TransformerBaseline": TransformerBaseline,
    "LinearRegressionBaseline": LinearRegressionBaseline,
    "MLP": MLP,
    "RegressionHead": RegressionHead,
    "CLSHead": CLSHead,
    "TransformerDecoder": TransformerDecoder,
    "TransformerTimeMixer": TransformerTimeMixer,
}
