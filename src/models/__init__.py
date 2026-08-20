from .encoders import TimeSeriesTransformerEncoder, build_encoder
from .revin import RevIN

from .forecasting import (
    LSTM,
    TransformerBaseline,
    LinearRegressionBaseline,
    MLP,
    TransformerDecoder,
    TransformerTimeMixer,
    ForecastModel,
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
    "ForecastModel": ForecastModel,
}
