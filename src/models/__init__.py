from .encoders import TimeSeriesTransformerEncoder, build_encoder

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

from .external_forecasting import (
    TimeXer,
    TimeMixer,
    PatchTST,
    DLinear,
    ITransformer,
    MICN,
    TimesNet,
    Mamba,
    Transformer
)


ModelClass = {
    "LSTM": LSTM,
    "TransformerBaseline": TransformerBaseline,
    "LinearRegressionBaseline": LinearRegressionBaseline,
    "MLP": MLP,
    "RegressionHead": RegressionHead,
    "CLSHead": CLSHead,
    "TransformerDecoder": TransformerDecoder,
    # "TransformerTimeMixer": TransformerTimeMixer,
    "ForecastModel": ForecastModel,
    # External baselines
    "TimeXer": TimeXer,
    "TimeMixer": TimeMixer,
    "PatchTST": PatchTST,
    "DLinear": DLinear,
    "ITransformer": ITransformer,
    "MICN": MICN,
    "TimesNet": TimesNet,
    "Mamba": Mamba,
    "Transformer": Transformer
}
