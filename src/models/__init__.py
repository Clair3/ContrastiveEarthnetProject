from .encoders import TimeSeriesTransformerEncoder
from .forecasting import (
    LSTM,
    PersistenceBaseline,
    SeasonalBaseline,
    TransformerForecast,
)

ModelClass = {
    "LSTM": LSTM,
    "PersistenceBaseline": PersistenceBaseline,
}
