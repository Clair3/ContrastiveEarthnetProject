import torch
import torch.nn as nn


class AnalogForecast(nn.Module):
    """
    Simple nearest-neighbour / analogue forecasting baseline.

    The analogue database is built from training samples only.

    Query:
        historical EVI
        historical weather
        location
        day-of-year

    Prediction:
        weighted average of the future EVI trajectories
        of the K nearest historical analogues.
    """

    def __init__(
        self,
        context_length,
        prediction_length,
        k=5,
        mode="evi",
        weather_weight=1.0,
        location_weight=0.0,
        season_weight=0.0,
    ):
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.k = k

        self.mode = mode
        self.weather_weight = weather_weight
        self.location_weight = location_weight
        self.season_weight = season_weight

        # Filled from the TRAINING set.
        self.register_buffer(
            "database_evi",
            torch.empty(0),
            persistent=False,
        )

        self.register_buffer(
            "database_weather",
            torch.empty(0),
            persistent=False,
        )

        self.register_buffer(
            "database_future",
            torch.empty(0),
            persistent=False,
        )

        self.register_buffer(
            "database_location",
            torch.empty(0),
            persistent=False,
        )

        self.register_buffer(
            "database_doy",
            torch.empty(0),
            persistent=False,
        )

    @torch.no_grad()
    def fit_database(self, dataset):
        """
        Build the analogue database from TRAINING samples.

        dataset should return:

            vegetation_history [N,T,1]
            weather_history     [N,T,W]
            vegetation_forecast [N,H,1]
            location            [N,2]
            hist_doy             [N,T]
        """

        evi = []
        weather = []
        future = []
        locations = []
        doys = []

        for i in range(len(dataset)):

            sample = dataset[i]

            evi.append(sample["vegetation_history"])

            weather.append(sample["weather_history"])

            future.append(sample["vegetation_forecast"])

            locations.append(
                torch.tensor(
                    sample["location"],
                    dtype=torch.float32,
                )
            )

            doys.append(sample["hist_doy"])

        self.database_evi = torch.stack(evi)
        self.database_weather = torch.stack(weather)
        self.database_future = torch.stack(future)
        self.database_location = torch.stack(locations)
        self.database_doy = torch.stack(doys)

    def _normalize_distance(self, x, database):

        # Flatten temporal/channel dimensions.
        x = x.reshape(x.shape[0], -1)
        database = database.reshape(database.shape[0], -1)

        # Per-query standardization prevents absolute EVI level
        # from completely dominating trajectory shape.
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)

        database_mean = database.mean(
            dim=-1,
            keepdim=True,
        )
        database_std = database.std(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        x = (x - x_mean) / x_std
        database = (database - database_mean) / database_std

        return torch.cdist(
            x,
            database,
            p=2,
        )

    def _distance(self, batch):

        query_evi = batch["vegetation_history"]
        query_weather = batch["weather_history"]

        # ---------------------------------------------------------
        # EVI trajectory similarity
        # ---------------------------------------------------------

        d_evi = self._normalize_distance(
            query_evi,
            self.database_evi,
        )

        distance = d_evi

        # ---------------------------------------------------------
        # Weather trajectory similarity
        # ---------------------------------------------------------

        if self.mode in (
            "evi_weather",
            "multimodal",
        ):

            d_weather = self._normalize_distance(
                query_weather,
                self.database_weather,
            )

            distance = distance + self.weather_weight * d_weather

        # ---------------------------------------------------------
        # Location similarity
        # ---------------------------------------------------------

        if self.location_weight > 0:

            query_location = torch.as_tensor(
                batch["location"],
                device=query_evi.device,
                dtype=torch.float32,
            )

            d_location = torch.cdist(
                query_location,
                self.database_location,
            )

            distance = distance + self.location_weight * d_location

        # ---------------------------------------------------------
        # Seasonal similarity
        # ---------------------------------------------------------

        if self.season_weight > 0:

            query_doy = batch["hist_doy"][:, -1].float()

            database_doy = self.database_doy[:, -1].float()

            # Circular distance on the annual cycle.
            d = torch.abs(query_doy[:, None] - database_doy[None, :])

            d_season = torch.minimum(
                d,
                365.0 - d,
            )

            distance = distance + self.season_weight * d_season

        return distance

    @torch.no_grad()
    def forward(self, batch):

        if self.database_evi.numel() == 0:
            raise RuntimeError(
                "Analogue database is empty. "
                "Call fit_database() using the training dataset first."
            )

        distances = self._distance(batch)

        k = min(
            self.k,
            self.database_future.shape[0],
        )

        nearest_distance, nearest_idx = torch.topk(
            distances,
            k=k,
            dim=1,
            largest=False,
        )

        # Soft weighting.
        weights = torch.softmax(
            -nearest_distance,
            dim=1,
        )

        retrieved_future = self.database_future[nearest_idx]

        prediction = (retrieved_future * weights[..., None, None]).sum(dim=1)

        return prediction
