# %%
import os
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
PREDICTIONS_DIR = PROJECT_DIR / "outputs" / "predictions"


class EvaluationMetrics:
    def __init__(self):
        pass

    @staticmethod
    def rmse(preds, targets):
        return np.sqrt(np.mean((preds - targets) ** 2))

    @staticmethod
    def iav(yearly_means):
        return np.std(yearly_means)

    def rmse_per_year(self, preds, targets, years):
        unique_years = np.unique(years)
        rmse_results = {}
        for y in unique_years:
            mask = years == y
            rmse_results[y] = self.rmse(preds[mask], targets[mask])
        return rmse_results

    def load_all_folds(self, predictions_path: Path):
        all_data = []

        year_folders = sorted(os.listdir(predictions_path))

        for year in year_folders:
            folder_path = predictions_path / year
            results_file = next(folder_path.glob("*.npz"))
            data = np.load(results_file)
            df = pd.DataFrame({key: data[key] for key in data.files})

            all_data.append(df)
        return pd.concat(all_data, ignore_index=True)

    def evaluate_all_folds(pred_dir):
        all_preds, all_targets, all_years = [], [], []

        for fold_path in glob.glob(f"{pred_dir}/fold_*/**/*.npz", recursive=True):
            data = np.load(fold_path)
            all_preds.append(data["preds"])
            all_targets.append(data["targets"])
            all_years.append(data["years"])

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        years = np.concatenate(all_years)

        # Example metric: RMSE per year
        unique_years = np.unique(years)
        for y in unique_years:
            mask = years == y
            rmse = np.sqrt(np.mean((preds[mask] - targets[mask]) ** 2))
            print(f"Year {y}: RMSE = {rmse:.4f}")

        # Compute IAV
        yearly_means = [np.mean(targets[years == y]) for y in unique_years]
        iav = np.std(yearly_means)
        print(f"Interannual variability (IAV) = {iav:.4f}")

        # Additional drought / percentile metrics can be added here


# %%

if __name__ == "__main__":
    predictions_path = PREDICTIONS_DIR / "LSTM" / "2026-04-02_17-00-43"
    evaluation = EvaluationMetrics()
    results = evaluation.load_all_folds(predictions_path)
    # %%
    print(results.head())

# %%
