def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def main(config_path):

    config = load_config(config_path)
    data_config = load_config(config.data_config)

    seed_everything(config.seed, workers=True)

    task_cls = TASKS[config.task]

    task = task_cls(
        config=config,
        data_config=data_config,
    )

    runner = ExperimentRunner(
        model=task.build_model(),
        datamodule=task.build_datamodule(),
        config=config,
        logger=build_logger(config),
    )

    best_ckpt = runner.fit()

    runner.test(best_ckpt)