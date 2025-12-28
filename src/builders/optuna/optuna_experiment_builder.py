import optuna
from src.builders.pipeline.pipeline_builder import PipelineBuilder
from src.containers.experiment import ExperimentSetup
from src.dto.config import OptunaExperimentConfig

from .optuna_grid_distribution_builder import OptunaGridDistributionBuilder
from .optuna_trial_grid_builder import OptunaTrialGridBuilder


class OptunaExperimentBuilder:
    @staticmethod
    def build(
        cfg: OptunaExperimentConfig, trial: optuna.Trial | None = None
    ) -> ExperimentSetup:
        """
        Builds an ExperimentSetup, including the pipeline and parameter space.

        If a trial is provided, samples parameters for that trial.
        Otherwise, builds the full Optuna search space.
        """
        params = (
            OptunaTrialGridBuilder().build(
                trial=trial,
                optuna_params=cfg.optuna_model_config.params,
                model_params=cfg.model.params,
                transformers=cfg.transformers.to_dict(),
            )
            if trial is not None
            else OptunaGridDistributionBuilder.build(
                optuna_params=cfg.optuna_model_config.params,
                model_params=cfg.model.params,
            )
        )

        build_kwargs = {
            "model_cfg": cfg.model,
            "features_cfg": cfg.features,
        }

        if trial is not None:
            build_kwargs["transformation"] = trial.params.get("transformation")

        pipeline = PipelineBuilder.build(**build_kwargs)
        return ExperimentSetup(pipeline, params)
