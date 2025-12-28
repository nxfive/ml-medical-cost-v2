from typing import Any

import optuna
from src.builders.pipeline.pipeline_grid_builder import PipelineGridBuilder
from src.builders.transformer.wrapper_grid_builder import WrapperGridBuilder
from src.params.optuna_grid import OptunaGrid
from src.tuning.transformers.registry import TRANSFORMERS

from .optuna_space_builder import OptunaSpaceBuilder


class OptunaTrialGridBuilder:
    @staticmethod
    def _build_optuna_space_params(
        params: dict[str, Any],
        optuna_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Builds an Optuna search space from given parameters and optional 
        Optuna config.
        """
        return OptunaSpaceBuilder.build(params, optuna_params)

    @staticmethod
    def _build_trial_params(
        trial: optuna.Trial, optuna_space: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Suggests parameter values for a given Optuna trial based on the search 
        space.
        """
        return OptunaGrid.create_trial_params(trial, optuna_space)

    @staticmethod
    def _merge_and_prefix(
        model_params: dict[str, Any],
        transformer_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Merges model and transformer parameters and applies pipeline/wrapper 
        prefixes.
        """
        if transformer_params is None:
            return PipelineGridBuilder.build(model_params)
        else:
            pipeline_grid = PipelineGridBuilder.build(model_params)
            wrapper_grid = WrapperGridBuilder.build(
                param_grid=pipeline_grid, transformer_params=transformer_params
            )
            return wrapper_grid

    def build(
        self,
        trial: optuna.Trial,
        optuna_params: dict[str, Any],
        model_params: dict[str, Any],
        transformers: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Builds the full trial parameter grid.
        Includes transformer parameters only if a transformer is selected for 
        optimization.
        """
        chosen_transformer = trial.suggest_categorical(
            "transformation", list(transformers)
        )

        spec = TRANSFORMERS[chosen_transformer]
        if spec.is_identity:
            model_prefixed = self._merge_and_prefix(model_params)
            model_space = self._build_optuna_space_params(
                params=model_prefixed,
                optuna_params=optuna_params,
            )
            return self._build_trial_params(trial, model_space)

        else:
            transformer_params = transformers[chosen_transformer].params
            prefixed = self._merge_and_prefix(model_params, transformer_params)
            space = self._build_optuna_space_params(
                params=prefixed,
                optuna_params=optuna_params,
            )

            return self._build_trial_params(trial, space)
