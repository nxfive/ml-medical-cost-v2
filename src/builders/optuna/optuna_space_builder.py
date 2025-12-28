from typing import Any

from src.params.optuna_grid import OptunaGrid
from src.params.optuna_updater import OptunaParamUpdater
from src.params.validator import ParamValidator


class OptunaSpaceBuilder:
    @staticmethod
    def build(
        params: dict[str, Any], optuna_params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Builds an Optuna search space by validating inputs, filling missing parameter values,
        converting them to Optuna distributions, and applying pipeline namespaces.
        """
        if optuna_params:
            ParamValidator.validate_optuna(optuna_params)

        params = OptunaParamUpdater.update(params, optuna_params)
        return OptunaGrid.create_optuna_space(params)
