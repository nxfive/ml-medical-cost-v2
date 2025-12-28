from typing import Any

LARGE_RANGE_THRESHOLD = 100
DEFAULT_LARGE_RANGE_STEP = 5
DEFAULT_SMALL_RANGE_STEP = 1


class OptunaParamUpdater:
    @staticmethod
    def update(
        params: dict[str, Any],
        optuna_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Updates model parameters with optimized values from Optuna results.

        - For missing numeric parameters: creates min/max/step
        - For missing categorical parameters: creates choices
        """
        optuna_params = optuna_params or {}
        updated = {}

        for full_key, values in params.items():
            base_param = full_key.split("__")[-1]

            if base_param in optuna_params:
                updated[full_key] = optuna_params[base_param]

            else:
                first_value = values[0]
                if isinstance(first_value, (bool, str)):
                    updated[full_key] = {"choices": values}

                elif isinstance(first_value, (int, float)):
                    min_val, max_val = min(values), max(values)
                    step = (
                        DEFAULT_LARGE_RANGE_STEP
                        if (max_val - min_val) > LARGE_RANGE_THRESHOLD
                        else DEFAULT_SMALL_RANGE_STEP
                    )
                    updated[full_key] = {"min": min_val, "max": max_val, "step": step}
                else:
                    raise ValueError(
                        f"Unsupported parameter type for '{full_key}': {type(first_value)}"
                    )

        return updated
