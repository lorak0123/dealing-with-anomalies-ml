from itertools import product

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from prediction_system.utils import Model

KNN_REGRESSOR = KNeighborsRegressor(n_neighbors=10)
DECISION_TREE_REGRESSOR = DecisionTreeRegressor(max_depth=50)
SVR_REGRESSOR = SVR(kernel='poly')
RANDOM_FOREST_REGRESSOR = RandomForestRegressor(n_estimators=50)

ALL_MODELS = [
    KNN_REGRESSOR,
    DECISION_TREE_REGRESSOR,
    SVR_REGRESSOR,
    RANDOM_FOREST_REGRESSOR,
]

RANDOM_FOREST_REGRESSOR_MODELS_EVALUATION = []

params_config = {
    'n_estimators': list(range(10, 101, 20)),
    'criterion': ['squared_error', 'absolute_error'],
    'max_depth': list(range(10, 101, 20)) + [None],
    'max_features': [None, 'sqrt', 'log2'],
    'bootstrap': [True, False],
}
for params in [dict(zip(params_config.keys(), values)) for values in product(*params_config.values())]:
    cls = type(
        f'RandomForestRegressor_{"-".join(f"{k}={v}" for k, v in params.items())}',
        (RandomForestRegressor,),
        {}
    )
    RANDOM_FOREST_REGRESSOR_MODELS_EVALUATION.append(cls(**params))


BASE_MODEL = RandomForestRegressor(
    n_estimators=90,
    criterion='squared_error',
    max_depth=30,
    max_features='sqrt',
    bootstrap=False,
)


def get_model_by_name(model_name: str) -> list[Model]:
    """
    Returns model by its name

    Args:
        model_name: Name of the model

    Returns:
        Model or list of models

    Raises:
        ValueError: If model is not defined or is not a subclass of Model
    """
    model = globals().get(model_name.upper())
    if model is None:
        raise ValueError(f'Model {model_name} is not defined.')

    if isinstance(model, list) and all(isinstance(m, Model) for m in model):
        return model
    elif isinstance(model, Model):
        return [model]

    raise ValueError(f'Model {model_name} must be a subclass of Model. Got {type(model)} instead.')
