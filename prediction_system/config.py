from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from prediction_system.utils import Model

KNN_REGRESSOR = KNeighborsRegressor(n_neighbors=10)
DECISION_TREE_REGRESSOR = DecisionTreeRegressor(max_depth=50)
SVR_REGRESSOR = SVR(kernel='poly')
RANDOM_FOREST_REGRESSOR = RandomForestRegressor(n_estimators=50)


BASE_MODEL = RANDOM_FOREST_REGRESSOR

ALL_MODELS = [
    KNN_REGRESSOR,
    DECISION_TREE_REGRESSOR,
    SVR_REGRESSOR,
    RANDOM_FOREST_REGRESSOR,
]


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
