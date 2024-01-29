from typing import Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
    def fit(self, X, y): ...
    def predict(self, X): ...
