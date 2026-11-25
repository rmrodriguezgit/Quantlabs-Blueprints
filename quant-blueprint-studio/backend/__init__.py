"""Backend del Studio."""
# macOS: xgboost y torch traen cada uno su libomp; si torch se carga primero, XGBoost hace segfault al entrenar.
# Cargar xgboost antes que torch evita el choque. Opcional: sin xgboost no pasa nada.
try:
    import xgboost  # noqa: F401
except ImportError:
    pass
