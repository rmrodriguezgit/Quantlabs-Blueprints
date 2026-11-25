"""Carga xgboost antes que torch en todos los tests (ver backend/__init__.py)."""
# macOS: xgboost y torch traen cada uno su libomp; si torch se carga primero, XGBoost hace segfault al entrenar.
# Cargar xgboost antes que torch evita el choque. Opcional: sin xgboost no pasa nada.
try:
    import xgboost  # noqa: F401
except ImportError:
    pass
