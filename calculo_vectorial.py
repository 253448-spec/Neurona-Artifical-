"""Núcleo matemático del simulador de una neurona lineal.

Contiene los cálculos numéricos del proyecto: datos sintéticos, predicción,
pérdida MSE, derivadas parciales, gradiente, derivada direccional y
descenso del gradiente. No genera figuras ni decide el orden de ejecución.
"""

from __future__ import annotations

import numpy as np


def _validar_datos(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convierte y valida las muestras usadas por las funciones numéricas."""
    try:
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("x e y deben contener únicamente valores numéricos.") from error

    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("x e y deben ser arreglos unidimensionales.")
    if x_array.size == 0:
        raise ValueError("x e y no pueden estar vacíos.")
    if x_array.shape != y_array.shape:
        raise ValueError("x e y deben contener la misma cantidad de muestras.")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("x e y no pueden contener valores NaN o infinitos.")

    return x_array, y_array


def _validar_parametros_lineales(w: float, b: float) -> None:
    """Valida los parámetros de la recta predictiva."""
    if not np.isfinite(w) or not np.isfinite(b):
        raise ValueError("w y b deben ser valores finitos.")


def _validar_h(h: float) -> float:
    """Valida el paso h usado en las diferencias finitas."""
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h debe ser un valor finito mayor que cero.")
    return float(h)


def generar_datos(
    n: int = 100,
    w_real: float = 2.5,
    b_real: float = 10.0,
    ruido: float = 5.0,
    semilla: int = 42,
    x_min: float = 0.0,
    x_max: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera datos sintéticos reproducibles para una relación lineal.

    La carga ``x`` se distribuye uniformemente y la latencia ``y`` sigue la
    relación ``w_real * x + b_real`` con ruido gaussiano. El generador local
    evita modificar el estado aleatorio global de NumPy.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n debe ser un entero positivo.")
    if isinstance(semilla, bool) or not isinstance(semilla, (int, np.integer)) or semilla < 0:
        raise ValueError("semilla debe ser un entero no negativo.")
    if not np.isfinite(ruido) or ruido < 0:
        raise ValueError("ruido debe ser un valor finito no negativo.")
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError("El rango de x debe ser finito y cumplir x_max > x_min.")
    _validar_parametros_lineales(w_real, b_real)

    generador = np.random.default_rng(semilla)
    x = generador.uniform(x_min, x_max, size=int(n))
    errores = generador.normal(loc=0.0, scale=ruido, size=int(n))
    y = w_real * x + b_real + errores
    return x, y


def predecir(x: np.ndarray | list[float], w: float, b: float) -> np.ndarray:
    """Calcula la predicción ``y_hat = w * x + b``."""
    _validar_parametros_lineales(w, b)
    try:
        x_array = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("x debe contener únicamente valores numéricos.") from error
    if not np.all(np.isfinite(x_array)):
        raise ValueError("x no puede contener valores NaN o infinitos.")
    return w * x_array + b


def _perdida_mse(
    x_array: np.ndarray,
    y_array: np.ndarray,
    w: float,
    b: float,
) -> float:
    """Calcula el MSE sobre arreglos ya validados."""
    residuos = y_array - (w * x_array + b)
    return float(np.mean(residuos**2))


def perdida_mse(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    w: float,
    b: float,
) -> float:
    """Calcula el error cuadrático medio de la neurona lineal."""
    x_array, y_array = _validar_datos(x, y)
    _validar_parametros_lineales(w, b)
    return _perdida_mse(x_array, y_array, w, b)
