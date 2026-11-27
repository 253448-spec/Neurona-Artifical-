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


def derivada_parcial_w(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    w: float,
    b: float,
    h: float,
) -> float:
    """Aproxima dL/dw con diferencias finitas centradas."""
    x_array, y_array = _validar_datos(x, y)
    _validar_parametros_lineales(w, b)
    paso = _validar_h(h)
    return float(
        (
            _perdida_mse(x_array, y_array, w + paso, b)
            - _perdida_mse(x_array, y_array, w - paso, b)
        )
        / (2 * paso)
    )


def derivada_parcial_b(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    w: float,
    b: float,
    h: float,
) -> float:
    """Aproxima dL/db con diferencias finitas centradas."""
    x_array, y_array = _validar_datos(x, y)
    _validar_parametros_lineales(w, b)
    paso = _validar_h(h)
    return float(
        (
            _perdida_mse(x_array, y_array, w, b + paso)
            - _perdida_mse(x_array, y_array, w, b - paso)
        )
        / (2 * paso)
    )


def gradiente(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    w: float,
    b: float,
    h: float,
) -> np.ndarray:
    """Construye el gradiente numerico (dL/dw, dL/db)."""
    x_array, y_array = _validar_datos(x, y)
    _validar_parametros_lineales(w, b)
    paso = _validar_h(h)
    derivada_w = (
        _perdida_mse(x_array, y_array, w + paso, b)
        - _perdida_mse(x_array, y_array, w - paso, b)
    ) / (2 * paso)
    derivada_b = (
        _perdida_mse(x_array, y_array, w, b + paso)
        - _perdida_mse(x_array, y_array, w, b - paso)
    ) / (2 * paso)
    return np.array([float(derivada_w), float(derivada_b)])


def gradiente_analitico(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    w: float,
    b: float,
) -> np.ndarray:
    """Calcula el gradiente exacto para comparar con el numerico."""
    x_array, y_array = _validar_datos(x, y)
    _validar_parametros_lineales(w, b)
    residuos = y_array - (w * x_array + b)
    n = x_array.size
    derivada_w = float(-2.0 * np.sum(x_array * residuos) / n)
    derivada_b = float(-2.0 * np.sum(residuos) / n)
    return np.array([derivada_w, derivada_b])


def normalizar_direccion(v: np.ndarray | list[float]) -> np.ndarray:
    """Normaliza un vector de direccion y rechaza el vector cero."""
    try:
        vector = np.asarray(v, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("v debe contener únicamente valores numéricos.") from error

    if vector.ndim != 1 or vector.size != 2:
        raise ValueError("v debe ser un vector de dos componentes (v_w, v_b).")
    if not np.all(np.isfinite(vector)):
        raise ValueError("v no puede contener valores NaN o infinitos.")

    norma = float(np.linalg.norm(vector))
    if norma == 0.0:
        raise ValueError("v no puede ser el vector (0, 0) porque no se puede normalizar.")

    return vector / norma


def derivada_direccional(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    w: float,
    b: float,
    h: float,
    v: np.ndarray | list[float],
) -> dict[str, object]:
    """Calcula u, el gradiente y la derivada direccional D_u L."""
    vector_unitario = normalizar_direccion(v)
    vector_gradiente = gradiente(x, y, w, b, h)
    derivada = float(np.dot(vector_gradiente, vector_unitario))
    return {
        "u": vector_unitario,
        "gradiente": vector_gradiente,
        "derivada": derivada,
        "norma_gradiente": float(np.linalg.norm(vector_gradiente)),
    }


def comprobar_teorema(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    w: float,
    b: float,
    h: float,
    semilla: int = 42,
    cantidad: int = 100,
    tolerancia: float = 1e-8,
) -> dict[str, object]:
    """Verifica -||gradiente|| <= D_u L <= ||gradiente|| con vectores aleatorios."""
    if isinstance(cantidad, bool) or not isinstance(cantidad, (int, np.integer)) or cantidad <= 0:
        raise ValueError("cantidad debe ser un entero positivo.")
    if isinstance(semilla, bool) or not isinstance(semilla, (int, np.integer)) or semilla < 0:
        raise ValueError("semilla debe ser un entero no negativo.")
    if not np.isfinite(tolerancia) or tolerancia < 0:
        raise ValueError("tolerancia debe ser un valor finito no negativo.")

    vector_gradiente = gradiente(x, y, w, b, h)
    norma = float(np.linalg.norm(vector_gradiente))
    generador = np.random.default_rng(semilla)

    casos: list[dict[str, float]] = []
    cumple = True
    for _ in range(int(cantidad)):
        vector = generador.normal(size=2)
        vector_unitario = vector / np.linalg.norm(vector)
        derivada = float(np.dot(vector_gradiente, vector_unitario))
        dentro = -norma - tolerancia <= derivada <= norma + tolerancia
        cumple = cumple and dentro
        casos.append(
            {
                "derivada": derivada,
                "limite_inferior": -norma,
                "limite_superior": norma,
                "cumple": bool(dentro),
            }
        )

    return {
        "cumple": bool(cumple),
        "norma_gradiente": norma,
        "cantidad": int(cantidad),
        "casos": casos,
    }


def _validar_alpha(alpha: float) -> float:
    """Valida la tasa de aprendizaje."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha debe ser un valor finito mayor que cero.")
    return float(alpha)


def _validar_tolerancia(tolerancia: float) -> float:
    """Valida la tolerancia del criterio de paro."""
    if not np.isfinite(tolerancia) or tolerancia <= 0:
        raise ValueError("tolerancia debe ser un valor finito mayor que cero.")
    return float(tolerancia)


def _validar_max_iter(max_iter: int) -> int:
    """Valida el limite de iteraciones."""
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter debe ser un entero positivo.")
    return int(max_iter)


def descenso_gradiente(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    w_inicial: float,
    b_inicial: float,
    alpha: float,
    h: float,
    tolerancia: float,
    max_iter: int,
    factor_divergecia: float = 1e6,
) -> dict[str, object]:
    """Entrena la neurona con descenso del gradiente y registra el historial.

    El paro ocurre por tolerancia del gradiente, por maximo de iteraciones o
    cuando la perdida crece demasiado, lo que indica una alpha mal calibrada.
    """
    x_array, y_array = _validar_datos(x, y)
    _validar_parametros_lineales(w_inicial, b_inicial)
    paso_alpha = _validar_alpha(alpha)
    paso_h = _validar_h(h)
    limite_tolerancia = _validar_tolerancia(tolerancia)
    limite_iteraciones = _validar_max_iter(max_iter)
    if not np.isfinite(factor_divergecia) or factor_divergecia <= 1:
        raise ValueError("factor_divergecia debe ser un valor finito mayor que uno.")

    w = float(w_inicial)
    b = float(b_inicial)
    perdida_inicial = _perdida_mse(x_array, y_array, w, b)
    historial_w = [w]
    historial_b = [b]
    historial_perdida = [perdida_inicial]

    motivo_paro = "maximo_iteraciones"
    convergio = False
    divergio = False
    iteraciones = 0

    for _ in range(limite_iteraciones):
        vector_gradiente = gradiente(x_array, y_array, w, b, paso_h)
        if float(np.linalg.norm(vector_gradiente)) < limite_tolerancia:
            convergio = True
            motivo_paro = "tolerancia"
            break

        w = w - paso_alpha * float(vector_gradiente[0])
        b = b - paso_alpha * float(vector_gradiente[1])
        iteraciones += 1
        historial_w.append(w)
        historial_b.append(b)
        perdida_actual = _perdida_mse(x_array, y_array, w, b)
        historial_perdida.append(perdida_actual)

        if not np.isfinite(perdida_actual) or perdida_actual > factor_divergecia * perdida_inicial:
            divergio = True
            motivo_paro = "divergencia"
            break

    gradiente_final = gradiente(x_array, y_array, w, b, paso_h)

    return {
        "w": w,
        "b": b,
        "w_hist": np.array(historial_w),
        "b_hist": np.array(historial_b),
        "perdida_hist": np.array(historial_perdida),
        "perdida_inicial": perdida_inicial,
        "perdida_final": historial_perdida[-1],
        "gradiente_final": gradiente_final,
        "norma_gradiente_final": float(np.linalg.norm(gradiente_final)),
        "iteraciones": iteraciones,
        "motivo_paro": motivo_paro,
        "convergio": convergio,
        "divergio": divergio,
    }


def comparar_tasas(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    w_inicial: float,
    b_inicial: float,
    alphas: list[float],
    h: float,
    tolerancia: float,
    max_iter: int,
    factor_divergecia: float = 1e6,
) -> dict[str, dict[str, object]]:
    """Ejecuta el descenso con varias tasas usando los mismos datos y punto inicial."""
    if not isinstance(alphas, (list, tuple)) or len(alphas) == 0:
        raise ValueError("alphas debe ser una lista no vacia de tasas.")

    resultados: dict[str, dict[str, object]] = {}
    for alpha in alphas:
        resultado = descenso_gradiente(
            x,
            y,
            w_inicial,
            b_inicial,
            alpha,
            h,
            tolerancia,
            max_iter,
            factor_divergecia,
        )
        resultados[str(alpha)] = resultado
    return resultados
