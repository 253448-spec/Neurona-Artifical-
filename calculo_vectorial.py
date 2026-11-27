"""Modulo de calculo vectorial y optimizacion para la neurona lineal.

Contiene las funciones matematicas del proyecto: generacion de datos,
prediccion lineal, perdida MSE, derivadas parciales con diferencias
finitas centradas, vector gradiente, derivada direccional y descenso
del gradiente.
"""

import numpy as np


def _validar_datos(x, y):
    """Valida y convierte x e y a arreglos unidimensionales de NumPy."""
    try:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("x e y deben contener unicamente valores numericos.") from err

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x e y deben ser arreglos unidimensionales.")
    if x_arr.size == 0:
        raise ValueError("x e y no pueden estar vacios.")
    if x_arr.shape != y_arr.shape:
        raise ValueError("x e y deben contener la misma cantidad de muestras.")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("x e y no pueden contener valores NaN o infinitos.")

    return x_arr, y_arr


def generar_datos(
    n: int,
    w_real: float,
    b_real: float,
    ruido: float,
    semilla: int,
    x_min: float = 0.0,
    x_max: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera datos sinteticos para la relacion lineal y = w*x + b con ruido."""
    if type(n) is not int or n <= 0:
        raise ValueError("n debe ser un entero positivo.")
    if type(semilla) is not int or semilla < 0:
        raise ValueError("semilla debe ser un entero no negativo.")
    if not np.isfinite(ruido) or ruido < 0:
        raise ValueError("ruido debe ser un valor finito no negativo.")
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError("El rango de x debe ser finito y cumplir x_max > x_min.")
    if not np.isfinite(w_real) or not np.isfinite(b_real):
        raise ValueError("w y b deben ser valores finitos.")

    # Generador local para mantener la reproducibilidad sin alterar el estado global
    rng = np.random.default_rng(semilla)
    x = rng.uniform(x_min, x_max, size=n)
    errores = rng.normal(loc=0.0, scale=ruido, size=n)
    y = w_real * x + b_real + errores
    return x, y


def predecir(x, w: float, b: float) -> np.ndarray:
    """Calcula la prediccion del modelo lineal: y_hat = w*x + b."""
    if not np.isfinite(w) or not np.isfinite(b):
        raise ValueError("w y b deben ser valores finitos.")
    try:
        x_arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("x debe contener unicamente valores numericos.") from err
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x no puede contener valores NaN o infinitos.")
    return w * x_arr + b


def _calcular_mse(x_arr: np.ndarray, y_arr: np.ndarray, w: float, b: float) -> float:
    """Calculo directo del MSE sobre arreglos de NumPy ya validados."""
    residuos = y_arr - (w * x_arr + b)
    return float(np.mean(residuos**2))


def perdida_mse(x, y, w: float, b: float) -> float:
    """Calcula el Error Cuadratico Medio (MSE) de la neurona lineal."""
    x_arr, y_arr = _validar_datos(x, y)
    if not np.isfinite(w) or not np.isfinite(b):
        raise ValueError("w y b deben ser valores finitos.")
    return _calcular_mse(x_arr, y_arr, w, b)


def derivada_parcial_w(x, y, w: float, b: float, h: float) -> float:
    """Aproxima dL/dw usando diferencias finitas centradas."""
    x_arr, y_arr = _validar_datos(x, y)
    if not np.isfinite(w) or not np.isfinite(b):
        raise ValueError("w y b deben ser valores finitos.")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h debe ser un valor finito mayor que cero.")

    # Formula centrada: [L(w+h, b) - L(w-h, b)] / (2h)
    l_mas = _calcular_mse(x_arr, y_arr, w + h, b)
    l_menos = _calcular_mse(x_arr, y_arr, w - h, b)
    return float((l_mas - l_menos) / (2.0 * h))


def derivada_parcial_b(x, y, w: float, b: float, h: float) -> float:
    """Aproxima dL/db usando diferencias finitas centradas."""
    x_arr, y_arr = _validar_datos(x, y)
    if not np.isfinite(w) or not np.isfinite(b):
        raise ValueError("w y b deben ser valores finitos.")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h debe ser un valor finito mayor que cero.")

    # Formula centrada: [L(w, b+h) - L(w, b-h)] / (2h)
    l_mas = _calcular_mse(x_arr, y_arr, w, b + h)
    l_menos = _calcular_mse(x_arr, y_arr, w, b - h)
    return float((l_mas - l_menos) / (2.0 * h))


def gradiente(x, y, w: float, b: float, h: float) -> np.ndarray:
    """Calcula el vector gradiente numerico: grad L = [dL/dw, dL/db]."""
    x_arr, y_arr = _validar_datos(x, y)
    if not np.isfinite(w) or not np.isfinite(b):
        raise ValueError("w y b deben ser valores finitos.")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h debe ser un valor finito mayor que cero.")

    # Evaluacion con diferencias finitas centradas
    dw = (_calcular_mse(x_arr, y_arr, w + h, b) - _calcular_mse(x_arr, y_arr, w - h, b)) / (2.0 * h)
    db = (_calcular_mse(x_arr, y_arr, w, b + h) - _calcular_mse(x_arr, y_arr, w, b - h)) / (2.0 * h)
    return np.array([float(dw), float(db)])


def gradiente_analitico(x, y, w: float, b: float) -> np.ndarray:
    """Calcula el gradiente exacto usando las formulas analiticas del MSE."""
    x_arr, y_arr = _validar_datos(x, y)
    if not np.isfinite(w) or not np.isfinite(b):
        raise ValueError("w y b deben ser valores finitos.")

    residuos = y_arr - (w * x_arr + b)
    n = x_arr.size
    dw = float(-2.0 * np.sum(x_arr * residuos) / n)
    db = float(-2.0 * np.sum(residuos) / n)
    return np.array([dw, db])


def normalizar_direccion(v) -> np.ndarray:
    """Normaliza un vector de direccion v = (v_w, v_b) y rechaza el vector nulo."""
    try:
        vec = np.asarray(v, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("v debe contener unicamente valores numericos.") from err

    if vec.ndim != 1 or vec.size != 2:
        raise ValueError("v debe ser un vector de dos componentes (v_w, v_b).")
    if not np.all(np.isfinite(vec)):
        raise ValueError("v no puede contener valores NaN o infinitos.")

    norma = float(np.linalg.norm(vec))
    if norma == 0.0:
        raise ValueError("v no puede ser el vector (0, 0) porque no se puede normalizar.")

    return vec / norma


def derivada_direccional(x, y, w: float, b: float, h: float, v) -> dict[str, object]:
    """Calcula la derivada direccional D_u L = grad L . u."""
    u = normalizar_direccion(v)
    grad = gradiente(x, y, w, b, h)
    derivada = float(np.dot(grad, u))
    return {
        "u": u,
        "gradiente": grad,
        "derivada": derivada,
        "norma_gradiente": float(np.linalg.norm(grad)),
    }


def comprobar_teorema(
    x,
    y,
    w: float,
    b: float,
    h: float,
    semilla: int,
    cantidad: int = 100,
    tolerancia: float = 1e-8,
) -> dict[str, object]:
    """Comprueba el teorema: -||grad L|| <= D_u L <= ||grad L|| en direcciones aleatorias."""
    if type(cantidad) is not int or cantidad <= 0:
        raise ValueError("cantidad debe ser un entero positivo.")
    if type(semilla) is not int or semilla < 0:
        raise ValueError("semilla debe ser un entero no negativo.")
    if not np.isfinite(tolerancia) or tolerancia < 0:
        raise ValueError("tolerancia debe ser un valor finito no negativo.")

    grad = gradiente(x, y, w, b, h)
    norma = float(np.linalg.norm(grad))
    rng = np.random.default_rng(semilla)

    casos = []
    cumple = True
    for _ in range(cantidad):
        vec = rng.normal(size=2)
        norma_vec = np.linalg.norm(vec)
        if norma_vec == 0.0:
            continue
        u = vec / norma_vec
        d_u = float(np.dot(grad, u))
        dentro = (-norma - tolerancia) <= d_u <= (norma + tolerancia)
        if not dentro:
            cumple = False
        casos.append(
            {
                "derivada": d_u,
                "limite_inferior": -norma,
                "limite_superior": norma,
                "cumple": dentro,
            }
        )

    return {
        "cumple": cumple,
        "norma_gradiente": norma,
        "cantidad": len(casos),
        "casos": casos,
    }


def descenso_gradiente(
    x,
    y,
    w_inicial: float,
    b_inicial: float,
    alpha: float,
    h: float,
    tolerancia: float,
    max_iter: int,
    factor_divergencia: float = 1e6,
) -> dict[str, object]:
    """Entrena la neurona ajustando w y b mediante descenso del gradiente."""
    x_arr, y_arr = _validar_datos(x, y)
    if not np.isfinite(w_inicial) or not np.isfinite(b_inicial):
        raise ValueError("w y b deben ser valores finitos.")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha debe ser un valor finito mayor que cero.")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h debe ser un valor finito mayor que cero.")
    if not np.isfinite(tolerancia) or tolerancia <= 0:
        raise ValueError("tolerancia debe ser un valor finito mayor que cero.")
    if type(max_iter) is not int or max_iter <= 0:
        raise ValueError("max_iter debe ser un entero positivo.")
    if not np.isfinite(factor_divergencia) or factor_divergencia <= 1:
        raise ValueError("factor_divergencia debe ser un valor finito mayor que uno.")

    w = float(w_inicial)
    b = float(b_inicial)
    perdida_inicial = _calcular_mse(x_arr, y_arr, w, b)

    historial_w = [w]
    historial_b = [b]
    historial_perdida = [perdida_inicial]

    # Umbral para detectar si el algoritmo se desborda por una tasa alpha excesiva
    umbral_divergencia = max(factor_divergencia * perdida_inicial, 1e6)

    motivo_paro = "maximo_iteraciones"
    convergio = False
    divergio = False
    iteraciones = 0

    for _ in range(max_iter):
        # Gradiente en el punto actual con diferencias finitas centradas
        dw = (_calcular_mse(x_arr, y_arr, w + h, b) - _calcular_mse(x_arr, y_arr, w - h, b)) / (2.0 * h)
        db = (_calcular_mse(x_arr, y_arr, w, b + h) - _calcular_mse(x_arr, y_arr, w, b - h)) / (2.0 * h)
        norma_grad = float(np.sqrt(dw**2 + db**2))

        # Criterio de convergencia: ||grad L|| < tolerancia
        if norma_grad < tolerancia:
            convergio = True
            motivo_paro = "tolerancia"
            break

        # Paso de actualizacion: P_{k+1} = P_k - alpha * grad L
        w = w - alpha * dw
        b = b - alpha * db
        iteraciones += 1

        historial_w.append(w)
        historial_b.append(b)

        perdida_actual = _calcular_mse(x_arr, y_arr, w, b)
        historial_perdida.append(perdida_actual)

        # Criterio de divergencia
        if not np.isfinite(perdida_actual) or perdida_actual > umbral_divergencia:
            divergio = True
            motivo_paro = "divergencia"
            break

    # Gradiente en el ultimo punto alcanzado
    dw_fin = (_calcular_mse(x_arr, y_arr, w + h, b) - _calcular_mse(x_arr, y_arr, w - h, b)) / (2.0 * h)
    db_fin = (_calcular_mse(x_arr, y_arr, w, b + h) - _calcular_mse(x_arr, y_arr, w, b - h)) / (2.0 * h)
    gradiente_final = np.array([float(dw_fin), float(db_fin)])

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
    x,
    y,
    w_inicial: float,
    b_inicial: float,
    alphas: list[float],
    h: float,
    tolerancia: float,
    max_iter: int,
    factor_divergencia: float = 1e6,
) -> dict[str, dict[str, object]]:
    """Ejecuta el descenso del gradiente para distintas tasas de aprendizaje alpha."""
    if not isinstance(alphas, (list, tuple)) or len(alphas) == 0:
        raise ValueError("alphas debe ser una lista no vacia de tasas.")

    resultados = {}
    for alpha in alphas:
        res = descenso_gradiente(
            x=x,
            y=y,
            w_inicial=w_inicial,
            b_inicial=b_inicial,
            alpha=alpha,
            h=h,
            tolerancia=tolerancia,
            max_iter=max_iter,
            factor_divergencia=factor_divergencia,
        )
        resultados[str(alpha)] = res
    return resultados
