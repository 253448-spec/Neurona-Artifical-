"""Generación de las tres figuras del proyecto de cálculo vectorial.

Contrato de integración:
- Un historial es un arreglo de forma (n, 3) con columnas ``w``, ``b`` y ``L``.
- ``funcion_perdida(w, b)`` debe aceptar escalares y arreglos de NumPy.
- ``funcion_gradiente(w, b)`` recibe dos escalares y devuelve ``(dL_dw, dL_db)``.

Este módulo solamente visualiza resultados; no genera datos ni entrena la neurona.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray


Historial: TypeAlias = NDArray[np.float64]
FuncionPerdida: TypeAlias = Callable[[ArrayLike, ArrayLike], ArrayLike]
FuncionGradiente: TypeAlias = Callable[[float, float], ArrayLike]

DIRECTORIO_PROYECTO = Path(__file__).resolve().parent
DIRECTORIO_FIGURAS = DIRECTORIO_PROYECTO / "latex" / "figuras"
RUTA_FIGURA_1 = DIRECTORIO_FIGURAS / "figura_1_superficie_3d.png"
RUTA_FIGURA_2 = DIRECTORIO_FIGURAS / "figura_2_curvas_nivel.png"
RUTA_FIGURA_3 = DIRECTORIO_FIGURAS / "figura_3_convergencia.png"
DPI_MINIMO = 300


def _validar_dpi(dpi: int) -> None:
    if isinstance(dpi, bool) or not isinstance(dpi, (int, np.integer)):
        raise ValueError("dpi debe ser un número entero mayor o igual que 300.")
    if dpi < DPI_MINIMO:
        raise ValueError("Las figuras deben exportarse al menos a 300 DPI.")


def _validar_historial(historial: ArrayLike, nombre: str = "historial") -> Historial:
    try:
        arreglo = np.asarray(historial, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{nombre} debe contener valores numéricos.") from exc

    if arreglo.ndim != 2 or arreglo.shape[1] != 3 or arreglo.shape[0] == 0:
        raise ValueError(
            f"{nombre} debe tener forma (n, 3), con columnas w, b y L, y n > 0."
        )
    if not np.all(np.isfinite(arreglo)):
        raise ValueError(f"{nombre} contiene valores no finitos (NaN o infinito).")
    return arreglo


def _validar_rango(
    rango: ArrayLike | None, valores: NDArray[np.float64], nombre: str
) -> tuple[float, float]:
    if rango is None:
        minimo = float(np.min(valores))
        maximo = float(np.max(valores))
        margen = max((maximo - minimo) * 0.25, 1.0)
        return minimo - margen, maximo + margen

    try:
        limites = np.asarray(rango, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{nombre} debe ser una tupla (mínimo, máximo).") from exc

    if limites.shape != (2,) or not np.all(np.isfinite(limites)):
        raise ValueError(f"{nombre} debe ser una tupla finita (mínimo, máximo).")
    if limites[0] >= limites[1]:
        raise ValueError(f"El mínimo de {nombre} debe ser menor que el máximo.")
    return float(limites[0]), float(limites[1])


def _crear_malla(
    rango_w: tuple[float, float], rango_b: tuple[float, float], puntos: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(puntos, bool) or not isinstance(puntos, (int, np.integer)) or puntos < 2:
        raise ValueError("La cantidad de puntos de la malla debe ser un entero mayor que 1.")
    valores_w = np.linspace(*rango_w, puntos)
    valores_b = np.linspace(*rango_b, puntos)
    return np.meshgrid(valores_w, valores_b)


def _evaluar_perdida(
    funcion_perdida: FuncionPerdida,
    malla_w: NDArray[np.float64],
    malla_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    try:
        valores = np.asarray(funcion_perdida(malla_w, malla_b), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "funcion_perdida debe aceptar arreglos de NumPy y devolver valores numéricos."
        ) from exc

    if valores.ndim == 0:
        valores = np.full(malla_w.shape, float(valores))
    if valores.shape != malla_w.shape:
        raise ValueError("funcion_perdida no devolvió un valor para cada punto de la malla.")
    if not np.all(np.isfinite(valores)):
        raise ValueError("La función de pérdida produjo valores no finitos en la malla.")
    return valores


def _comprobar_perdidas_historial(
    funcion_perdida: FuncionPerdida, historial: Historial
) -> None:
    esperadas = np.asarray(
        funcion_perdida(historial[:, 0], historial[:, 1]), dtype=float
    )
    if esperadas.shape != (historial.shape[0],) or not np.allclose(
        historial[:, 2], esperadas, rtol=1e-6, atol=1e-9
    ):
        raise ValueError(
            "Las pérdidas del historial no coinciden con funcion_perdida(w, b)."
        )


def _guardar_figura(figura: plt.Figure, ruta_salida: str | Path, dpi: int) -> Path:
    ruta = Path(ruta_salida)
    if ruta.suffix.lower() != ".png":
        raise ValueError("La ruta de salida debe tener extensión .png.")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta, dpi=dpi, bbox_inches="tight")
    return ruta


def graficar_superficie_3d(
    funcion_perdida: FuncionPerdida,
    historial: ArrayLike,
    rango_w: ArrayLike | None = None,
    rango_b: ArrayLike | None = None,
    ruta_salida: str | Path = RUTA_FIGURA_1,
    dpi: int = DPI_MINIMO,
    puntos_malla: int = 60,
) -> Path:
    """Exporta la superficie de pérdida con la trayectoria completa de entrenamiento."""
    _validar_dpi(dpi)
    datos = _validar_historial(historial)
    _comprobar_perdidas_historial(funcion_perdida, datos)
    limites_w = _validar_rango(rango_w, datos[:, 0], "rango_w")
    limites_b = _validar_rango(rango_b, datos[:, 1], "rango_b")
    malla_w, malla_b = _crear_malla(limites_w, limites_b, puntos_malla)
    malla_perdida = _evaluar_perdida(funcion_perdida, malla_w, malla_b)

    figura = plt.figure(figsize=(11, 8))
    try:
        eje = figura.add_subplot(111, projection="3d")
        superficie = eje.plot_surface(
            malla_w,
            malla_b,
            malla_perdida,
            cmap="viridis",
            alpha=0.78,
            linewidth=0,
            antialiased=True,
        )
        eje.plot(
            datos[:, 0],
            datos[:, 1],
            datos[:, 2],
            color="#111827",
            marker="o",
            markersize=3,
            linewidth=2,
            label="Trayectoria",
        )
        eje.scatter(
            *datos[0], color="#16a34a", s=80, depthshade=False, label="Punto inicial"
        )
        eje.scatter(
            *datos[-1], color="#dc2626", s=90, depthshade=False, label="Punto final"
        )
        eje.set(
            title="Superficie de pérdida y trayectoria del descenso del gradiente",
            xlabel="w (pendiente)",
            ylabel="b (sesgo)",
            zlabel="L(w, b)",
        )
        eje.view_init(elev=28, azim=-55)
        eje.legend(loc="upper left")
        figura.colorbar(superficie, ax=eje, shrink=0.65, pad=0.1, label="L(w, b)")
        return _guardar_figura(figura, ruta_salida, dpi)
    finally:
        plt.close(figura)


def graficar_curvas_nivel(
    funcion_perdida: FuncionPerdida,
    funcion_gradiente: FuncionGradiente,
    historial: ArrayLike,
    punto_prueba: ArrayLike,
    vector_u: ArrayLike,
    rango_w: ArrayLike | None = None,
    rango_b: ArrayLike | None = None,
    ruta_salida: str | Path = RUTA_FIGURA_2,
    dpi: int = DPI_MINIMO,
    puntos_malla: int = 50,
    densidad_gradiente: int = 12,
) -> Path:
    """Exporta curvas de nivel, gradientes normalizados, vector u y trayectoria."""
    _validar_dpi(dpi)
    datos = _validar_historial(historial)
    _comprobar_perdidas_historial(funcion_perdida, datos)
    limites_w = _validar_rango(rango_w, datos[:, 0], "rango_w")
    limites_b = _validar_rango(rango_b, datos[:, 1], "rango_b")
    malla_w, malla_b = _crear_malla(limites_w, limites_b, puntos_malla)
    malla_perdida = _evaluar_perdida(funcion_perdida, malla_w, malla_b)

    punto = np.asarray(punto_prueba, dtype=float)
    direccion = np.asarray(vector_u, dtype=float)
    if punto.shape != (2,) or not np.all(np.isfinite(punto)):
        raise ValueError("punto_prueba debe ser un vector finito (w, b).")
    if direccion.shape != (2,) or not np.all(np.isfinite(direccion)):
        raise ValueError("vector_u debe ser un vector finito de dos componentes.")
    if not np.isclose(np.linalg.norm(direccion), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("vector_u debe estar normalizado: su norma debe ser uno.")

    malla_gw, malla_gb = _crear_malla(limites_w, limites_b, densidad_gradiente)
    gradiente_w = np.empty_like(malla_gw)
    gradiente_b = np.empty_like(malla_gb)
    for indice in np.ndindex(malla_gw.shape):
        gradiente = np.asarray(
            funcion_gradiente(float(malla_gw[indice]), float(malla_gb[indice])),
            dtype=float,
        )
        if gradiente.shape != (2,) or not np.all(np.isfinite(gradiente)):
            raise ValueError(
                "funcion_gradiente debe devolver dos valores finitos: (dL_dw, dL_db)."
            )
        gradiente_w[indice], gradiente_b[indice] = gradiente

    normas = np.hypot(gradiente_w, gradiente_b)
    unitario_w = np.divide(
        gradiente_w, normas, out=np.zeros_like(normas), where=normas > 0
    )
    unitario_b = np.divide(
        gradiente_b, normas, out=np.zeros_like(normas), where=normas > 0
    )
    longitud_flecha = 0.08 * min(
        limites_w[1] - limites_w[0], limites_b[1] - limites_b[0]
    )

    figura, eje = plt.subplots(figsize=(11, 8))
    try:
        curvas = eje.contour(
            malla_w, malla_b, malla_perdida, levels=20, cmap="viridis"
        )
        eje.clabel(curvas, inline=True, fontsize=8, fmt="%.2g")
        eje.quiver(
            malla_gw,
            malla_gb,
            unitario_w * longitud_flecha,
            unitario_b * longitud_flecha,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#475569",
            alpha=0.75,
            width=0.003,
            label="Gradiente (dirección normalizada)",
        )
        eje.plot(
            datos[:, 0],
            datos[:, 1],
            color="#111827",
            marker="o",
            markersize=3,
            linewidth=2,
            label="Trayectoria",
        )
        eje.scatter(
            datos[0, 0], datos[0, 1], color="#16a34a", s=80, label="Punto inicial", zorder=5
        )
        eje.scatter(
            datos[-1, 0], datos[-1, 1], color="#dc2626", s=90, label="Punto final", zorder=5
        )
        eje.scatter(
            *punto, color="#7c3aed", marker="D", s=70, label="Punto de prueba", zorder=5
        )
        eje.quiver(
            punto[0],
            punto[1],
            direccion[0] * longitud_flecha * 1.8,
            direccion[1] * longitud_flecha * 1.8,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#dc2626",
            width=0.006,
            label="u unitario (longitud ajustada para visualizar)",
            zorder=6,
        )
        eje.set(
            title="Curvas de nivel, gradiente y dirección de prueba",
            xlabel="w (pendiente)",
            ylabel="b (sesgo)",
            xlim=limites_w,
            ylim=limites_b,
        )
        eje.set_aspect("equal", adjustable="box")
        eje.grid(alpha=0.2)
        eje.legend(loc="best", fontsize=9)
        figura.colorbar(curvas, ax=eje, label="L(w, b)")
        return _guardar_figura(figura, ruta_salida, dpi)
    finally:
        plt.close(figura)


def graficar_convergencia(
    historiales_por_alpha: Mapping[float, ArrayLike],
    ruta_salida: str | Path = RUTA_FIGURA_3,
    dpi: int = DPI_MINIMO,
    escala_logaritmica: bool = False,
) -> Path:
    """Compara la pérdida por iteración para dos o más tasas de aprendizaje."""
    _validar_dpi(dpi)
    if not isinstance(historiales_por_alpha, Mapping) or len(historiales_por_alpha) < 2:
        raise ValueError("Proporciona al menos dos historiales con valores distintos de α.")

    historiales: list[tuple[float, Historial]] = []
    punto_inicial: NDArray[np.float64] | None = None
    for alpha, historial in historiales_por_alpha.items():
        try:
            tasa = float(alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError("Cada clave debe ser una tasa α numérica.") from exc
        if not np.isfinite(tasa) or tasa <= 0:
            raise ValueError("Todas las tasas α deben ser finitas y positivas.")

        datos = _validar_historial(historial, f"historial para α={tasa}")
        if punto_inicial is None:
            punto_inicial = datos[0, :2].copy()
        elif not np.allclose(datos[0, :2], punto_inicial, rtol=1e-9, atol=1e-12):
            raise ValueError("Las tasas deben compararse desde el mismo punto inicial P₀.")
        if escala_logaritmica and np.any(datos[:, 2] <= 0):
            raise ValueError("La escala logarítmica requiere pérdidas estrictamente positivas.")
        historiales.append((tasa, datos))

    figura, eje = plt.subplots(figsize=(10, 6))
    try:
        for tasa, datos in sorted(historiales, key=lambda elemento: elemento[0]):
            iteraciones = np.arange(datos.shape[0])
            eje.plot(
                iteraciones,
                datos[:, 2],
                marker="o",
                markersize=3,
                markevery=max(1, datos.shape[0] // 12),
                linewidth=2,
                label=f"α = {tasa:g}",
            )
        if escala_logaritmica:
            eje.set_yscale("log")
        etiqueta_perdida = "Pérdida L(w, b)"
        if escala_logaritmica:
            etiqueta_perdida += " (escala logarítmica)"
        eje.set(
            title="Convergencia para distintas tasas de aprendizaje",
            xlabel="Iteración",
            ylabel=etiqueta_perdida,
        )
        eje.grid(True, which="both", alpha=0.25)
        eje.legend()
        figura.tight_layout()
        return _guardar_figura(figura, ruta_salida, dpi)
    finally:
        plt.close(figura)


__all__ = [
    "graficar_superficie_3d",
    "graficar_curvas_nivel",
    "graficar_convergencia",
    "RUTA_FIGURA_1",
    "RUTA_FIGURA_2",
    "RUTA_FIGURA_3",
]
