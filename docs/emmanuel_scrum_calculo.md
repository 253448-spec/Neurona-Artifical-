# Guía de Emmanuel — Scrum Master y módulo matemático

[Volver al README principal](../README.md)

## Objetivo de tu responsabilidad

Conseguir que el equipo avance de forma ordenada y que el núcleo matemático produzca resultados correctos, comprobables y fáciles de integrar.

## 1. Lo que tienes que hacer

1. Organizar las tareas, los tiempos y los commits; registrar acuerdos o bloqueos y revisar que todos puedan participar.
2. Acordar con Aarón y Carlos los nombres de las funciones, sus entradas y sus salidas antes de programar la integración.
3. Definir en `calculo_vectorial.py` los datos sintéticos, la predicción `ŷ = wx + b` y la pérdida MSE `L(w, b)`.
4. Programar las diferencias finitas centradas y construir con ellas `∇L = (∂L/∂w, ∂L/∂b)`.
5. Programar la derivada direccional: rechazar `v = (0, 0)`, calcular `u = v/||v||` y obtener `DᵤL = ∇L·u`.
6. Comprobar con direcciones aleatorias que `-||∇L|| ≤ DᵤL ≤ ||∇L||`.
7. Implementar el descenso del gradiente con `α`, tolerancia, máximo de iteraciones e historial de `w`, `b` y `L` por iteración.
8. Probar primero el módulo con casos sencillos y después entregar a Aarón funciones listas para llamar desde `main.py`.

## 2. Temas que debes estudiar

- **Python básico:** variables, listas, funciones, argumentos, valores de retorno y lectura de errores.
- **NumPy:** arreglos, operaciones elemento por elemento, sumas, producto punto y normas.
- **MSE:** significado de cada término y diferencia entre pérdida y predicción.
- **Derivadas parciales y diferencias finitas:** fórmulas centradas, importancia de `h` y error numérico.
- **Gradiente y derivada direccional:** interpretación geométrica, normalización y desigualdad del teorema.
- **Descenso del gradiente:** actualización, tasa de aprendizaje, convergencia, divergencia y criterios de paro.

## 3. Dónde guardar cada cosa

| Ruta | Qué va ahí |
|---|---|
| `calculo_vectorial.py` | Todo el código numérico: datos, pérdida, derivadas, gradiente, derivada direccional, teorema y descenso. |
| `docs/emmanuel_scrum_calculo.md` | Esta guía, pendientes, responsables, acuerdos, bloqueos, decisiones matemáticas y notas de revisión. |

No guardes resultados finales como si fueran notas fijas: el programa debe calcularlos. Las figuras las genera Carlos y el orden de ejecución lo integra Aarón.

## 4. Lo que debes saber explicar en la defensa

- Por qué el MSE mide el error de la neurona.
- Cómo se obtiene cada derivada mediante diferencias centradas.
- Por qué se normaliza `v` y cómo se calcula `DᵤL`.
- Por qué se avanza por `-∇L`.
- Qué diferencia existe entre `h` y `α`.
- Por qué llegar al máximo de iteraciones no siempre significa converger.

## 5. Seguimiento Scrum

Usa la tabla siguiente durante el desarrollo. Actualízala con tareas reales y elimina las filas cuando se sustituyan por el seguimiento del equipo.

| Fecha | Tarea o acuerdo | Responsable | Estado | Bloqueo o siguiente paso |
|---|---|---|---|---|
| 24/11/2026 | Estructura inicial, README y guías del equipo | Emmanuel | Hecho | Revisar reparto con Aarón y Carlos |
| 26/11/2026 | Datos sintéticos, predicción y pérdida MSE | Emmanuel | Hecho | Validado con `x=[0,1,2,3]`, `y=[1,3,5,7]` |
| 26/11/2026 | Diferencias finitas centradas y gradiente numérico | Emmanuel | Hecho | Coincide con el gradiente analítico |
| 26/11/2026 | Derivada direccional y comprobación del teorema | Emmanuel | Hecho | `v=(0,0)` produce `ValueError` |
| 26/11/2026 | Descenso del gradiente con historial y paro | Emmanuel | Hecho | Convergencia y divergencia diferenciadas |
| Pendiente | Acordar el caso base con Aarón | Emmanuel y Aarón | Pendiente | Definir `P₀`, `α`, `h`, `tolerancia` y `max_iter` |
| Pendiente | Acordar el formato de resultados con Carlos | Emmanuel y Carlos | Pendiente | Definir qué recibe `graficas.py` |
| Pendiente | Revisar la integración completa | Todo el equipo | Pendiente | Ejecutar `python main.py` |

## 6. API del módulo para la integración

Todas las funciones están en `calculo_vectorial.py` y usan solo NumPy.

| Función | Entradas | Salida |
|---|---|---|
| `generar_datos(n, w_real, b_real, ruido, semilla, x_min, x_max)` | Parámetros del experimento | `x`, `y` como arreglos de NumPy |
| `predecir(x, w, b)` | Muestras y parámetros | `ŷ = wx + b` |
| `perdida_mse(x, y, w, b)` | Muestras y parámetros | `L(w, b)` como `float` |
| `derivada_parcial_w(x, y, w, b, h)` | Muestras, punto y `h` | `∂L/∂w` numérica |
| `derivada_parcial_b(x, y, w, b, h)` | Muestras, punto y `h` | `∂L/∂b` numérica |
| `gradiente(x, y, w, b, h)` | Muestras, punto y `h` | `np.array([∂L/∂w, ∂L/∂b])` |
| `gradiente_analitico(x, y, w, b)` | Muestras y punto | Gradiente exacto, solo para comparar |
| `normalizar_direccion(v)` | Vector de dos componentes | `u = v/‖v‖` |
| `derivada_direccional(x, y, w, b, h, v)` | Muestras, punto, `h` y `v` | Diccionario con `u`, `gradiente`, `derivada` y `norma_gradiente` |
| `comprobar_teorema(x, y, w, b, h, semilla, cantidad)` | Muestras, punto y `h` | Diccionario con `cumple`, `norma_gradiente` y `casos` |
| `descenso_gradiente(x, y, w_inicial, b_inicial, alpha, h, tolerancia, max_iter)` | Muestras, `P₀` y parámetros | Diccionario con `w`, `b`, historiales, `motivo_paro`, `convergio` y `divergio` |
| `comparar_tasas(x, y, w_inicial, b_inicial, alphas, h, tolerancia, max_iter)` | Varias tasas `α` | Diccionario de resultados indexado por `α` |

Los historiales `w_hist`, `b_hist` y `perdida_hist` incluyen el punto inicial, por lo que su longitud es `iteraciones + 1`.

## 7. Validaciones del módulo

- `v = (0, 0)` produce `ValueError` con un mensaje claro.
- `h`, `alpha` y `tolerancia` deben ser positivos y finitos.
- `max_iter` debe ser un entero positivo.
- `x` e `y` deben ser arreglos unidimensionales, de la misma longitud y finitos.
- `n` y `semilla` deben ser enteros válidos.
- El motivo de paro distingue `tolerancia`, `maximo_iteraciones` y `divergencia`.

## 8. Comprobaciones ejecutadas

| Prueba | Resultado |
|---|---|
| MSE con `x=[0,1,2,3]`, `y=[1,3,5,7]`, `w=0.5`, `b=2` | Gradiente numérico `[-7.5, -2.5]` igual al analítico |
| Normalización de `v=[3,4]` | `u=[0.6, 0.8]` con `‖u‖ = 1` |
| Teorema con 200 direcciones aleatorias | Se cumple en todos los casos |
| Descenso con `alpha=0.05` en el caso sencillo | Converge a `w=2`, `b=1` con motivo `tolerancia` |
| Tasa demasiado grande | La pérdida crece y el motivo de paro es `divergencia` |
| Misma semilla dos veces | Se obtienen los mismos datos y resultados |

## 9. Observaciones para el caso base

Con `x` entre 0 y 100 las derivadas parciales tienen magnitudes del orden de `10⁴`, por lo que una tasa `alpha` cercana a `10⁻⁴` todavía diverge y una de `10⁻⁵` avanza muy lento. Dos opciones para que el caso base sea estable:

1. Usar `x` entre 0 y 1 y `ruido=0.5`, donde `alpha` entre `0.05` y `0.2` converge en menos de 3000 iteraciones.
2. Mantener la escala original y normalizar las variables antes del entrenamiento.

Recomiendo la opción 1 para la demostración en vivo, porque muestra la convergencia con pocas iteraciones. La decisión final es de Aarón, que concentra los parámetros en `main.py`.

