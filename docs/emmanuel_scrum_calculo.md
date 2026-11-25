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
| Pendiente | Definir entradas y salidas de las funciones | Emmanuel, Aarón y Carlos | Pendiente | Reunirse antes de integrar |
| Pendiente | Implementar y validar el módulo matemático | Emmanuel | Pendiente | Comparar con cálculos sencillos |
| Pendiente | Revisar la integración completa | Todo el equipo | Pendiente | Ejecutar el caso base |

