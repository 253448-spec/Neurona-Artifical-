# Guía de Sarid — Cálculos manuales, LaTeX y apoyo en la defensa

[Volver al README principal](../README.md)

## Objetivo de tu responsabilidad

Demostrar matemáticamente que los resultados de Python son razonables y convertir el trabajo del equipo en un reporte claro, completo y bien sustentado.

## 1. Lo que tienes que hacer

1. Solicitar a Aarón y Emmanuel un caso base con datos, semilla, `P₀`, `v`, `α` y `h` fijos; no inventes valores distintos para los cálculos y el programa.
2. Plantear `L(w, b)` y obtener a mano `∂L/∂w` y `∂L/∂b`.
3. Normalizar `v`, calcular `u` y desarrollar `DᵤL = ∇L·u` en `P₀`.
4. Calcular a mano las primeras **dos iteraciones** con el mismo punto inicial y `α`. Indica claramente si usas el gradiente exacto o una aproximación para no mezclar métodos.
5. Elaborar la tabla variando `h`: derivada exacta, derivada numérica, error absoluto y error relativo porcentual. Si el valor exacto es cero, reporta error absoluto y explica por qué no divides entre cero.
6. Comparar las iteraciones y derivadas manuales con las salidas reales de Python; explica las diferencias por aproximación o redondeo.
7. Redactar el reporte en la plantilla oficial: resumen, marco teórico, desarrollo analítico, arquitectura, resultados, figuras, interpretación y conclusiones.
8. Incorporar las figuras generadas por Carlos, citar las fuentes en la bibliografía, compilar el PDF final y ayudar a organizar la presentación.

## 2. Temas que debes estudiar

- **Cálculo:** derivadas de sumas y potencias, regla de la cadena, derivadas parciales del MSE y producto punto.
- **Cálculo vectorial:** gradiente, norma, normalización, derivada direccional y desigualdad del teorema.
- **Optimización:** actualización por descenso del gradiente y efecto de `P₀` y `α` en las iteraciones.
- **Análisis numérico:** diferencias centradas, error de truncamiento, redondeo, error absoluto y error relativo.
- **LaTeX:** plantilla oficial, ecuaciones, alineación, tablas, inserción de imágenes, referencias cruzadas, citas y bibliografía.

## 3. Dónde guardar cada cosa

| Ruta | Qué va ahí |
|---|---|
| `docs/sarid_latex_analitico.md` | Esta guía, borradores de cálculos, comprobaciones, pendientes y notas para la defensa. |
| `latex/reporte_final.tex` | Reporte completo basado en la plantilla oficial. |
| `latex/bibliografia.bib` | Libros, artículos, documentación y demás fuentes citadas. |
| `latex/figuras/` | Figuras finales generadas por Carlos; no necesitas dibujarlas ni exportarlas manualmente. |
| `latex/reporte_final.pdf` | Documento final compilado después de revisar ecuaciones, tablas, imágenes y referencias. |

## 4. Lo que debes saber explicar en la defensa

- Cómo obtuviste cada derivada y por qué normalizaste `v`.
- Cómo realizaste las dos primeras iteraciones.
- Qué muestra la tabla al cambiar `h`.
- Por qué los valores exactos y numéricos pueden diferir.
- Cuándo no se puede calcular el error relativo y qué se reporta en su lugar.
- Qué conclusión matemática deja el experimento y cómo se relaciona con las gráficas.

## 5. Revisión del reporte antes de compilar la entrega

- [ ] Los datos y parámetros coinciden con el caso base de `main.py`.
- [ ] El gradiente y la derivada direccional están desarrollados paso a paso.
- [ ] Las primeras dos iteraciones manuales están completas.
- [ ] La tabla compara varios valores de `h` sin dividir entre cero.
- [ ] Las tres figuras aparecen con número, descripción e interpretación.
- [ ] El texto distingue resultados exactos, numéricos y redondeados.
- [ ] El documento usa la plantilla oficial y contiene todas las secciones obligatorias.
- [ ] Todas las fuentes citadas aparecen en `bibliografia.bib`.
- [ ] El PDF abre correctamente y no tiene referencias o imágenes faltantes.

