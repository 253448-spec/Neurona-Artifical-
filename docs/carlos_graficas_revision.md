# Guía de Carlos — Gráficas y apoyo técnico

[Volver al README principal](../README.md)

[Auditoría, pruebas e integración](carlos_plan_auditoria_graficas.md)

## Objetivo de tu responsabilidad

Convertir los resultados numéricos en tres figuras claras a 300 DPI y ayudar a detectar errores de integración o interpretación.

## 1. Lo que tienes que hacer

1. Acordar con Emmanuel y Aarón cómo recibirás los datos, el punto inicial, la dirección `u` y los historiales de entrenamiento.
2. Programar la **Figura 1:** superficie 3D de `L(w, b)` con la trayectoria y marcas de inicio y fin.
3. Programar la **Figura 2:** curvas de nivel, campo gradiente con `quiver`, vector `u` y pasos del algoritmo.
4. Programar la **Figura 3:** iteraciones contra pérdida para al menos dos tasas `α`, usando condiciones iniciales comparables.
5. Añadir títulos, etiquetas y leyendas; exportar automáticamente cada PNG a **300 DPI** en `latex/figuras/`.
6. Revisar que una trayectoria, una flecha o una curva correspondan realmente a los resultados y no a datos inventados.
7. Apoyar la revisión técnica cuando otro módulo tenga errores, sin cambiar archivos ajenos sin avisar.

## 2. Temas que debes estudiar

- **Python y NumPy:** funciones, arreglos bidimensionales y `meshgrid`.
- **Matplotlib:** figuras, ejes, ejes 3D, curvas, leyendas y guardado de archivos.
- **Visualización matemática:** `plot_surface`, `contour`, `quiver`, curvas de nivel y relación con el gradiente.
- **Convergencia:** interpretación de curvas de pérdida, oscilación, estancamiento y divergencia.
- **Calidad visual:** rangos adecuados, legibilidad, escalas y diferencia entre normalizar una flecha para dibujarla y cambiar su valor matemático.

## 3. Dónde guardar cada cosa

| Ruta | Qué va ahí |
|---|---|
| `graficas.py` | Funciones que reciben resultados y construyen las tres figuras; aquí no se implementa otro optimizador. |
| `latex/figuras/figura_1_superficie_3d.png` | Superficie 3D definitiva con la trayectoria. |
| `latex/figuras/figura_2_curvas_nivel.png` | Curvas de nivel, gradiente, vector `u` y pasos. |
| `latex/figuras/figura_3_convergencia.png` | Comparación de convergencia para distintas tasas `α`. |
| `figuras/` | Solamente pruebas o borradores; no son la fuente final del reporte. |
| `docs/carlos_graficas_revision.md` | Esta guía, decisiones visuales, revisiones, errores encontrados y recomendaciones técnicas. |

## 4. Lo que debes saber explicar en la defensa

- Qué representa cada eje y cada trayectoria.
- Por qué el gradiente es perpendicular a las curvas de nivel.
- Qué indica el vector `u` y por qué es unitario.
- Cómo se interpreta una curva de convergencia.
- Qué cambia en las figuras cuando se modifica `P₀`, `v` o `α`.
- Cómo comprobaste que las imágenes se exportaron a 300 DPI.

## 5. Revisión técnica de las figuras

Marca cada punto únicamente después de comprobarlo con una ejecución real:

- [ ] La superficie usa la misma función de pérdida del módulo matemático.
- [ ] La trayectoria contiene los puntos inicial y final.
- [ ] El campo `quiver` y el vector `u` corresponden al punto indicado.
- [ ] La gráfica de convergencia compara al menos dos valores de `α`.
- [ ] Las etiquetas y leyendas se leen correctamente.
- [ ] Los tres archivos existen en `latex/figuras/` y fueron exportados a 300 DPI.
- [ ] Sarid comprobó que las imágenes se ven correctamente dentro del PDF.

## 6. Contrato propuesto para la integración

`graficas.py` ya expone tres funciones: `graficar_superficie_3d`, `graficar_curvas_nivel` y `graficar_convergencia`. Emmanuel y Aarón deben confirmar este contrato antes de conectar el flujo principal:

- Cada historial es un arreglo de forma `(n, 3)` con columnas `w`, `b` y `L`, incluido el punto inicial.
- `funcion_perdida(w, b)` acepta escalares y arreglos de NumPy y devuelve la pérdida correspondiente.
- `funcion_gradiente(w, b)` recibe escalares y devuelve `(dL_dw, dL_db)`.
- Para la comparación de convergencia se usa un diccionario `{alpha: historial}` con al menos dos tasas y el mismo `P₀`.
- Las rutas predeterminadas apuntan a `latex/figuras/`; otras rutas pueden usarse para borradores.

Las flechas del campo gradiente se normalizan solamente para que sean legibles. Esto se indica en la leyenda y no altera los valores matemáticos. El vector `u` debe llegar ya normalizado desde el módulo matemático; su longitud también se ajusta únicamente para visualizarlo.

## 7. Prueba local del módulo

Se ejecutaron las tres funciones con una pérdida cuadrática conocida y dos historiales de prueba. Los borradores se guardaron en `figuras/prueba_*.png`, no en las rutas definitivas. Se comprobó que los tres PNG son legibles, contienen sus elementos principales y conservan metadatos de 300 DPI. También se verificaron los rechazos de DPI inferior a 300, vector no unitario y comparación con una sola tasa.

Estas pruebas no sustituyen la integración con los datos reales de `calculo_vectorial.py` ni la revisión del PDF.

