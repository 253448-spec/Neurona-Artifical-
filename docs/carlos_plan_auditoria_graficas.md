# Auditoría, pruebas e integración de `graficas.py`

Responsable: Carlos

Alcance: documentar el trabajo realizado, revisar el módulo de visualización y planificar su integración con Emmanuel, Aarón y Sarid.

## 1. Estado actual del trabajo

Se implementó `graficas.py` sin añadir generación de datos, cálculo de gradientes ni otro optimizador. El módulo recibe resultados y produce las tres figuras solicitadas. Las funciones cierran sus figuras de Matplotlib después de exportarlas para evitar acumular recursos.

También se ejecutó una prueba local con una función cuadrática conocida y dos historiales de prueba. Los archivos resultantes son borradores en `figuras/prueba_*.png`; no son resultados definitivos del proyecto.

`calculo_vectorial.py` y `main.py` todavía están vacíos. Por ello, la integración real y las figuras definitivas siguen pendientes.

## 2. Qué hace actualmente el código

| Función | Entradas principales | Resultado |
|---|---|---|
| `graficar_superficie_3d` | Función de pérdida, historial `(w, b, L)`, rangos y ruta | Superficie 3D de `L(w,b)`, trayectoria completa y marcas de inicio/fin |
| `graficar_curvas_nivel` | Funciones de pérdida y gradiente, historial, punto de prueba, vector `u`, rangos y ruta | Curvas de nivel, campo gradiente, trayectoria, punto de prueba y vector `u` |
| `graficar_convergencia` | Diccionario `{alpha: historial}`, ruta y escala opcional | Pérdida contra iteración para dos o más tasas de aprendizaje |

Las tres funciones devuelven un objeto `Path` con la ruta del PNG generado. Por defecto utilizan los nombres definitivos dentro de `latex/figuras/`; las pruebas deben indicar otra carpeta.

### Contrato de datos

- Cada historial debe ser un arreglo numérico de forma `(n, 3)`, con columnas `w`, `b` y `L` y al menos una fila.
- La primera fila representa `P₀`; la última, el punto final registrado.
- `funcion_perdida(w, b)` debe aceptar escalares y arreglos de NumPy.
- `funcion_gradiente(w, b)` recibe escalares y devuelve `(dL_dw, dL_db)`.
- `vector_u` debe llegar normalizado desde el módulo matemático.
- Las comparaciones de convergencia deben usar el mismo punto inicial y al menos dos tasas positivas.

### Validaciones ya incorporadas

- Historial con forma correcta, valores numéricos y finitos.
- Correspondencia entre las pérdidas del historial y `funcion_perdida(w,b)` en las figuras 1 y 2.
- Rangos finitos y crecientes.
- DPI entero y no inferior a 300.
- Vector `u` de dos componentes, finito y unitario.
- Gradiente de dos componentes finitas en cada punto de la malla.
- Al menos dos historiales, tasas positivas y mismo `P₀` en convergencia.
- Pérdidas positivas cuando se solicita escala logarítmica.
- Extensión `.png` en la ruta de salida.

### Decisiones visuales

- Las flechas del gradiente se normalizan solamente para mejorar su legibilidad; la leyenda lo declara. No se modifican los cálculos matemáticos.
- La longitud de `u` se ajusta solamente para dibujarla. Su dirección se conserva y el módulo comprueba que sea unitario.
- Las figuras se cierran después de guardarse y no se llaman automáticamente `plt.show()` para que la ejecución del simulador no se bloquee.

## 3. Decisión sobre figuras 3D interactivas

Las reglas actuales prevalecen sobre una preferencia por interactividad:

- El README exige **PNG a 300 DPI** en `latex/figuras/` y no acepta capturas como sustituto de la exportación automática.
- La Figura 1 debe ser una **superficie 3D**.
- Las figuras 2 y 3 están definidas como curvas de nivel/campo gradiente y convergencia, respectivamente; no se solicita que sean 3D.
- Las dependencias científicas permitidas son NumPy, Matplotlib y, opcionalmente, SymPy.

Por tanto, la entrega obligatoria será estática y solo la primera figura será 3D. Un PNG no puede ser interactivo. Como apoyo opcional para explorar la superficie durante la defensa se podrá añadir posteriormente una visualización local con el backend interactivo de Matplotlib, sin sustituir los PNG ni añadir Plotly. Esa mejora queda después de completar la integración y no es un requisito actual.

## 4. Resultado de la auditoría

La revisión contrastó `graficas.py`, el README, la guía de Carlos y los tres borradores. El módulo tiene una base correcta, pero todavía no está listo para entrega porque faltan correcciones, pruebas repetibles e integración real.

### 4.1 Bloqueo de integración

- **C-01 — Flujo definitivo inexistente:** `main.py`, `calculo_vectorial.py`, `requirements.txt` y `latex/reporte_final.tex` están vacíos. En `latex/figuras/` todavía no existen los PNG definitivos. Esto no es un error interno de las funciones de Carlos, pero impide aceptar el trabajo integrado.

### 4.2 Prioridad alta

- **A-01 — Comparación de tasas insuficientemente validada:** dos claves como `0.1` y `"0.10"` se convierten en la misma tasa y actualmente serían aceptadas. Además, comprobar solo `P₀` no detecta historiales con distinta pérdida inicial, datos o semilla. Se deben rechazar tasas duplicadas tras su conversión, comparar también `L(P₀)` y hacer que `main.py` garantice que únicamente cambie `α`. Las etiquetas deben conservar precisión suficiente para no mostrar dos tasas distintas como iguales.
- **A-02 — Posible recorte de elementos obligatorios:** los rangos personalizados pueden excluir parte de la trayectoria o el punto de prueba. La flecha `u` puede llegar al borde y quedar recortada, como ocurre en el borrador de curvas de nivel. Antes de integrar se debe ampliar el rango o rechazarlo con un mensaje claro; no se aceptará un recorte silencioso.
- **A-03 — Contrato del MSE vectorizado sin comprobar:** una función de pérdida que cierre sobre arreglos `x` e `y` puede producir broadcasting incorrecto al recibir una malla. La forma del resultado no basta para demostrar que cada celda contiene el MSE correcto. Se debe comparar contra evaluaciones escalares independientes, usando tamaños de datos distintos de las dimensiones de la malla.

### 4.3 Prioridad media

- **M-01 — Marcas superpuestas:** cuando `punto_prueba == P₀`, el rombo morado cubre el punto inicial verde. Si inicio y fin coinciden, una marca también puede ocultar la otra. Se debe definir una representación que conserve ambos elementos.
- **M-02 — Visibilidad en 3D:** la transparencia, oclusión y cámara fija pueden reducir el contraste de la trayectoria o los extremos. Se revisará con trayectorias convergentes, oscilantes y divergentes antes de fijar la presentación final.
- **M-03 — Divergencia no finita:** el módulo rechaza `NaN` e infinitos, lo cual evita ocultar errores, pero una tasa que desborde no podrá representarse sin una decisión de integración. `main.py` deberá detener y diagnosticar el desbordamiento antes de perder valores finitos; no se eliminarán filas silenciosamente.
- **M-04 — Tipo de `escala_logaritmica`:** falta exigir un valor booleano para evitar que una cadena como `"False"` active la escala.
- **M-05 — Sin suite automatizada:** la prueba local y los borradores no sustituyen pruebas repetibles de validación, regresión, recursos e integración.

### 4.4 Prioridad baja y casos límite

- Detectar y rechazar entradas complejas en lugar de descartar silenciosamente su parte imaginaria.
- Uniformar mensajes de error para conversiones y excepciones de las funciones externas.
- Evitar desbordamientos intermedios con rangos extremos.
- Rechazar o tratar explícitamente una pérdida constante en la Figura 2, porque no produce curvas de nivel útiles.

### 4.5 Aspectos correctos que se conservarán

- No hay entrenamiento, generación de datos ni optimización ocultos en `graficas.py`.
- Las rutas predeterminadas coinciden con el README y dependen de `__file__`, no del directorio actual.
- El historial incluye `P₀` y la Figura 3 numera desde la iteración cero.
- Las pérdidas del historial se contrastan con la función recibida en las figuras 1 y 2.
- El gradiente se normaliza solo para visualizarlo y se evita dividir entre cero.
- Las figuras se cierran con `finally`, incluso si la exportación falla.
- Los tres borradores tienen firma PNG válida y metadato `pHYs=11811`, equivalente a aproximadamente 299.9994 DPI; la diferencia respecto de 300 es la conversión entera normal de pulgadas a metros.

## 5. Plan de pruebas

### 5.1 Prueba local ya realizada

Se usó una pérdida cuadrática conocida, su gradiente y dos trayectorias generadas únicamente para la prueba. Se verificó:

- Creación de los tres PNG.
- Apertura y revisión visual de los borradores.
- Metadatos de aproximadamente 300 DPI.
- Rechazo de DPI inferior a 300.
- Rechazo de vector no unitario.
- Rechazo de comparación con una sola tasa.

Esta prueba no demuestra todavía la integración con el MSE ni con las diferencias finitas del módulo matemático.

### 5.2 Pruebas automatizadas que se incorporarán

Se creará `tests/test_graficas.py` con `unittest`, backend `Agg` y carpetas temporales, de modo que las pruebas no sobrescriban las figuras definitivas.

Casos previstos:

1. Exportación correcta de cada figura y existencia del archivo.
2. Comprobación de metadatos DPI y formato PNG.
3. Historial vacío, forma incorrecta y valores `NaN`/infinito.
4. Rangos invertidos, no finitos y mallas insuficientes.
5. Pérdida del historial inconsistente con la función recibida.
6. Vector `u` nulo, no unitario o con dimensiones incorrectas.
7. Punto de prueba fuera del rango y comportamiento esperado documentado.
8. Gradiente con forma incorrecta o valores no finitos; gradiente cero sin errores de normalización.
9. Menos de dos tasas, tasas duplicadas tras conversión numérica, tasas inválidas y puntos iniciales distintos.
10. Escala logarítmica con pérdidas cero o negativas.
11. Ejecuciones repetidas para comprobar que `plt.get_fignums()` no crezca, incluso cuando falle `savefig`.
12. Registro de advertencias para detectar conversiones complejas, curvas degeneradas o problemas de valores no finitos.

Comando previsto desde la raíz:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### 5.3 Prueba específica del MSE vectorizado

Con datos sintéticos fijos, se comparará la función vectorizada contra una evaluación escalar independiente:

1. Evaluar varios pares `(w,b)` individualmente y como vectores.
2. Evaluar una malla 2D y comprobar celdas seleccionadas manualmente.
3. Usar primero `N=7` observaciones y una malla `9×9`.
4. Repetir con `N=9` para detectar broadcasting que solo parezca funcionar cuando las dimensiones coinciden accidentalmente.
5. Confirmar que el promedio se calcula sobre observaciones, no sobre parámetros.

### 5.4 Prueba de integración pendiente

Cuando Emmanuel y Aarón completen sus módulos:

1. Generar datos con la semilla configurada.
2. Entrenar con el MSE y diferencias finitas reales.
3. Pasar el mismo historial a las tres funciones.
4. Confirmar que inicio/fin, pérdida y gradiente coinciden con la consola.
5. Comparar al menos dos `α` con los mismos datos y `P₀`.
6. Revisar visualmente los PNG definitivos.
7. Confirmar que Sarid los inserta correctamente en el PDF.

Solo después se marcarán los elementos correspondientes de la lista de revisión de `docs/carlos_graficas_revision.md`.

## 6. Dependencias para el trabajo de los demás

Para que otra persona pueda instalar el proyecto, la propuesta mínima para `requirements.txt` es:

```text
numpy>=1.26,<3.0
matplotlib>=3.8,<4.0
```

Son las dos dependencias directas de `graficas.py`. No se propone Plotly. Pillow ya llega como dependencia de Matplotlib y se usará únicamente para inspeccionar metadatos en las pruebas; no es una dependencia directa del simulador.

La prueba local se realizó con Python 3.15.2, NumPy 2.5.5 y Matplotlib 3.11.2. Antes de entregar se debe ejecutar también con la versión de Python que utilice el equipo, preferentemente una versión común compatible con Python 3.10 o superior.

`requirements.txt` pertenece a Aarón: Carlos debe compartir esta propuesta y Aarón debe confirmarla e integrarla con las dependencias de los demás módulos.

## 7. Qué se debe compartir con el equipo

En el cambio o pull request de Carlos se debe incluir:

- `graficas.py` auditado.
- `tests/test_graficas.py`, una vez creado y ejecutado.
- La documentación de Carlos actualizada.
- La propuesta o cambio acordado de `requirements.txt`, coordinado con Aarón.

No se deben incluir:

- `.venv/`, `__pycache__/` ni archivos temporales.
- Los borradores `figuras/prueba_*.png` como si fueran entregables.
- Figuras definitivas obtenidas con datos inventados.
- Cambios en `main.py` o `calculo_vectorial.py` sin coordinación.

Para Emmanuel y Aarón, el elemento principal es el contrato de datos de la sección 2. Para Sarid se entregarán posteriormente los tres PNG definitivos y sus rutas.

## 8. Orden de ejecución y criterios de terminado

1. **Completar auditoría:** registrar hallazgos y decidir correcciones mínimas.
2. **Corregir y documentar:** aplicar solamente los cambios necesarios en `graficas.py` y actualizar esta guía.
3. **Automatizar pruebas:** crear y ejecutar la suite aislada; no usar las rutas definitivas.
4. **Acordar integración:** confirmar contrato y `requirements.txt` con Emmanuel y Aarón.
5. **Integrar:** ejecutar `main.py` con datos y resultados reales.
6. **Revisar entrega:** comprobar PNG a 300 DPI, legibilidad e inclusión en el PDF.

El módulo se considerará terminado cuando las pruebas automatizadas pasen, la integración real genere las tres figuras, la lista de revisión se complete con evidencia y el equipo pueda explicar qué representa cada elemento visual.
