# Proyecto final de Cálculo Vectorial
## Opción A: Entrenamiento de una neurona artificial

**Carrera:** Ingeniería en Tecnologías de la Información e Innovación Digital  
**Cuatrimestre:** 4.º  
**Tema:** Derivadas direccionales, vector gradiente y aplicaciones de optimización  
**Entrega y defensa:** jueves 3 de diciembre de 2026, durante la clase

> **Estado actual del repositorio:** el proyecto está en etapa de implementación. Este README establece la explicación, las responsabilidades y las instrucciones que deberá cumplir el programa. Los comandos de ejecución funcionarán cuando estén terminados `main.py`, `calculo_vectorial.py`, `graficas.py` y `requirements.txt`; todavía no se presuponen resultados numéricos ni figuras generadas.

---

## Contenido

1. [¿Qué vamos a construir?](#1-qué-vamos-a-construir)
2. [Equipo y responsabilidades](#2-equipo-y-responsabilidades)
3. [Base matemática explicada con calma](#3-base-matemática-explicada-con-calma)
4. [Parámetros que podremos cambiar](#4-parámetros-que-podremos-cambiar)
5. [Instalación y ejecución desde cero](#5-instalación-y-ejecución-desde-cero)
6. [Estructura del repositorio](#6-estructura-del-repositorio)
7. [Figuras obligatorias](#7-figuras-obligatorias)
8. [Validación matemática](#8-validación-matemática)
9. [Preparación de la defensa en vivo](#9-preparación-de-la-defensa-en-vivo)
10. [Forma de trabajo y commits](#10-forma-de-trabajo-y-commits)
11. [Solución de problemas comunes](#11-solución-de-problemas-comunes)
12. [Limitaciones y entregables](#12-limitaciones-y-entregables)
13. [Glosario mínimo](#13-glosario-mínimo)

---

## 1. ¿Qué vamos a construir?

Vamos a crear un **simulador numérico en Python** que entrene una sola neurona artificial. El ejemplo aplicado será la relación entre la **carga de un servidor** y su **latencia**, usando datos sintéticos: los generaremos nosotros mediante una relación conocida y les añadiremos ruido.

En este proyecto, la neurona será un modelo lineal con dos parámetros:

- `w`: indica cuánto cambia la latencia estimada cuando cambia la carga.
- `b`: es el valor base o sesgo de la latencia estimada.
- `x`: es la carga del servidor usada como entrada.
- `ŷ`: es la latencia que predice la neurona.
- `y`: es la latencia sintética que el modelo intenta aproximar.

La predicción se calcula así:

$$
\hat{y}=wx+b.
$$

**Entrenar no significa que la neurona piense.** Significa ajustar poco a poco `w` y `b` para reducir una función que mide el error de sus predicciones. Ese ajuste se hará con **descenso del gradiente**, programado por nosotros.

### Idea intuitiva

Imaginemos la función de error como el relieve de una montaña:

- Cada punto `(w, b)` representa una combinación de parámetros.
- La altura `L(w, b)` representa cuánto se equivoca la neurona.
- El gradiente señala la dirección local de mayor aumento del error.
- Por eso avanzamos en la dirección contraria, `-∇L`, para intentar bajar hacia un mínimo.

El flujo general será:

```text
Datos sintéticos con ruido
        ↓
Predicciones ŷ = wx + b
        ↓
Cálculo de la pérdida L(w, b)
        ↓
Gradiente mediante diferencias finitas centradas
        ↓
Actualización de w y b con descenso del gradiente
        ↓
Repetir hasta cumplir un criterio de paro
        ↓
Validar, exportar las tres figuras y analizar resultados
```

También compararemos la dirección `-∇L` con direcciones subóptimas y observaremos qué sucede cuando la tasa de aprendizaje es demasiado pequeña, adecuada o demasiado grande.

### Objetivo académico

No buscamos únicamente obtener una recta que se ajuste a los datos. El objetivo principal es **demostrar cómo las derivadas parciales, el gradiente y la derivada direccional permiten construir y entender un algoritmo de optimización**.

---

## 2. Equipo y responsabilidades

Las instrucciones detalladas de cada integrante están en su propio documento de `docs/`. Cada guía explica qué hacer, qué estudiar, dónde guardar el trabajo y qué dominar para la defensa.

| Integrante | Responsabilidad principal | Archivos principales | Guía individual |
|---|---|---|---|
| **Emmanuel** | Coordinación Scrum, datos sintéticos, pérdida y algoritmos numéricos. | `calculo_vectorial.py` y `docs/emmanuel_scrum_calculo.md` | [Abrir guía de Emmanuel](docs/emmanuel_scrum_calculo.md) |
| **Carlos** | Tres figuras a 300 DPI, exportación e interpretación visual; apoyo técnico. | `graficas.py`, `latex/figuras/` y `docs/carlos_graficas_revision.md` | [Abrir guía de Carlos](docs/carlos_graficas_revision.md) |
| **Aarón** | Flujo principal, parámetros, validaciones, dependencias e instrucciones de ejecución. | `main.py`, `requirements.txt`, `README.md` y `docs/aaron_main_repo.md` | [Abrir guía de Aarón](docs/aaron_main_repo.md) |
| **Sarid** | Cálculos manuales, tabla comparativa, reporte LaTeX y apoyo en la defensa. | `latex/reporte_final.tex`, `latex/bibliografia.bib`, `latex/reporte_final.pdf` y `docs/sarid_latex_analitico.md` | [Abrir guía de Sarid](docs/sarid_latex_analitico.md) |

### Responsabilidad de todos

- **Una sola configuración:** Aarón concentra los parámetros en `main.py`; Emmanuel, Carlos y Sarid trabajarán sobre ese mismo caso para que código, figuras y cálculos sean comparables.
- **Cada archivo tiene un dueño principal:** cualquier cambio en un módulo ajeno se comenta primero con su responsable y con Emmanuel.
- **Cada cosa en su lugar:** código en los `.py`, notas en `docs/`, figuras finales en `latex/figuras/` y reporte en `latex/`.
- **Nada de resultados inventados:** los números deben salir del programa o de cálculos manuales verificables.
- **Todos ejecutan y explican:** aunque cada persona tenga una especialidad, todos deben saber correr el programa y explicar el efecto de `P₀`, `v` y `α`.
- **Todos dejan evidencia:** cada integrante debe realizar commits de su propio trabajo y participar en las revisiones de integración.
- **Carlos apoya, no sustituye:** su revisión técnica ayuda al equipo, pero cada responsable sigue siendo responsable de entender y completar su parte.


---

## 3. Base matemática explicada con calma

### 3.1 Datos sintéticos

Generaremos `n` pares `(xᵢ, yᵢ)`. Las entradas representarán carga del servidor y las salidas seguirán una relación lineal con ruido:

$$
y_i=w_{\text{real}}x_i+b_{\text{real}}+\varepsilon_i,
$$

donde `εᵢ` representa una perturbación aleatoria. Usaremos una **semilla fija** para poder repetir exactamente el mismo experimento.

Estos datos sirven para una demostración educativa; no constituyen mediciones reales del rendimiento de un servidor.

### 3.2 Función de pérdida

Usaremos el **error cuadrático medio** (MSE, por sus siglas en inglés):

$$
L(w,b)=\frac{1}{n}\sum_{i=1}^{n}\left(y_i-(wx_i+b)\right)^2.
$$

- Si las predicciones se acercan a los datos, la pérdida disminuye.
- Si las predicciones se alejan, la pérdida aumenta.
- El cuadrado evita que errores positivos y negativos se cancelen y penaliza con más fuerza los errores grandes.

Para validación analítica, sus derivadas parciales son:

$$
\frac{\partial L}{\partial w}
=-\frac{2}{n}\sum_{i=1}^{n}x_i\left(y_i-wx_i-b\right),
$$

$$
\frac{\partial L}{\partial b}
=-\frac{2}{n}\sum_{i=1}^{n}\left(y_i-wx_i-b\right).
$$

Estas expresiones se usarán para comprobar los cálculos hechos a mano, **no para sustituir el cálculo numérico solicitado en el módulo principal**.

### 3.3 Diferencias finitas centradas

El programa aproximará cada derivada parcial evaluando la pérdida a ambos lados del punto actual:

$$
\frac{\partial L}{\partial w}(w,b)
\approx
\frac{L(w+h,b)-L(w-h,b)}{2h},
$$

$$
\frac{\partial L}{\partial b}(w,b)
\approx
\frac{L(w,b+h)-L(w,b-h)}{2h}.
$$

El parámetro `h` debe ser positivo:

- Un `h` grande puede producir un error de aproximación considerable.
- Un `h` extremadamente pequeño puede aumentar el error de redondeo de la computadora.
- Por eso compararemos varios valores de `h`, en lugar de suponer que “más pequeño siempre es mejor”.

### 3.4 Vector gradiente

El gradiente reúne las derivadas parciales:

$$
\nabla L(w,b)=
\begin{bmatrix}
\dfrac{\partial L}{\partial w}\\[4pt]
\dfrac{\partial L}{\partial b}
\end{bmatrix}.
$$

Indica la dirección de **mayor crecimiento local** de la pérdida y su norma indica qué tan pronunciado es ese crecimiento. La dirección de descenso más pronunciado localmente es `-∇L`.

### 3.5 Derivada direccional y normalización

Para estudiar el cambio de la pérdida en una dirección `v = (v_w, v_b)`, primero comprobaremos que no sea el vector cero y después lo normalizaremos:

$$
\mathbf{u}=\frac{\mathbf{v}}{\|\mathbf{v}\|},
\qquad \mathbf{v}\neq\mathbf{0}.
$$

La derivada direccional se calculará como:

$$
D_{\mathbf{u}}L=\nabla L\cdot\mathbf{u}.
$$

Su signo tiene una interpretación directa:

- `DᵤL > 0`: la pérdida aumenta localmente al avanzar en la dirección `u`.
- `DᵤL < 0`: la pérdida disminuye localmente.
- `DᵤL = 0`: no hay cambio instantáneo en esa dirección; geométricamente, la dirección es tangente a una curva de nivel.

Como `u` es unitario, debe cumplirse:

$$
-\|\nabla L\|\leq D_{\mathbf{u}}L\leq\|\nabla L\|.
$$

La igualdad superior se alcanza en la dirección del gradiente y la inferior en la dirección contraria. Verificaremos esta desigualdad con vectores aleatorios no nulos.

**Importante:** en el descenso del gradiente ordinario, el vector `v` sirve para el estudio de la derivada direccional y las comparaciones; no sustituye automáticamente a `-∇L` como dirección de entrenamiento.

### 3.6 Descenso del gradiente

Si `Pₖ = (wₖ, bₖ)` es la posición actual, la actualización será:

$$
P_{k+1}=P_k-\alpha\nabla L(P_k),
$$

donde `α > 0` es la tasa de aprendizaje. En componentes:

$$
w_{k+1}=w_k-\alpha\frac{\partial L}{\partial w},
\qquad
b_{k+1}=b_k-\alpha\frac{\partial L}{\partial b}.
$$

El algoritmo se detendrá cuando la norma del gradiente sea menor que una tolerancia o cuando alcance el número máximo de iteraciones. También se registrarán `w`, `b` y `L` en cada paso para construir las figuras y la tabla de resultados.

Llegar al máximo de iteraciones **no significa necesariamente que el algoritmo convergió**; el programa deberá distinguir ambos casos.

---

## 4. Parámetros que podremos cambiar

Los parámetros estarán reunidos en un bloque claramente identificado de `main.py`, no dispersos dentro del código. Sus valores iniciales se justificarán cuando se implemente y valide el experimento.

| Parámetro | Significado sencillo | ¿Qué puede ocurrir al cambiarlo? |
|---|---|---|
| `P₀ = (w₀, b₀)` | Punto donde comienza el entrenamiento. | Cambia la trayectoria y puede cambiar el número de iteraciones necesarias. No modifica los datos. |
| `v` | Dirección usada para calcular `DᵤL` y dibujar el vector de prueba. | Cambia el valor y el signo de la derivada direccional. Debe ser distinto de `(0, 0)`. |
| `α` | Tamaño del paso de descenso. | Muy pequeño: avance lento. Adecuado: descenso estable. Muy grande: oscilaciones o divergencia. |
| `h` | Separación usada en las diferencias finitas. | Altera la precisión de las derivadas aproximadas; no debe ser cero ni negativo. |
| `tolerancia` | Precisión exigida para detener el entrenamiento. | Una tolerancia menor suele exigir más iteraciones. |
| `max_iter` | Límite de seguridad de iteraciones. | Evita ejecuciones indefinidas; debe ser un entero positivo. |
| `semilla` | Controla la secuencia aleatoria. | Permite reproducir los mismos datos y comparaciones. |
| `n` y ruido | Cantidad de datos y variabilidad añadida. | Modifican el conjunto de entrenamiento y la dificultad del ajuste. |

Para que una comparación de tasas sea justa, las ejecuciones deberán usar **los mismos datos, el mismo punto inicial y la misma semilla**, cambiando únicamente `α`.

---

## 5. Instalación y ejecución desde cero

Estas instrucciones están pensadas para **Windows y Visual Studio Code**. No necesitas conocer Python para seguirlas, pero conviene ejecutar los pasos en orden.

### 5.1 Programas necesarios

- **Python 3.10 o superior recomendado**, descargado desde [python.org](https://www.python.org/downloads/). Durante la instalación en Windows, conviene marcar la opción **Add Python to PATH**.
- **Visual Studio Code**, para editar los archivos y abrir una terminal.
- **Git**, para registrar los cambios y conservar el historial de commits del equipo.
- Para compilar el reporte se necesitará además una distribución de LaTeX; no es necesaria para ejecutar el simulador.

Las dependencias científicas permitidas son:

- **NumPy:** arreglos, operaciones vectoriales y generación de datos.
- **Matplotlib:** construcción y exportación de gráficas.
- **SymPy:** únicamente opcional para verificación simbólica.

No utilizaremos TensorFlow, PyTorch, scikit-learn, optimizadores automáticos ni funciones de diferenciación automática para resolver el entrenamiento.

### 5.2 Abrir la carpeta correcta

1. Abre Visual Studio Code.
2. Selecciona **Archivo → Abrir carpeta**.
3. Abre `proyecto-calculo-vectorial`, es decir, la carpeta donde están `main.py` y este README.
4. En el menú, selecciona **Terminal → Nuevo terminal**.
5. La terminal debe quedar ubicada en esa carpeta. Los comandos siguientes se escriben ahí y se confirman con Enter.

### 5.3 Comprobar Python

En PowerShell:

```powershell
py --version
```

Si aparece una versión de Python, podemos continuar. Si el comando no existe, revisa la sección de problemas comunes antes de instalar dependencias.

### 5.4 Crear y activar un entorno virtual

Un entorno virtual guarda las dependencias del proyecto sin mezclarlas con otros programas de Python. Se crea una sola vez:

```powershell
py -m venv .venv
```

Para activarlo en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Cuando está activo, suele aparecer `(.venv)` al principio de la línea de la terminal. La activación debe repetirse cada vez que abras una terminal nueva.

### 5.5 Instalar las dependencias

Con el entorno virtual activo:

```powershell
python -m pip install -r requirements.txt
```

Este comando lee las dependencias que Aarón registrará en `requirements.txt`. No instala LaTeX ni hace falta instalar los paquetes uno por uno manualmente.

### 5.6 Ejecutar el simulador

Cuando los módulos estén implementados:

```powershell
python main.py
```

Una ejecución completa deberá:

1. Validar los parámetros de entrada.
2. Generar los datos sintéticos con la semilla configurada.
3. Mostrar la configuración utilizada, sin ocultar los valores de `α`, `h` o `P₀`.
4. Ejecutar el entrenamiento y las comparaciones.
5. Informar la pérdida inicial y final, los parámetros obtenidos y el motivo de paro.
6. Mostrar la derivada direccional y el resultado de la comprobación del teorema.
7. Exportar automáticamente las tres figuras solicitadas, sin tener que guardar capturas manualmente.

Los resultados definitivos se guardarán como:

```text
latex/figuras/figura_1_superficie_3d.png
latex/figuras/figura_2_curvas_nivel.png
latex/figuras/figura_3_convergencia.png
```

Guardarlas directamente en `latex/figuras/` evita mantener copias distintas y cumple la estructura de entrega del reporte.

### 5.7 Si Windows bloquea la activación del entorno

No es indispensable cambiar la configuración de seguridad del sistema. Se puede llamar directamente al Python del entorno virtual:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Estos dos comandos sustituyen la activación; no son pasos adicionales si el entorno ya está activo.

---

## 6. Estructura del repositorio

```text
proyecto-calculo-vectorial/
├── README.md
├── requirements.txt
├── main.py
├── calculo_vectorial.py
├── graficas.py
├── docs/
│   ├── contexto_y_recomendaciones.md
│   ├── emmanuel_scrum_calculo.md
│   ├── carlos_graficas_revision.md
│   ├── aaron_main_repo.md
│   └── sarid_latex_analitico.md
├── figuras/
│   └── .gitkeep
└── latex/
    ├── reporte_final.tex
    ├── bibliografia.bib
    ├── figuras/
    │   └── .gitkeep
    └── reporte_final.pdf
```

### ¿Para qué sirve cada parte?

| Ruta | Propósito |
|---|---|
| `main.py` | Punto de entrada: configuración, validaciones, orden de ejecución y conexión entre módulos. |
| `calculo_vectorial.py` | Pérdida, derivadas numéricas, gradiente, derivada direccional, teorema y descenso del gradiente. |
| `graficas.py` | Recibe resultados y crea las tres figuras; no debe esconder otro algoritmo de optimización. |
| `requirements.txt` | Versiones o nombres de las dependencias necesarias para reproducir la ejecución. |
| `docs/` | Acuerdos, seguimiento Scrum, revisiones técnicas y apoyo analítico del equipo. |
| `figuras/` | Espacio para pruebas o borradores; no será la fuente definitiva de las figuras del reporte. |
| `latex/figuras/` | Imágenes finales exportadas por Python e incorporadas al documento LaTeX. |
| `reporte_final.tex` | Fuente del reporte, elaborado obligatoriamente con la plantilla oficial. |
| `bibliografia.bib` | Fuentes bibliográficas citadas en el reporte. |
| `reporte_final.pdf` | Reporte compilado para la entrega. |
| `.gitkeep` | Permite conservar carpetas vacías en Git; no es un archivo del simulador. |

La regla principal de esta arquitectura es que **`main.py` coordina, `calculo_vectorial.py` calcula y `graficas.py` visualiza**. Mantener esas responsabilidades separadas facilita revisar, corregir y explicar el programa.

---

## 7. Figuras obligatorias

Carlos programará las figuras en `graficas.py` usando los resultados producidos por el módulo matemático. Las tres deberán tener título, nombres de ejes, unidades o etiquetas comprensibles, leyenda cuando sea necesaria y una resolución de **300 DPI**.

### Figura 1: superficie 3D con trayectoria

Debe mostrar:

- La superficie `z = L(w, b)` en un rango que permita apreciar el relieve.
- La trayectoria completa seguida por los puntos `(wₖ, bₖ, L(wₖ, bₖ))`.
- Una marca para el punto inicial y otra para el punto final.
- Ejes identificados como `w`, `b` y `L(w, b)`.

**Interpretación:** permite ver cómo el algoritmo se desplaza sobre la superficie de error. Una trayectoria que oscila o se aleja también es un resultado útil y debe analizarse, no ocultarse.

### Figura 2: curvas de nivel, gradiente y dirección de prueba

Debe combinar:

- Un mapa `contour` de `L(w, b)`.
- El campo del gradiente con `quiver`.
- El vector unitario `u` obtenido a partir de `v` en el punto de prueba.
- Los pasos del algoritmo y sus puntos inicial y final.

**Interpretación:** el gradiente es perpendicular a la curva de nivel que pasa por el punto. Los pasos deben seguir la dirección contraria al gradiente numérico utilizado en el entrenamiento.

Si se normalizan flechas para que el dibujo sea legible, eso deberá indicarse: es una decisión visual y **no cambia los valores matemáticos del gradiente**.

### Figura 3: convergencia para distintas tasas

Debe mostrar el número de iteración en el eje horizontal y `L(w, b)` en el vertical, comparando **al menos dos valores de `α`** bajo las mismas condiciones iniciales.

**Interpretación:**

- Una curva que desciende muy lentamente puede indicar una tasa pequeña o un problema de escala.
- Una curva que desciende de manera sostenida sugiere convergencia durante las iteraciones observadas.
- Oscilaciones, crecimiento o valores no finitos pueden indicar una tasa demasiado grande.
- Si se usa escala logarítmica, deberá explicarse y los valores de pérdida deberán ser compatibles con ella.

Las imágenes se generarán como archivos `.png` dentro de `latex/figuras/`; no se aceptarán capturas de pantalla como sustituto de la exportación automática.

---

## 8. Validación matemática

El programa no se considerará validado únicamente porque “corre sin errores”. Debemos comprobar que sus resultados tengan sentido matemático.

### 8.1 Cálculos manuales de Sarid

Con los mismos datos y parámetros usados por Python, el reporte deberá incluir:

1. El planteamiento explícito de `L(w, b)`.
2. La obtención analítica de `∇L`.
3. La normalización de `v` y el cálculo de `DᵤL` en `P₀`.
4. Las primeras **dos iteraciones manuales** del descenso del gradiente.
5. La comparación con los valores numéricos obtenidos en Python.

Los cálculos manuales y el programa deben usar exactamente la misma convención del MSE, incluido el divisor `n`; de lo contrario, la comparación no será válida.

### 8.2 Tabla variando `h`

Para varios valores positivos de `h`, se registrarán como mínimo:

- El valor de `h` utilizado.
- Cada derivada exacta en el punto de evaluación.
- Cada derivada obtenida por diferencias finitas.
- El error absoluto.
- El error relativo porcentual.

Para una cantidad escalar, los errores se calcularán como:

$$
\text{Error absoluto}=|x_{\text{numérico}}-x_{\text{exacto}}|,
$$

$$
\text{Error relativo porcentual}
=\frac{|x_{\text{numérico}}-x_{\text{exacto}}|}{|x_{\text{exacto}}|}\times 100.
$$

Si el valor exacto es cero, el error relativo anterior no está definido; se reportará el error absoluto y se explicará esa situación en lugar de dividir entre cero.

### 8.3 Comprobaciones mínimas del programa

- `v = (0, 0)` debe producir una validación clara, no una división silenciosa entre cero.
- Después de normalizar, la norma de `u` debe ser aproximadamente uno.
- Para vectores aleatorios no nulos se verificará `-||∇L|| ≤ DᵤL ≤ ||∇L||`, considerando una pequeña tolerancia numérica.
- El gradiente numérico se comparará con el gradiente analítico en uno o varios puntos.
- Las primeras dos iteraciones de Python deben coincidir, salvo error de aproximación, con las iteraciones manuales.
- Se compararán al menos dos tasas `α` y direcciones de avance favorables y subóptimas.
- Los datos y resultados serán reproducibles mediante una semilla fija.
- No se escribirán a mano resultados “finales” dentro del código para simular un entrenamiento.

SymPy puede apoyar una verificación independiente, pero el gradiente empleado por el descenso principal debe obtenerse mediante las diferencias finitas implementadas por el equipo.

---

## 9. Preparación de la defensa en vivo

La sesión tendrá una duración de **10 a 15 minutos**:

- Aproximadamente **5 minutos de exposición** del caso de estudio, modelo matemático y gráficas.
- Entre **5 y 10 minutos de ejecución y prueba en vivo**, con cambios solicitados por el docente.

### 9.1 Antes de presentar

- Probar desde cero la instalación y la ejecución en el equipo que se usará.
- Dejar un caso base validado y guardar sus resultados para comparar.
- Confirmar que las tres figuras se exportan correctamente a 300 DPI.
- Tener el reporte y el código abiertos, sin depender de internet durante la ejecución.
- Revisar que todos sepan dónde se cambian los parámetros en `main.py`.
- No memorizar únicamente valores numéricos: hay que entender las causas y las gráficas.

### 9.2 Cómo responder a un cambio en vivo

1. Indicar cuál es el valor actual del parámetro.
2. Cambiar **solo el parámetro solicitado** para aislar su efecto.
3. Conservar la misma semilla y los mismos datos cuando se quiera comparar.
4. Ejecutar de nuevo y observar pérdida, iteraciones, motivo de paro y figuras.
5. Explicar primero la predicción matemática y después lo observado.
6. Si el resultado no es el esperado, revisar validaciones y escala sin ocultar el resultado.

| Cambio solicitado | Lo que debemos explicar |
|---|---|
| **`P₀` más lejano o distinto** | Los datos y la superficie no cambian, pero sí el inicio y la trayectoria. Puede variar el número de iteraciones. Si converge, puede llegar al mismo mínimo; con un límite de iteraciones, el resultado final puede ser diferente. |
| **Otra dirección `v`** | Se vuelve a normalizar para obtener `u` y cambia `DᵤL = ∇L·u`. En el descenso ordinario, este cambio no modifica la trayectoria de entrenamiento; sí modifica el análisis direccional y el vector mostrado en la Figura 2. |
| **`α` más pequeña** | Normalmente produce pasos más cortos y puede requerir más iteraciones. No garantiza por sí sola una mejor precisión si se alcanza antes el máximo de iteraciones. |
| **`α` más grande** | Puede acelerar el descenso dentro de un rango adecuado, pero también causar oscilaciones, aumento de pérdida o divergencia. El resultado depende de la escala del problema. |
| **Otro valor de `h`** | Cambia la aproximación de las derivadas. Debe relacionarse con la tabla de error, no confundirse con el tamaño del paso `α`. |

Una divergencia causada por una tasa mal elegida no necesariamente significa que el programa esté mal: puede ser precisamente el fenómeno que se quiere estudiar. Sin embargo, valores `NaN`, errores de dimensiones o validaciones ausentes sí deben revisarse técnicamente.

### 9.3 Qué explicará cada integrante

- **Emmanuel:** cómo se construye `L`, cómo se obtiene `∇L`, por qué se usa `-∇L`, cómo se comprueba el teorema y cómo se decide el paro.
- **Carlos:** cómo se construye cada figura, qué representan flechas, trayectorias y curvas, y qué diferencia hay entre las tasas comparadas.
- **Aarón:** dónde se configuran los parámetros, cómo se validan, cómo se ejecuta el flujo completo y cómo se interpretan errores de entrada.
- **Sarid:** desarrollo analítico, normalización, primeras dos iteraciones, tabla variando `h` y correspondencia entre cálculo manual y numérico.

Todos deben estar preparados para relacionar su explicación con `P₀`, `v` y `α`, aunque el docente formule la pregunta fuera de su módulo principal.

---

## 10. Forma de trabajo y commits

Emmanuel coordinará el trabajo como Scrum Master con tareas pequeñas, responsables claros y revisiones breves. Los acuerdos, bloqueos y avances podrán registrarse en `docs/`, especialmente en `emmanuel_scrum_calculo.md`.

### Flujo básico

1. Antes de trabajar, actualizar la copia local y revisar el estado del repositorio.
2. Elegir una tarea concreta y avisar qué archivo se modificará.
3. Hacer una prueba local antes de compartir el cambio.
4. Crear commits pequeños y descriptivos; no acumular todo el proyecto en un solo commit.
5. Revisar la integración antes de ejecutar el caso final.
6. Actualizar README o documentación cuando cambie una instrucción, ruta o parámetro.

Comandos básicos, ejecutados dentro de la carpeta del repositorio:

```powershell
git status
git pull
git add main.py
git commit -m "feat: conecta el flujo principal del simulador"
git push
```

`git add main.py` es solo un ejemplo: debe escribirse el nombre del archivo realmente terminado. Conviene agregar archivos de forma explícita y **no incluir** `.venv/`, `__pycache__/`, archivos temporales de LaTeX ni resultados accidentales.

Ejemplos de mensajes útiles:

```text
feat: implementa la pérdida MSE y diferencias centradas
feat: agrega el descenso del gradiente con criterio de paro
feat: exporta la superficie 3D a 300 DPI
fix: valida el vector dirección distinto de cero
docs: documenta la ejecución y los parámetros en vivo
docs: integra los cálculos manuales y la tabla comparativa
```

El historial debe reflejar aportaciones de **todos los integrantes**, como solicita la rúbrica. No se deben hacer commits a nombre de otra persona ni borrar trabajo ajeno para resolver apresuradamente un conflicto.

### Una tarea está terminada cuando

- Hace lo que acordó el equipo y no utiliza una “caja negra” prohibida.
- Maneja entradas inválidas de forma comprensible.
- Se integra con los demás módulos sin inventar datos de entrada.
- Puede ser explicada por su responsable.
- Fue revisada y probada en el flujo principal.
- Su documentación y sus rutas siguen siendo correctas.

---

## 11. Solución de problemas comunes

| Mensaje o situación | Qué revisar |
|---|---|
| `py no se reconoce como un comando` | Verifica la instalación de Python y del lanzador `py`. Cierra y vuelve a abrir la terminal después de instalar. Si es necesario, reinstala desde python.org y habilita Python en PATH. |
| `python no se reconoce` | Comprueba si el entorno virtual está activo. También puedes usar directamente `.\.venv\Scripts\python.exe`. |
| Windows bloquea `Activate.ps1` | No cambies políticas globales: utiliza los comandos directos de la sección 5.7. |
| `No module named numpy` o `matplotlib` | Activa `.venv` y ejecuta `python -m pip install -r requirements.txt`. Comprueba que no estés usando otro intérprete de Python. |
| No aparece `main.py` al ejecutar | Probablemente la terminal está en otra carpeta. Abre la raíz `proyecto-calculo-vectorial` y vuelve a intentarlo. |
| `requirements.txt` no instala lo necesario | Revisa que el archivo ya contenga las dependencias acordadas y que estés usando el archivo de este proyecto. |
| El vector dirección produce una excepción | Comprueba que `v` no sea `(0, 0)`. Rechazar ese vector es correcto porque no se puede normalizar. |
| La pérdida aumenta, oscila o se vuelve infinita | Revisa primero `α`, la escala de los datos y el punto inicial. Conserva el caso base y cambia una sola variable para diagnosticar. |
| Las derivadas numéricas son imprecisas | Compara varios valores de `h` y revisa que las diferencias sean centradas. No uses automáticamente un `h` extremadamente pequeño. |
| No se generan las figuras | Lee el primer error de la terminal, verifica que el entrenamiento terminó y comprueba `latex/figuras/`. No sustituyas las figuras por capturas de pantalla. |
| LaTeX no encuentra una imagen | Ejecuta primero el simulador y revisa que el nombre y la ruta usados en el `.tex` coincidan exactamente con el archivo exportado. |
| Git muestra una ruta o cambios inesperados | Ejecuta `git status`, detente y coordina con Emmanuel antes de borrar, forzar o sobrescribir cambios. |

Cuando aparezca un error, conviene leer **la primera línea del mensaje que identifica el archivo y la causa**, no copiar comandos al azar. Si el equipo no lo entiende, se registra el mensaje completo y se solicita revisión técnica.

---

## 12. Limitaciones y entregables

### 12.1 Limitaciones del simulador

- Es una **sola neurona lineal**, no una red neuronal profunda.
- Solo aprende dos parámetros, `w` y `b`, para una entrada y una salida.
- Los datos son sintéticos; las conclusiones no deben generalizarse a servidores reales.
- El MSE penaliza fuertemente los valores atípicos y no siempre será la mejor pérdida para otros problemas.
- Las diferencias finitas producen aproximaciones y son sensibles a `h` y al redondeo.
- `-∇L` indica la mayor disminución **local**, pero no garantiza convergencia con cualquier `α` o en un número fijo de iteraciones.
- La escala de los datos influye en el condicionamiento del problema y en las tasas que funcionan.
- El objetivo es educativo; el simulador no pretende competir con bibliotecas profesionales de aprendizaje automático.

### 12.2 Contenido obligatorio del reporte

Sarid elaborará el documento en la **plantilla oficial de LaTeX**, con una extensión sugerida de 3 a 6 páginas:

1. Resumen del problema, método y hallazgos.
2. Marco teórico de derivadas, gradiente, derivada direccional y diferencias finitas.
3. Desarrollo analítico y primeras dos iteraciones manuales.
4. Arquitectura del software, excepciones y fragmentos clave de código.
5. Resultados, tabla variando `h`, porcentaje de error, tres figuras e interpretación geométrica.
6. Conclusiones sobre el cálculo multivariable y el desarrollo de algoritmos.

### 12.3 Relación con la rúbrica

| Criterio | Ponderación |
|---|---:|
| Rigor matemático | 30% |
| Código en Python | 30% |
| Reporte en LaTeX | 25% |
| Presentación y defensa | 15% |
| **Total** | **100%** |

### 12.4 Lista final de entrega

- [ ] `main.py`, `calculo_vectorial.py` y `graficas.py` integrados.
- [ ] `requirements.txt` completo e instrucciones de README comprobadas.
- [ ] Gradientes y optimización implementados por el equipo, sin cajas negras.
- [ ] Cálculos manuales, dos primeras iteraciones y comparación numérica terminados.
- [ ] Tabla con distintos valores de `h` y errores correctamente calculados.
- [ ] Tres figuras automáticas a 300 DPI dentro de `latex/figuras/`.
- [ ] `reporte_final.tex`, `bibliografia.bib` y `reporte_final.pdf` completos.
- [ ] Reporte elaborado en la plantilla oficial y con citas bibliográficas.
- [ ] Repositorio o archivo ZIP preparado, con historial de commits de todos.
- [ ] Ejecución en vivo ensayada con cambios de `P₀`, `v` y `α`.
- [ ] Participación de todos preparada para la defensa del 3 de diciembre de 2026.

---

## 13. Glosario mínimo

| Término | Explicación breve |
|---|---|
| **Dato sintético** | Dato generado artificialmente para controlar y repetir un experimento. |
| **Ruido** | Variación añadida para que los datos no sigan una relación perfecta. |
| **Neurona lineal** | Modelo que calcula `ŷ = wx + b`; aquí no representa una red profunda. |
| **Pérdida** | Número que mide el error del modelo; el entrenamiento intenta reducirlo. |
| **Iteración** | Una actualización de `w` y `b` durante el entrenamiento. |
| **Derivada parcial** | Cambio de una función respecto a una variable mientras las demás se mantienen fijas. |
| **Gradiente** | Vector de derivadas parciales; apunta al mayor crecimiento local. |
| **Diferencias finitas** | Método que aproxima derivadas usando evaluaciones de la función. |
| **Derivada direccional** | Ritmo de cambio de una función al moverse en una dirección unitaria. |
| **Tasa de aprendizaje `α`** | Factor que controla la longitud del paso en cada actualización. |
| **Paso `h`** | Separación usada para aproximar derivadas; no es lo mismo que `α`. |
| **Convergencia** | Acercamiento progresivo a una solución bajo el criterio definido. |
| **Divergencia** | Comportamiento en el que los valores crecen o se alejan en lugar de estabilizarse. |
| **Semilla aleatoria** | Valor que permite reproducir la misma secuencia de números aleatorios. |
| **Entorno virtual** | Instalación aislada de dependencias para este proyecto. |
| **DPI** | Puntos por pulgada; 300 DPI es la resolución mínima solicitada para las figuras. |

> **Idea clave para todo el equipo:** no basta con copiar código ni obtener una gráfica atractiva. Debemos poder explicar qué calcula cada función, por qué se normaliza una dirección, cómo se valida el resultado y qué efecto tiene cada parámetro.

