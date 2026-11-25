# Guía de Aarón — Integración, validaciones y ejecución

[Volver al README principal](../README.md)

## Objetivo de tu responsabilidad

Lograr que cualquier integrante pueda ejecutar el proyecto, cambiar los parámetros solicitados y comprender lo que aparece en pantalla.

## 1. Lo que tienes que hacer

1. Reunir en un bloque visible de `main.py` los parámetros `P₀`, `v`, `α`, `h`, tolerancia, máximo de iteraciones, semilla, cantidad de datos y ruido.
2. Importar y llamar las funciones de `calculo_vectorial.py` en el orden correcto; no vuelvas a escribir dentro de `main.py` las derivadas o el optimizador.
3. Entregar a `graficas.py` los resultados que necesita y comprobar que las figuras se guarden en las rutas acordadas.
4. Validar antes de entrenar: `α > 0`, `h > 0`, tolerancia positiva, máximo entero positivo, datos no vacíos y `v ≠ (0, 0)`.
5. Mostrar mensajes claros ante una entrada incorrecta. No uses un `except` general para ocultar un error y continuar como si nada hubiera pasado.
6. Presentar en consola la configuración, la pérdida inicial y final, los parámetros obtenidos, el motivo de paro y las rutas de las figuras.
7. Registrar en `requirements.txt` las dependencias utilizadas y mantener actualizados los comandos del README.
8. Probar el flujo completo después de cada integración y dejar un caso base estable para la demostración en vivo.

## 2. Temas que debes estudiar

- **Python básico:** variables, tipos de datos, tuplas, listas, funciones e importaciones.
- **Punto de entrada:** significado de `if __name__ == "__main__":` y orden de ejecución.
- **Errores y validaciones:** `ValueError`, `try`, `except`, mensajes útiles y diferencia entre error de entrada y error de programación.
- **Rutas:** carpetas relativas al proyecto y uso básico de `pathlib`.
- **Entorno de ejecución:** Python, `.venv`, pip, `requirements.txt` y selección del intérprete en VS Code.

## 3. Dónde guardar cada cosa

| Ruta | Qué va ahí |
|---|---|
| `main.py` | Configuración, validaciones, llamadas entre módulos, presentación de resultados y flujo principal. |
| `requirements.txt` | Dependencias necesarias para instalar y ejecutar el simulador. |
| `README.md` | Instrucciones de instalación, ejecución, parámetros y solución de problemas que debes mantener actualizadas. |
| `docs/aaron_main_repo.md` | Esta guía, decisiones de integración, pruebas realizadas, errores frecuentes y notas del repositorio. |

## 4. Lo que debes saber explicar en la defensa

- Cómo se ejecuta el proyecto y qué función se llama en cada etapa.
- Dónde se cambia cada parámetro y por qué se valida.
- Qué hacer ante un mensaje de error.
- Cómo repetir una comparación usando la misma semilla.
- Qué diferencia existe entre cambiar `P₀`, `v`, `α` y `h`.
- Por qué alcanzar el máximo de iteraciones debe mostrarse como un motivo de paro distinto de la convergencia.

## 5. Comprobaciones antes de entregar el flujo principal

Marca cada punto únicamente después de probarlo:

- [ ] `python main.py` funciona desde la raíz del proyecto con el entorno virtual activo.
- [ ] Los parámetros están reunidos y son fáciles de localizar.
- [ ] Las entradas inválidas muestran mensajes claros.
- [ ] El programa informa pérdida, parámetros finales, iteraciones y motivo de paro.
- [ ] Las tres figuras aparecen en `latex/figuras/`.
- [ ] `requirements.txt` permite instalar las dependencias en un entorno limpio.
- [ ] Los comandos del README coinciden con la ejecución real.
- [ ] El caso base se ejecuta dos veces con la misma semilla y produce resultados reproducibles.

