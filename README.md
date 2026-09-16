# Tarea 2: Números Pseudoaleatorios y Métodos de Monte Carlo

Curso Libre de Configuración 2: Análisis de Datos
Facultad de Ingeniería, Universidad del Istmo de Guatemala

Implementación desde cero de cinco generadores de números pseudoaleatorios
(LCG, cuadrados medios, Mersenne Twister, Blum Blum Shub y RANDU) y de los
métodos de Monte Carlo de la segunda parte. Ni `random` ni `numpy.random` se
usan como motor de ningún generador.

Todo el documento se reproduce corriendo cinco scripts y compilando el `.tex`.
Abajo está el procedimiento completo.

---

## 1. Requisitos

- **Python 3.10 o superior.** Verificar con:

  ```bash
  python3 --version
  ```

- **numpy y matplotlib.** Sólo se usan para graficar y para operaciones de
  resumen; los generadores y las pruebas estadísticas están hechos a mano.

  ```bash
  pip install numpy matplotlib
  ```

- **Una distribución de LaTeX**, para compilar el PDF. Cualquiera de estas
  tres opciones sirve:

  ```bash
  sudo apt install texlive-latex-extra texlive-lang-spanish   # opción 1
  ```

  ```bash
  curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh   # opción 2: tectonic
  ```

  La opción 3 es subir la carpeta `doc/` junto con `figs/` a Overleaf y
  compilar ahí, sin instalar nada.

---

## 2. Reproducir todo, paso a paso

Los comandos se corren **desde la raíz del proyecto** (la carpeta donde está
este archivo).

### Paso 1. Comprobación rápida

Antes de generar nada, conviene revisar que los dos módulos base funcionen.

```bash
python3 src/prngs.py
```

Imprime cinco valores de cada generador. Debe empezar así:

```
LCG              [0.23092, 0.41495, 0.9775, 0.40124, 0.94545]
```

```bash
python3 src/stats_tests.py
```

Compara las distribuciones programadas a mano contra valores de tabla. Los
tres resultados deben dar `0.05`.

### Paso 2. Parte 1, los cinco generadores

```bash
python3 src/part1_generators.py
```

Tarda unos 6 segundos. Genera los histogramas, aplica las cuatro pruebas de
hipótesis, hace la prueba espectral y compara contra `random` y `secrets`.

En pantalla debe aparecer, entre otras cosas:

- `RANDU: 15 planos, residuo maximo 0.0e+00`
- `colapso en i=8864, ciclo de largo 1`
- `la implementacion propia da los mismos numeros que random: True`

### Paso 3. Parte 2.2, integrales en una dimensión

```bash
python3 src/part2_integrales.py
```

Tarda unos 20 segundos. Estima las dos integrales con su intervalo de
confianza y hace el estudio de convergencia en log–log. Las pendientes
ajustadas deben salir todas cerca de `-0.5`.

### Paso 4. Parte 2.3, alta dimensión

```bash
python3 src/part2_dimension.py
```

Tarda unos 10 segundos. Estima el volumen de la bola unitaria en dimensión 2,
5, 10 y 20, y lo compara contra una rejilla determinista. En `d=20` la
estimación de Monte Carlo debe dar cero aciertos: eso no es un error, es el
resultado que se analiza en el documento.

### Paso 5. Parte 2.4, reducción de varianza

```bash
python3 src/part2_varianza.py
```

Tarda unos 16 segundos. Aplica variables antitéticas y variables de control a
las dos integrales. El factor de las antitéticas en la integral del seno debe
salir `0.50x`, o sea peor que Monte Carlo simple; también es el resultado
esperado y está explicado en el documento.

### Paso 6. Parte 2.5, el caso RANDU

```bash
python3 src/part2_randu.py
```

Tarda unos 32 segundos, que es el más lento porque repite la estimación con
100 semillas distintas. RANDU debe salir con un sesgo cercano a `+25%` en la
bola de radio 0.1, y el Mersenne Twister cerca de `+1.6%`.

### Paso 7. Compilar el documento

```bash
cd doc
tectonic -X compile tarea2.tex
```

Con una instalación normal de LaTeX, el equivalente es correr `pdflatex
tarea2.tex` **dos veces** (la primera pasada arma el índice y la segunda lo
coloca bien).

El resultado es `doc/tarea2.pdf`, de 35 páginas.

---

## 3. Qué produce cada script

Los scripts escriben sus figuras en `figs/` y sus tablas en `doc/`. El `.tex`
las lee de ahí, así que hay que correrlos antes de compilar.

| Script | Figuras | Tablas |
|---|---|---|
| `part1_generators.py` | `hist_*.png` (5), `cubo_*.png` (3), `randu_planos.png`, `ms_colapso.png`, `comparacion_librerias.png` | `datos_parte1.tex`, `pruebas_*.tex` (5), `tabla_velocidad.tex`, `tabla_librerias.tex` |
| `part2_integrales.py` | `convergencia_1d.png` | `tabla_integrales.tex`, `tabla_pendientes.tex`, `datos_integrales.tex` |
| `part2_dimension.py` | `fraccion_bola.png`, `error_mc_vs_rejilla.png` | `tabla_volumenes.tex`, `tabla_rejilla.tex`, `datos_dimension.tex` |
| `part2_varianza.py` | `reduccion_varianza.png` | `tabla_varianza.tex`, `datos_varianza.tex` |
| `part2_randu.py` | `randu_montecarlo.png` | `tabla_randu.tex`, `datos_randu.tex` |

Los archivos que empiezan con `datos_` sólo contienen definiciones de valores
y se leen desde el preámbulo del `.tex`. Ninguno de los archivos generados se
edita a mano: se sobrescriben en cada ejecución.

---

## 4. Semillas

Están fijas dentro de cada script, así que dos corridas distintas dan
exactamente los mismos resultados.

| Semilla | Dónde se usa |
|---|---|
| 141903 | LCG, cuadrados medios, Mersenne Twister y RANDU |
| 14773 | Blum Blum Shub |
| 140405 | Volumen de la bola en alta dimensión |
| 77140 | Reducción de varianza |

Dos restricciones que vale la pena conocer: RANDU necesita semilla impar
(141903 lo es) y Blum Blum Shub necesita una semilla coprima con su módulo.
Si se cambian por otras que no cumplan, el constructor avisa con un error.

---

## 5. Si algo falla

**`ModuleNotFoundError: No module named 'numpy'`**
Faltan las dependencias del paso 1: `pip install numpy matplotlib`.

**`ModuleNotFoundError: No module named 'prngs'`**
Se está corriendo el script desde otra carpeta. Los comandos van desde la
raíz del proyecto, en la forma `python3 src/nombre.py`.

**El PDF compila pero las tablas salen vacías o con signos de interrogación**
Faltó correr los scripts antes de compilar. Hay que hacer los pasos 2 al 6 y
volver a compilar.

**`Undefined control sequence` al compilar**
Igual que el anterior: el `.tex` está buscando un valor que genera alguno de
los scripts.

---

## 6. Estructura del proyecto

```
src/
  prngs.py              los cinco generadores
  stats_tests.py        chi-cuadrado, KS, autocorrelación y rachas
  montecarlo.py         estimador de Monte Carlo e intervalos de confianza
  part1_generators.py   Parte 1
  part2_integrales.py   Parte 2.2
  part2_dimension.py    Parte 2.3
  part2_varianza.py     Parte 2.4
  part2_randu.py        Parte 2.5
doc/
  tarea2.tex            el documento
  tarea2.pdf            el PDF compilado
  (tablas generadas por los scripts)
figs/                   figuras generadas por los scripts
README.md
```

Correr los cinco scripts toma alrededor de un minuto y medio en total.
