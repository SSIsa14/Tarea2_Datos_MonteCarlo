"""Nucleo comun de la Parte 2: estimador de Monte Carlo e intervalos de confianza."""

import math
import os

Z95 = 1.959964  # cuantil 0.975 de la normal estandar

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
FIGS = os.path.join(RAIZ, "figs")
DOC = os.path.join(RAIZ, "doc")


# --- Estimador e intervalos de confianza ---


def media_ic(valores, z=Z95):
    """Media, error estandar e intervalo de confianza de una muestra."""
    n = len(valores)
    media = sum(valores) / n
    var = sum((v - media) ** 2 for v in valores) / (n - 1)
    ee = math.sqrt(var / n)
    return {"n": n, "media": media, "var": var, "ee": ee,
            "lo": media - z * ee, "hi": media + z * ee}


def integral(f, a, b, n, fuente):
    """Estima la integral de f en [a,b] con n muestras uniformes."""
    valores = [(b - a) * f(a + (b - a) * fuente.random()) for _ in range(n)]
    return media_ic(valores)


def texto_ic(r, decimales=6):
    return (f"{r['media']:.{decimales}f} [{r['lo']:.{decimales}f}, "
            f"{r['hi']:.{decimales}f}]")


def ic_tex(r, decimales=6):
    return (f"${r['media']:.{decimales}f}$ & "
            f"$[{r['lo']:.{decimales}f},\\ {r['hi']:.{decimales}f}]$")


# --- Salida en LaTeX ---

CABECERA = ("% Generado por los scripts de src/.\n"
            "% No editar a mano: se sobrescribe en cada ejecucion.\n")


def escribir_tex(nombre, lineas):
    ruta = os.path.join(DOC, nombre)
    with open(ruta, "w") as f:
        f.write(CABECERA + "\n".join(lineas) + "\n")
    return ruta


def macro(nombre, valor):
    return f"\\newcommand{{\\{nombre}}}{{{valor}}}"


def miles(x, decimales=0):
    return f"{x:,.{decimales}f}".replace(",", "{,}")


def cientifica(x, decimales=2):
    """Notacion cientifica lista para LaTeX, en vez del 1.2e-07 de Python."""
    if x == 0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(x))))
    mantisa = x / 10 ** exp
    if round(abs(mantisa), decimales) >= 10:   # evita cosas como 10.0e4
        mantisa /= 10
        exp += 1
    return rf"${mantisa:.{decimales}f}\times 10^{{{exp}}}$"
