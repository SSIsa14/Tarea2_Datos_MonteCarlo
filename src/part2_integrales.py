"""Parte 2.2: integrales en una dimension y estudio de convergencia.

Estima las dos integrales con su intervalo de confianza, grafica el error
contra N en escala log-log y compara tres fuentes de aleatoriedad.

Uso:  python3 src/part2_integrales.py
"""

import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prngs import LCG, MersenneTwister
from montecarlo import integral, escribir_tex, macro, miles, cientifica, FIGS

SEMILLA = 141903
N_FINAL = 1_000_000     # muestras de la estimacion principal
REPETICIONES = 10       # corridas independientes del estudio de convergencia
N_MAX = 1_000_000       # N mas grande del estudio de convergencia

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})


def seno(x):
    return math.sin(math.pi * x)


def normal(x):
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


# (clave, f, version para numpy, a, b, valor exacto, etiqueta)
INTEGRALES = [
    ("seno", seno, lambda x: np.sin(np.pi * x), 0.0, 1.0, 2.0 / math.pi,
     r"(a) $\int_0^1 \sin(\pi x)\,dx$"),
    ("normal", normal, lambda x: np.exp(-x * x / 2) / np.sqrt(2 * np.pi),
     0.0, 2.0, 0.5 * math.erf(2.0 / math.sqrt(2.0)),
     r"(b) $\int_0^2 \frac{1}{\sqrt{2\pi}}e^{-x^2/2}\,dx$"),
]

NOMBRES_FUENTE = ["random", "LCG", "MT19937"]


def fuente(nombre, semilla):
    if nombre == "random":
        return random.Random(semilla)
    if nombre == "LCG":
        return LCG(semilla)
    return MersenneTwister(semilla)


# --- Estimacion principal con intervalo de confianza ---


def estimaciones():
    """Estimacion con IC del 95% para cada integral y cada fuente."""
    filas = []
    for clave, f, _, a, b, exacto, _ in INTEGRALES:
        for nombre in NOMBRES_FUENTE:
            r = integral(f, a, b, N_FINAL, fuente(nombre, SEMILLA))
            r.update({"integral": clave, "fuente": nombre, "exacto": exacto,
                      "error": abs(r["media"] - exacto),
                      "cubre": r["lo"] <= exacto <= r["hi"]})
            filas.append(r)
            print(f"  {clave:7s} {nombre:8s} {r['media']:.6f} "
                  f"[{r['lo']:.6f}, {r['hi']:.6f}]  error={r['error']:.2e}  "
                  f"{'el IC cubre' if r['cubre'] else 'el IC NO cubre'}")
    return filas


# --- Estudio de convergencia en log-log ---


def convergencia():
    """Error RMS contra N en log-log y pendiente de la recta ajustada."""
    enes = np.unique(np.logspace(1, math.log10(N_MAX), 25).astype(int))
    curvas, pendientes = {}, {}

    for nombre in NOMBRES_FUENTE:
        errores = {clave: np.zeros((REPETICIONES, len(enes)))
                   for clave, *_ in INTEGRALES}
        for rep in range(REPETICIONES):
            gen = fuente(nombre, SEMILLA + 137 * rep)
            u = np.fromiter((gen.random() for _ in range(N_MAX)),
                            dtype=float, count=N_MAX)
            for clave, _, f_np, a, b, exacto, _ in INTEGRALES:
                valores = (b - a) * f_np(a + (b - a) * u)
                medias = np.cumsum(valores) / np.arange(1, N_MAX + 1)
                errores[clave][rep] = np.abs(medias[enes - 1] - exacto)
        for clave in errores:
            curva = np.sqrt((errores[clave] ** 2).mean(axis=0))
            curvas[(nombre, clave)] = curva
            pendientes[(nombre, clave)] = np.polyfit(np.log10(enes),
                                                     np.log10(curva), 1)[0]
        print(f"  {nombre} listo")

    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (clave, _, _, _, _, _, etiqueta) in zip(ejes, INTEGRALES):
        for nombre in NOMBRES_FUENTE:
            ax.loglog(enes, curvas[(nombre, clave)], marker="o", ms=3, lw=1,
                      label=f"{nombre} (pendiente {pendientes[(nombre, clave)]:.3f})")
        referencia = curvas[("MT19937", clave)][0] * (enes / enes[0]) ** -0.5
        ax.loglog(enes, referencia, "k--", lw=1, label="$N^{-1/2}$")
        ax.set_title(etiqueta, fontsize=10)
        ax.set_xlabel("N")
        ax.set_ylabel("error absoluto (RMS)")
        ax.legend(fontsize=7)
    fig.suptitle(f"Convergencia del error ({REPETICIONES} corridas por punto)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "convergencia_1d.png"))
    plt.close(fig)

    return enes, curvas, pendientes


def escribir(filas, pendientes):
    L = [r"\begin{table}[htbp]\centering",
         r"\caption{Estimaciones con $N=\NFinal$ muestras e intervalo de "
         r"confianza del $95\%$.}",
         r"\label{tab:integrales}", r"\small",
         r"\begin{tabular}{l|l|r|l|r|c}", r"\toprule",
         r"Integral & Fuente & Estimacion & IC $95\%$ & Error & Cubre \\",
         r"\midrule"]
    for r in filas:
        etiqueta = "(a) seno" if r["integral"] == "seno" else "(b) normal"
        L.append(f"{etiqueta} & \\texttt{{{r['fuente']}}} & ${r['media']:.6f}$ & "
                 f"$[{r['lo']:.6f},\\ {r['hi']:.6f}]$ & "
                 f"{cientifica(r['error'], 1)} & "
                 f"{'si' if r['cubre'] else 'no'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    escribir_tex("tabla_integrales.tex", L)

    L = [r"\begin{table}[htbp]\centering",
         r"\caption{Pendiente ajustada al error RMS en escala log--log. "
         r"El valor teorico es $-0.5$.}",
         r"\label{tab:pendientes}",
         r"\begin{tabular}{l|r|r}", r"\toprule",
         r"Fuente & Integral (a) & Integral (b) \\", r"\midrule"]
    for nombre in NOMBRES_FUENTE:
        L.append(f"\\texttt{{{nombre}}} & {pendientes[(nombre, 'seno')]:.3f} & "
                 f"{pendientes[(nombre, 'normal')]:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    escribir_tex("tabla_pendientes.tex", L)

    mt_a = next(r for r in filas if r["integral"] == "seno"
                and r["fuente"] == "MT19937")
    mt_b = next(r for r in filas if r["integral"] == "normal"
                and r["fuente"] == "MT19937")
    L = [macro("NFinal", miles(N_FINAL)),
         macro("Repeticiones", REPETICIONES),
         macro("ExactoSeno", f"{INTEGRALES[0][5]:.6f}"),
         macro("ExactoNormal", f"{INTEGRALES[1][5]:.6f}"),
         macro("EstSeno", f"{mt_a['media']:.6f}"),
         macro("ICSeno", f"[{mt_a['lo']:.6f},\\ {mt_a['hi']:.6f}]"),
         macro("ErrorSeno", cientifica(mt_a["error"], 1)),
         macro("EstNormal", f"{mt_b['media']:.6f}"),
         macro("ICNormal", f"[{mt_b['lo']:.6f},\\ {mt_b['hi']:.6f}]"),
         macro("ErrorNormal", cientifica(mt_b["error"], 1)),
         macro("PendienteSeno", f"{pendientes[('MT19937', 'seno')]:.3f}"),
         macro("PendienteNormal", f"{pendientes[('MT19937', 'normal')]:.3f}")]
    escribir_tex("datos_integrales.tex", L)


def main():
    print("Parte 2.2: integrales en una dimension")
    print(f"  valor exacto (a) = {INTEGRALES[0][5]:.8f}")
    print(f"  valor exacto (b) = {INTEGRALES[1][5]:.8f}\n")
    filas = estimaciones()
    print("\nEstudio de convergencia...")
    _, _, pendientes = convergencia()
    for (nombre, clave), p in pendientes.items():
        print(f"  {nombre:8s} {clave:7s} pendiente = {p:.3f}")
    escribir(filas, pendientes)
    print("\nListo.")


if __name__ == "__main__":
    main()
