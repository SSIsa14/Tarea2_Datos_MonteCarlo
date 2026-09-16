"""Parte 2.3: volumen de la bola unitaria en d dimensiones.

Compara Monte Carlo contra una rejilla determinista y muestra por que el
muestreo uniforme se vuelve inutil cuando d crece.

Uso:  python3 src/part2_dimension.py
"""

import itertools
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prngs import MersenneTwister
from montecarlo import escribir_tex, macro, miles, cientifica, Z95, FIGS

SEMILLA = 140405
N = 200_000           # muestras de Monte Carlo por dimension
DIMENSIONES = (2, 5, 10, 20)
PRESUPUESTO = 100_000  # puntos disponibles para comparar contra la rejilla

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})


def volumen_exacto(d):
    return math.pi ** (d / 2) / math.gamma(d / 2 + 1)


# --- Los dos estimadores: aleatorio y determinista ---


def monte_carlo(d, n, gen):
    """Fraccion de puntos del cubo [-1,1]^d que caen dentro de la bola."""
    dentro = 0
    for _ in range(n):
        suma = 0.0
        for _ in range(d):
            x = 2.0 * gen.random() - 1.0
            suma += x * x
        if suma <= 1.0:
            dentro += 1
    p = dentro / n
    ee = math.sqrt(p * (1 - p) / n)
    cubo = 2.0 ** d
    r = {"d": d, "dentro": dentro, "p": p, "vol": cubo * p,
         "lo": cubo * max(0.0, p - Z95 * ee), "hi": cubo * (p + Z95 * ee),
         "exacto": volumen_exacto(d)}
    # sin aciertos el IC normal se vuelve degenerado: se usa la regla de tres
    if dentro == 0:
        r["hi"] = cubo * 3.0 / n
    r["error"] = abs(r["vol"] - r["exacto"])
    return r


def rejilla(d, m):
    """Evalua el integrando en el centro de cada celda de una rejilla m^d."""
    centros = [-1.0 + (2 * i + 1) / m for i in range(m)]
    dentro = 0
    for punto in itertools.product(centros, repeat=d):
        if sum(x * x for x in punto) <= 1.0:
            dentro += 1
    total = m ** d
    vol = 2.0 ** d * dentro / total
    return {"d": d, "m": m, "puntos": total, "vol": vol,
            "error": abs(vol - volumen_exacto(d))}


# --- Figuras ---


def figura_fraccion():
    ds = list(range(1, 21))
    fracciones = [volumen_exacto(d) / 2.0 ** d for d in ds]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.semilogy(ds, fracciones, marker="o", ms=4, color="#4C72B0")
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel("$V_d / 2^d$")
    ax.set_title("Fraccion del cubo ocupada por la bola", fontsize=10)
    ax.set_xticks(range(0, 21, 2))
    fig.savefig(os.path.join(FIGS, "fraccion_bola.png"))
    plt.close(fig)
    return fracciones


def figura_errores(mc, rej):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ancho = 0.35
    xs = range(len(DIMENSIONES))
    ax.bar([x - ancho / 2 for x in xs], [r["error"] for r in mc], ancho,
           label="Monte Carlo", color="#4C72B0")
    ax.bar([x + ancho / 2 for x in xs], [r["error"] for r in rej], ancho,
           label="rejilla", color="#C44E52")
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"d={d}" for d in DIMENSIONES])
    ax.set_ylabel("error absoluto")
    ax.set_title(f"Error con el mismo presupuesto ({miles(PRESUPUESTO).replace('{,}', ',')} puntos)",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIGS, "error_mc_vs_rejilla.png"))
    plt.close(fig)


def escribir(mc, rej, fracciones):
    L = [r"\begin{table}[htbp]\centering",
         r"\caption{Volumen de la bola unitaria estimado con $N=\NDim$ puntos "
         r"uniformes en $[-1,1]^d$.}",
         r"\label{tab:volumenes}", r"\small",
         r"\begin{tabular}{rrrrll}", r"\toprule",
         r"$d$ & $V_d$ exacto & Aciertos & Estimacion & IC $95\%$ & Error \\",
         r"\midrule"]
    for r in mc:
        ic = (f"$[{r['lo']:.4f},\\ {r['hi']:.4f}]$" if r["dentro"] else
              f"$[0,\\ {r['hi']:.4f}]$")
        L.append(f"{r['d']} & {r['exacto']:.4f} & {miles(r['dentro'])} & "
                 f"{r['vol']:.4f} & {ic} & {r['error']:.4f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    escribir_tex("tabla_volumenes.tex", L)

    L = [r"\begin{table}[htbp]\centering",
         r"\caption{Monte Carlo contra cuadratura en rejilla con el mismo "
         r"presupuesto de puntos, y puntos que exigiria una rejilla de $m=10$ "
         r"subdivisiones por eje.}",
         r"\label{tab:rejilla}", r"\small",
         r"\begin{tabular}{rrrrrr}", r"\toprule",
         r"$d$ & $m$ posible & Puntos usados & Error rejilla & Error MC & "
         r"Puntos si $m=10$ \\", r"\midrule"]
    for r, q in zip(rej, mc):
        L.append(f"{r['d']} & {r['m']} & {miles(r['puntos'])} & "
                 f"{r['error']:.4f} & {q['error']:.4f} & $10^{{{r['d']}}}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    escribir_tex("tabla_rejilla.tex", L)

    d20 = next(r for r in mc if r["d"] == 20)
    L = [macro("NDim", miles(N)),
         macro("SemillaDim", SEMILLA),
         macro("Presupuesto", miles(PRESUPUESTO)),
         macro("FraccionDiez", cientifica(fracciones[9])),
         macro("FraccionVeinte", cientifica(fracciones[19])),
         macro("AciertosVeinte", d20["dentro"]),
         macro("CotaVeinte", f"{d20['hi']:.4f}"),
         macro("ExactoVeinte", f"{volumen_exacto(20):.4f}"),
         macro("MuestrasUnAcierto", miles(round(1 / fracciones[19])))]
    escribir_tex("datos_dimension.tex", L)


def main():
    print("Parte 2.3: volumen de la bola unitaria")
    gen = MersenneTwister(SEMILLA)
    mc = []
    for d in DIMENSIONES:
        r = monte_carlo(d, N, gen)
        mc.append(r)
        print(f"  d={d:2d}  exacto={r['exacto']:.4f}  estimado={r['vol']:.4f}  "
              f"aciertos={r['dentro']:,}  error={r['error']:.4f}")

    print("\nRejilla con el mismo presupuesto:")
    rej = []
    for d in DIMENSIONES:
        m = max(1, int(PRESUPUESTO ** (1.0 / d)))
        r = rejilla(d, m)
        rej.append(r)
        print(f"  d={d:2d}  m={m:3d}  puntos={r['puntos']:,}  "
              f"estimado={r['vol']:.4f}  error={r['error']:.4f}")

    fracciones = figura_fraccion()
    figura_errores(mc, rej)
    print(f"\n  V10/2^10 = {fracciones[9]:.2e}   V20/2^20 = {fracciones[19]:.2e}")
    print(f"  para un solo acierto en d=20 hacen falta ~{1/fracciones[19]:,.0f} puntos")

    escribir(mc, rej, fracciones)
    print("\nListo.")


if __name__ == "__main__":
    main()
