"""Parte 2.5: el caso RANDU.

Estima dos volumenes en 3D con ternas consecutivas de RANDU y del Mersenne
Twister: el octante de la bola unitaria (pi/6) y una bola pequena de radio
0.1. El segundo caso es el que deja ver los planos de RANDU.

Uso:  python3 src/part2_randu.py
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prngs import RANDU, MersenneTwister
from montecarlo import escribir_tex, macro, miles, Z95, FIGS

SEMILLA = 141903
N = 2_000_000     # ternas de la estimacion principal
K = 100           # semillas distintas para medir la cobertura del IC
N_REP = 50_000    # ternas por repeticion

RADIO = 0.1
EXACTO_OCTANTE = math.pi / 6.0
EXACTO_BOLA = math.pi * RADIO ** 3 / 6.0

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})


# --- Estimacion usando ternas consecutivas ---


def indicadores(gen, n):
    """Para cada terna marca si cae en la bola unitaria y en la bola chica."""
    octante = np.empty(n, dtype=np.int8)
    bola = np.empty(n, dtype=np.int8)
    r2_chico = RADIO * RADIO
    for i in range(n):
        x, y, z = gen.random(), gen.random(), gen.random()
        r2 = x * x + y * y + z * z
        octante[i] = r2 <= 1.0
        bola[i] = r2 <= r2_chico
    return octante, bola


def resumen(dentro, exacto):
    n = len(dentro)
    p = float(dentro.mean())
    ee = math.sqrt(p * (1 - p) / n) if p > 0 else 1.0 / n
    return {"n": n, "media": p, "ee": ee, "lo": p - Z95 * ee, "hi": p + Z95 * ee,
            "error": abs(p - exacto), "sigmas": (p - exacto) / ee,
            "cubre": abs(p - exacto) <= Z95 * ee,
            "relativo": 100.0 * (p - exacto) / exacto}


def curva_error(dentro, enes, exacto):
    medias = np.cumsum(dentro) / np.arange(1, len(dentro) + 1)
    return np.abs(medias[enes - 1] - exacto)


def figura(enes, curvas, repeticiones):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.8))

    for ax, clave, titulo, exacto in (
            (a1, "octante", r"Bola unitaria ($\pi/6$)", EXACTO_OCTANTE),
            (a2, "bola", f"Bola de radio {RADIO}", EXACTO_BOLA)):
        ax.loglog(enes, curvas[("RANDU", clave)], marker="o", ms=3, lw=1,
                  color="#C44E52", label="RANDU")
        ax.loglog(enes, curvas[("MT", clave)], marker="o", ms=3, lw=1,
                  color="#4C72B0", label="Mersenne Twister")
        referencia = curvas[("MT", clave)][0] * (enes / enes[0]) ** -0.5
        ax.loglog(enes, referencia, "k--", lw=1, label="$N^{-1/2}$")
        ax.set_xlabel("N (ternas)")
        ax.set_ylabel("error absoluto")
        ax.set_title(titulo, fontsize=10)
        ax.legend(fontsize=7)

    for etiqueta, datos, color in (("RANDU", repeticiones["RANDU"], "#C44E52"),
                                   ("MT19937", repeticiones["MT"], "#4C72B0")):
        a3.hist([d["bola"]["media"] for d in datos], bins=18, alpha=0.6,
                color=color, label=etiqueta)
    a3.axvline(EXACTO_BOLA, color="black", ls="--", lw=1.2, label="valor exacto")
    a3.set_xlabel("estimacion")
    a3.set_ylabel("frecuencia")
    a3.set_title(f"Bola de radio {RADIO}: {K} semillas", fontsize=10)
    a3.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "randu_montecarlo.png"))
    plt.close(fig)


def escribir(resultados, cobertura):
    L = [r"\begin{table}[htbp]\centering",
         r"\caption{Dos estimaciones en tres dimensiones con $N=\NRandu$ "
         r"ternas consecutivas de cada generador. La ultima columna es la "
         r"fraccion de los $K=\KRandu$ intervalos que cubren el valor real.}",
         r"\label{tab:randu}", r"\small",
         r"\begin{tabular}{l|l|r|l|r|r|c}", r"\toprule",
         r"Region & Fuente & Estimacion & IC $95\%$ & Sesgo relativo & "
         r"$|z|$ & Cobertura \\", r"\midrule"]
    etiquetas = {"octante": r"Bola unitaria", "bola": f"Bola $r={RADIO}$"}
    for clave in ("octante", "bola"):
        for fuente in ("RANDU", "MT"):
            r = resultados[(fuente, clave)]
            nombre = "RANDU" if fuente == "RANDU" else "Mersenne Twister"
            L.append(f"{etiquetas[clave]} & {nombre} & ${r['media']:.6f}$ & "
                     f"$[{r['lo']:.6f},\\ {r['hi']:.6f}]$ & "
                     f"${r['relativo']:+.2f}\\%$ & ${abs(r['sigmas']):.1f}$ & "
                     f"{cobertura[(fuente, clave)]:.0f}\\% \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    escribir_tex("tabla_randu.tex", L)

    ro = resultados[("RANDU", "octante")]
    mo = resultados[("MT", "octante")]
    rb = resultados[("RANDU", "bola")]
    mb = resultados[("MT", "bola")]
    L = [macro("NRandu", miles(N)),
         macro("KRandu", K),
         macro("NRepRandu", miles(N_REP)),
         macro("RadioBola", RADIO),
         macro("ExactoOctante", f"{EXACTO_OCTANTE:.6f}"),
         macro("ExactoBola", f"{EXACTO_BOLA:.6f}"),
         macro("EstRanduOctante", f"{ro['media']:.6f}"),
         macro("ICRanduOctante", f"[{ro['lo']:.6f},\\ {ro['hi']:.6f}]"),
         macro("SigmasRanduOctante", f"{abs(ro['sigmas']):.1f}"),
         macro("EstMTOctante", f"{mo['media']:.6f}"),
         macro("SigmasMTOctante", f"{abs(mo['sigmas']):.1f}"),
         macro("EstRanduBola", f"{rb['media']:.6f}"),
         macro("ICRanduBola", f"[{rb['lo']:.6f},\\ {rb['hi']:.6f}]"),
         macro("SigmasRanduBola", f"{abs(rb['sigmas']):.1f}"),
         macro("RelativoRanduBola", f"{rb['relativo']:+.1f}\\%"),
         macro("EstMTBola", f"{mb['media']:.6f}"),
         macro("ICMTBola", f"[{mb['lo']:.6f},\\ {mb['hi']:.6f}]"),
         macro("SigmasMTBola", f"{abs(mb['sigmas']):.1f}"),
         macro("RelativoMTBola", f"{mb['relativo']:+.1f}\\%"),
         macro("CoberturaRanduBola", f"{cobertura[('RANDU', 'bola')]:.0f}\\%"),
         macro("CoberturaMTBola", f"{cobertura[('MT', 'bola')]:.0f}\\%"),
         macro("CoberturaRanduOctante",
               f"{cobertura[('RANDU', 'octante')]:.0f}\\%"),
         macro("CoberturaMTOctante", f"{cobertura[('MT', 'octante')]:.0f}\\%")]
    escribir_tex("datos_randu.tex", L)


def main():
    print("Parte 2.5: el caso RANDU")
    print(f"  bola unitaria: pi/6      = {EXACTO_OCTANTE:.8f}")
    print(f"  bola de radio {RADIO}: pi r^3/6 = {EXACTO_BOLA:.8f}\n")

    enes = np.unique(np.logspace(2, math.log10(N), 20).astype(int))
    resultados, curvas = {}, {}

    for fuente, gen in (("RANDU", RANDU(SEMILLA)),
                        ("MT", MersenneTwister(SEMILLA))):
        octante, bola = indicadores(gen, N)
        for clave, dentro, exacto in (("octante", octante, EXACTO_OCTANTE),
                                      ("bola", bola, EXACTO_BOLA)):
            r = resumen(dentro, exacto)
            resultados[(fuente, clave)] = r
            curvas[(fuente, clave)] = curva_error(dentro, enes, exacto)
            print(f"  {fuente:6s} {clave:8s} {r['media']:.6f} "
                  f"[{r['lo']:.6f}, {r['hi']:.6f}]  "
                  f"sesgo={r['relativo']:+.2f}%  |z|={abs(r['sigmas']):.1f}  "
                  f"{'cubre' if r['cubre'] else 'NO cubre'}")

    print(f"\n  Repitiendo con {K} semillas distintas...")
    repeticiones = {"RANDU": [], "MT": []}
    for k in range(K):
        for fuente, gen in (("RANDU", RANDU(SEMILLA + 2 * k)),
                            ("MT", MersenneTwister(SEMILLA + k))):
            octante, bola = indicadores(gen, N_REP)
            repeticiones[fuente].append(
                {"octante": resumen(octante, EXACTO_OCTANTE),
                 "bola": resumen(bola, EXACTO_BOLA)})

    cobertura = {(fuente, clave):
                 100.0 * sum(d[clave]["cubre"] for d in repeticiones[fuente]) / K
                 for fuente in ("RANDU", "MT") for clave in ("octante", "bola")}
    for clave in ("octante", "bola"):
        print(f"  cobertura del IC 95% ({clave}): "
              f"RANDU {cobertura[('RANDU', clave)]:.0f}%, "
              f"MT {cobertura[('MT', clave)]:.0f}%")

    figura(enes, curvas, repeticiones)
    escribir(resultados, cobertura)
    print("\nListo.")


if __name__ == "__main__":
    main()
