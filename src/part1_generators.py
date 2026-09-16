"""Parte 1: analisis de los cinco generadores.

Genera los histogramas, las pruebas de hipotesis, la prueba espectral y la
comparacion con random y secrets. Guarda las figuras en figs/ y los
fragmentos de tablas en doc/.

Uso:  python3 src/part1_generators.py
"""

import math
import os
import random
import secrets
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prngs import (LCG, MiddleSquare, MersenneTwister, BlumBlumShub, RANDU,
                   detectar_periodo)
import stats_tests as st

# Semillas documentadas
SEMILLA = 141903
SEMILLA_BBS = 14773

N = 100_000        # valores por generador
N_TERNAS = 3_000   # ternas para la prueba espectral
CELDAS = 20        # celdas del histograma
CELDAS_CHI = 10    # celdas de la prueba chi-cuadrado

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
FIGS = os.path.join(RAIZ, "figs")
DOC = os.path.join(RAIZ, "doc")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})

AZUL, ROJO, VERDE, MORADO = "#4C72B0", "#C44E52", "#55A868", "#8172B2"

# nombre interno -> (constructor, semilla, etiqueta, sufijo para las macros)
GENERADORES = [
    ("LCG", lambda: LCG(SEMILLA), SEMILLA, "LCG", "LCG"),
    ("MiddleSquare", lambda: MiddleSquare(SEMILLA), SEMILLA,
     "Cuadrados medios", "CM"),
    ("MersenneTwister", lambda: MersenneTwister(SEMILLA), SEMILLA,
     "Mersenne Twister", "MT"),
    ("BlumBlumShub", lambda: BlumBlumShub(SEMILLA_BBS), SEMILLA_BBS,
     "Blum Blum Shub", "BBS"),
    ("RANDU", lambda: RANDU(SEMILLA), SEMILLA, "RANDU", "RANDU"),
]


# --- Figuras: histograma, cubo de ternas y planos de RANDU ---


def histograma(nombre, u, etiqueta, semilla):
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.hist(u, bins=CELDAS, range=(0, 1), color=AZUL, edgecolor="white", lw=0.5)
    ax.axhline(len(u) / CELDAS, color=ROJO, ls="--", lw=1.2,
               label="frecuencia esperada")
    ax.set_title(f"{etiqueta}  (semilla {semilla})", fontsize=10)
    ax.set_xlabel("$u_i$")
    ax.set_ylabel("frecuencia")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIGS, f"hist_{nombre}.png"))
    plt.close(fig)


def cubo_ternas(nombre, u, etiqueta, semilla):
    """Ternas consecutivas dentro del cubo unitario."""
    x, y, z = u[:N_TERNAS], u[1:N_TERNAS + 1], u[2:N_TERNAS + 2]
    fig = plt.figure(figsize=(4.6, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=1.5, alpha=0.5, color=AZUL)
    ax.set_title(f"{etiqueta}\n{N_TERNAS} ternas, semilla {semilla}", fontsize=9)
    ax.set_xlabel("$u_i$")
    ax.set_ylabel("$u_{i+1}$")
    ax.set_zlabel("$u_{i+2}$")
    for poner in (ax.set_xticks, ax.set_yticks, ax.set_zticks):
        poner([0, 0.5, 1])
    fig.savefig(os.path.join(FIGS, f"cubo_{nombre}.png"))
    plt.close(fig)


def planos_randu(u):
    """RANDU visto de canto y verificacion de 9u_i - 6u_{i+1} + u_{i+2} entero."""
    n = 20_000
    x = np.array(u[:n])
    y = np.array(u[1:n + 1])
    z = np.array(u[2:n + 2])

    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(x, y, z, s=0.6, alpha=0.45, color=ROJO)
    ax.view_init(elev=0, azim=np.degrees(np.arctan2(9.0, 6.0)))
    ax.set_title("Las ternas vistas desde la direccion del plano", fontsize=9)
    ax.set_xlabel("$u_i$")
    ax.set_ylabel("$u_{i+1}$")
    ax.set_zlabel("$u_{i+2}$")
    for poner in (ax.set_xticks, ax.set_yticks, ax.set_zticks):
        poner([0, 0.5, 1])

    combinacion = 9 * x - 6 * y + z
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(combinacion, bins=300, color=ROJO)
    ax2.set_title(r"$9u_i - 6u_{i+1} + u_{i+2}$ solo toma valores enteros",
                  fontsize=9)
    ax2.set_xlabel("valor de la combinacion")
    ax2.set_ylabel("frecuencia")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "randu_planos.png"))
    plt.close(fig)

    planos = len(np.unique(np.round(combinacion).astype(int)))
    residuo = float(np.max(np.abs(combinacion - np.round(combinacion))))
    return planos, residuo


def colapso_cuadrados_medios(u):
    """Trayectoria del generador y largo del tramo util para varias semillas."""
    g = MiddleSquare(SEMILLA)
    trayectoria, vistos, colapso = [], {}, None
    for i in range(200_000):
        if g.state in vistos:
            colapso = vistos[g.state]
            break
        vistos[g.state] = i
        trayectoria.append(g.state / g.m)
        g.next_int()

    util = trayectoria[:colapso]
    pruebas_util = st.todas(util, k=5) if len(util) > 50 else None

    largos = []
    for s in range(SEMILLA, SEMILLA + 200):
        gg = MiddleSquare(s)
        estados, pasos = set(), 0
        while gg.state not in estados and pasos < 20_000:
            estados.add(gg.state)
            gg.next_int()
            pasos += 1
        largos.append(pasos)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    a1.plot(trayectoria, lw=0.7, color=MORADO)
    if colapso:
        a1.axvline(colapso, color=ROJO, ls="--", lw=1.2,
                   label=f"colapso en i={colapso}")
        a1.legend(fontsize=8)
    a1.set_title(f"Trayectoria con semilla {SEMILLA}", fontsize=9)
    a1.set_xlabel("i")
    a1.set_ylabel("$u_i$")

    a2.hist(largos, bins=30, color=MORADO, edgecolor="white", lw=0.5)
    a2.set_title("Tramo util para 200 semillas consecutivas", fontsize=9)
    a2.set_xlabel("pasos antes de repetir un estado")
    a2.set_ylabel("frecuencia")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ms_colapso.png"))
    plt.close(fig)

    return {"colapso": colapso, "n_util": len(util), "pruebas_util": pruebas_util,
            "largo_min": min(largos), "largo_max": max(largos),
            "largo_medio": sum(largos) / len(largos),
            "frac_ceros": sum(1 for x in u if x == 0.0) / len(u)}


# --- Comparacion contra la libreria estandar y medicion de velocidad ---


def comparar_librerias():
    """MT19937 propio contra random y secrets."""
    g = MersenneTwister.como_cpython(SEMILLA)
    t0 = time.perf_counter()
    propio = [g.random53() for _ in range(N)]
    t_propio = time.perf_counter() - t0

    random.seed(SEMILLA)
    t0 = time.perf_counter()
    estandar = [random.random() for _ in range(N)]
    t_estandar = time.perf_counter() - t0

    t0 = time.perf_counter()
    seguro = [secrets.randbits(53) / float(1 << 53) for _ in range(N)]
    t_seguro = time.perf_counter() - t0

    fuentes = [("MT19937 propio", propio, t_propio),
               ("random", estandar, t_estandar),
               ("secrets", seguro, t_seguro)]

    fig, ejes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (etiqueta, u, _) in zip(ejes, fuentes):
        ax.hist(u, bins=CELDAS, range=(0, 1), color=VERDE, edgecolor="white", lw=0.5)
        ax.axhline(N / CELDAS, color=ROJO, ls="--", lw=1.2)
        ax.set_title(etiqueta, fontsize=9)
        ax.set_xlabel("$u_i$")
    ejes[0].set_ylabel("frecuencia")
    fig.suptitle(f"{N:,} valores de cada fuente (semilla {SEMILLA})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "comparacion_librerias.png"))
    plt.close(fig)

    filas = []
    for etiqueta, u, t in fuentes:
        r = st.todas(u, k=CELDAS_CHI)
        filas.append({"fuente": etiqueta, "vps": N / t, "pruebas": r})
    return filas, propio == estandar


def medir_velocidad():
    filas = {}
    for nombre, crear, _, _, _ in GENERADORES:
        g = crear()
        t0 = time.perf_counter()
        g.sample(50_000)
        filas[nombre] = 50_000 / (time.perf_counter() - t0)
    return filas


# --------------------------------------------------------------------------
# Salida en LaTeX
# --------------------------------------------------------------------------
CABECERA = ("% Generado por src/part1_generators.py.\n"
            "% No editar a mano: se sobrescribe en cada ejecucion.\n")


def fmt_p(p):
    if p < 1e-12:
        return r"$<10^{-12}$"
    if p < 1e-3:
        mantisa, exp = f"{p:.2e}".split("e")
        return rf"${mantisa}\times 10^{{{int(exp)}}}$"
    return f"{p:.4f}"


def veredicto(p):
    return "no rechaza" if p >= 0.05 else "rechaza"


def cientifica(x):
    """Notacion cientifica lista para LaTeX."""
    if x == 0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(x))))
    return rf"${x / 10 ** exp:.1f}\times 10^{{{exp}}}$"


def escribir(nombre, lineas):
    ruta = os.path.join(DOC, nombre)
    with open(ruta, "w") as f:
        f.write(CABECERA + "\n".join(lineas) + "\n")
    return ruta


def tabla_pruebas(nombre, sufijo, etiqueta, r):
    filas = [
        (r"$\chi^2$ ($k=10$)", f"{r['chi2']['stat']:.2f}", r["chi2"]["p"],
         "uniformidad"),
        ("Kolmogorov--Smirnov", f"{r['ks']['stat']:.5f}", r["ks"]["p"],
         "uniformidad"),
        ("Autocorrelacion lag-1", f"{r['lag1']['stat']:+.5f}", r["lag1"]["p"],
         "independencia"),
        ("Rachas", f"{r['rachas']['stat']:,}".replace(",", "{,}"),
         r["rachas"]["p"], "independencia"),
    ]
    L = [r"\begin{table}[htbp]\centering",
         rf"\caption{{Pruebas de hipotesis para {etiqueta} sobre $N=\NMuestra$ "
         r"valores. Se rechaza $H_0$ al $5\%$ si $p<0.05$.}",
         rf"\label{{tab:pruebas{sufijo}}}",
         r"\small",
         r"\begin{tabular}{l|l|r|r|l}", r"\toprule",
         r"Prueba & Mide & Estadistico & $p$-valor & Decision \\", r"\midrule"]
    for prueba, stat, p, mide in filas:
        L.append(f"{prueba} & {mide} & {stat} & {fmt_p(p)} & "
                 f"{veredicto(p)} $H_0$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return escribir(f"pruebas_{sufijo}.tex", L)


def tabla_velocidad(vel):
    L = [r"\begin{table}[htbp]\centering",
         r"\caption{Velocidad medida en Python puro, 50\,000 valores por "
         r"generador.}",
         r"\label{tab:velocidad}",
         r"\begin{tabular}{l|r|r}", r"\toprule",
         r"Generador & Valores por segundo & Costo relativo \\", r"\midrule"]
    mejor = max(vel.values())
    for nombre, _, _, etiqueta, _ in GENERADORES:
        v = vel[nombre]
        L.append(f"{etiqueta} & {v:,.0f} & {mejor/v:.1f}x \\\\".replace(",", "{,}"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return escribir("tabla_velocidad.tex", L)


def tabla_librerias(filas):
    L = [r"\begin{table}[htbp]\centering",
         r"\caption{MT19937 propio frente a \texttt{random} y \texttt{secrets} "
         r"($N=\NMuestra$).}",
         r"\label{tab:librerias}", r"\small",
         r"\begin{tabular}{l|r|r|r|r|r}", r"\toprule",
         r"Fuente & Valores/s & $X^2$ & $p_{\chi^2}$ & $D$ & $p_{KS}$ \\",
         r"\midrule"]
    for f in filas:
        r = f["pruebas"]
        L.append(f"\\texttt{{{f['fuente']}}} & {f['vps']:,.0f} & "
                 f"{r['chi2']['stat']:.2f} & {fmt_p(r['chi2']['p'])} & "
                 f"{r['ks']['stat']:.5f} & {fmt_p(r['ks']['p'])} \\\\"
                 .replace(",", "{,}"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return escribir("tabla_librerias.tex", L)


def escribir_datos(resultados, vel, cm, planos, residuo, identicas, periodo_cm):
    def macro(nombre, valor):
        return f"\\newcommand{{\\{nombre}}}{{{valor}}}"

    L = [macro("NMuestra", f"{N:,}".replace(",", "{,}")),
         macro("NTernas", f"{N_TERNAS:,}".replace(",", "{,}")),
         macro("Semilla", SEMILLA),
         macro("SemillaBBS", SEMILLA_BBS),
         macro("PlanosRandu", planos),
         macro("ResiduoRandu", cientifica(residuo))]

    for nombre, _, _, _, sufijo in GENERADORES:
        r = resultados[nombre]
        L += [macro(f"statChi{sufijo}", f"{r['chi2']['stat']:.2f}"),
              macro(f"pChi{sufijo}", f"{r['chi2']['p']:.4f}"),
              macro(f"statKS{sufijo}", f"{r['ks']['stat']:.5f}"),
              macro(f"pKS{sufijo}", f"{r['ks']['p']:.4f}"),
              macro(f"statLag{sufijo}", f"{r['lag1']['stat']:+.5f}"),
              macro(f"pLag{sufijo}", f"{r['lag1']['p']:.4f}"),
              macro(f"statRachas{sufijo}", f"{r['rachas']['stat']:,}"
                    .replace(",", "{,}")),
              macro(f"pRachas{sufijo}", f"{r['rachas']['p']:.4f}"),
              macro(f"vel{sufijo}", f"{vel[nombre]:,.0f}".replace(",", "{,}"))]

    L += [macro("CMColapso", f"{cm['colapso']:,}".replace(",", "{,}")),
          macro("CMFracCeros", f"{100*cm['frac_ceros']:.1f}\\%"),
          macro("CMLargoMin", cm["largo_min"]),
          macro("CMLargoMax", f"{cm['largo_max']:,}".replace(",", "{,}")),
          macro("CMLargoMedio", f"{cm['largo_medio']:,.0f}".replace(",", "{,}")),
          macro("CMCiclo", periodo_cm[1])]
    if cm["pruebas_util"]:
        pu = cm["pruebas_util"]
        L += [macro("CMUtilN", f"{cm['n_util']:,}".replace(",", "{,}")),
              macro("CMUtilChi", f"{pu['chi2']['stat']:.2f}"),
              macro("CMUtilChiP", f"{pu['chi2']['p']:.3f}"),
              macro("CMUtilKS", f"{pu['ks']['stat']:.4f}"),
              macro("CMUtilKSP", f"{pu['ks']['p']:.3f}")]
    L.append(macro("MTIdentico", "si" if identicas else "no"))
    return escribir("datos_parte1.tex", L)


def main():
    print(f"Parte 1 | semilla {SEMILLA} | N = {N:,}")

    muestras, resultados = {}, {}
    for nombre, crear, semilla, etiqueta, sufijo in GENERADORES:
        g = crear()
        u = g.sample(N)
        muestras[nombre] = u
        resultados[nombre] = st.todas(u, k=CELDAS_CHI)
        histograma(nombre, u, etiqueta, semilla)
        r = resultados[nombre]
        print(f"\n  {etiqueta} (semilla {semilla})")
        print(f"    chi2   {r['chi2']['stat']:12.2f}  p={r['chi2']['p']:.3e}")
        print(f"    KS     {r['ks']['stat']:12.5f}  p={r['ks']['p']:.3e}")
        print(f"    lag-1  {r['lag1']['stat']:+12.5f}  p={r['lag1']['p']:.3e}")
        print(f"    rachas {r['rachas']['stat']:12}  p={r['rachas']['p']:.3e}")

    print("\nPrueba espectral...")
    for nombre, _, semilla, etiqueta, _ in GENERADORES:
        cubo_ternas(nombre, muestras[nombre], etiqueta, semilla)
    planos, residuo = planos_randu(muestras["RANDU"])
    print(f"  RANDU: {planos} planos, residuo maximo {residuo:.1e}")

    print("\nCuadrados medios...")
    periodo_cm = detectar_periodo(MiddleSquare(SEMILLA), 200_000)
    cm = colapso_cuadrados_medios(muestras["MiddleSquare"])
    print(f"  colapso en i={cm['colapso']}, ciclo de largo {periodo_cm[1]}")
    print(f"  ceros en la muestra: {100*cm['frac_ceros']:.1f}%")
    print(f"  tramo util: min={cm['largo_min']}, medio={cm['largo_medio']:.0f}, "
          f"max={cm['largo_max']}")

    print("\nComparacion con random y secrets...")
    filas, identicas = comparar_librerias()
    for f in filas:
        print(f"  {f['fuente']:16s} {f['vps']:12,.0f} val/s")
    print(f"  la implementacion propia da los mismos numeros que random: {identicas}")

    print("\nVelocidad...")
    vel = medir_velocidad()
    for nombre, v in vel.items():
        print(f"  {nombre:16s} {v:12,.0f} val/s")

    archivos = [escribir_datos(resultados, vel, cm, planos, residuo, identicas,
                               periodo_cm),
                tabla_velocidad(vel), tabla_librerias(filas)]
    for nombre, _, _, etiqueta, sufijo in GENERADORES:
        archivos.append(tabla_pruebas(nombre, sufijo, etiqueta,
                                      resultados[nombre]))

    print("\nArchivos escritos:")
    for a in archivos:
        print(f"  {os.path.relpath(a, RAIZ)}")
    print(f"Figuras en {os.path.relpath(FIGS, RAIZ)}/")


if __name__ == "__main__":
    main()
