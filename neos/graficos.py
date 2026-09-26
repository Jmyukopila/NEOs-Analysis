"""Helpers de figuras: guardado uniforme y barras agrupadas."""

import os

import matplotlib.pyplot as plt

from neos.constantes import DIR_FIGURAS

DPI_FIGURAS = 110


def guardar_figura(nombre, dpi=DPI_FIGURAS, bbox_inches=None, mostrar=True):
    """Guarda la figura actual en results/figures/ y la muestra.

    El nombre es relativo a la carpeta de figuras, así que la ruta no depende del
    directorio de trabajo desde el que se ejecute el notebook o el script.
    """
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURAS, nombre), dpi=dpi, bbox_inches=bbox_inches)
    if mostrar:
        plt.show()


def barras_agrupadas(categorias, series, ancho_total=0.8, errores=None, colores=None,
                     etiquetar=False, formato="{:.1f}", capsize=3):
    """Dibuja un grupo de barras por categoría, una barra por serie.

    `series` es {nombre: valores} y `errores` opcionalmente {nombre: desviaciones}.
    Deja las etiquetas del eje X centradas bajo cada grupo.
    """
    ancho = ancho_total / len(series)
    for j, (nombre, valores) in enumerate(series.items()):
        posiciones = [i + j * ancho for i in range(len(categorias))]
        plt.bar(posiciones, valores, ancho,
                yerr=None if errores is None else errores[nombre],
                capsize=capsize, label=nombre,
                color=None if colores is None else colores[j])
        if etiquetar:
            for x, v in zip(posiciones, valores):
                plt.text(x, v, formato.format(v), ha="center", va="bottom", fontsize=8)

    centro = ancho_total / 2 - ancho / 2
    plt.xticks([i + centro for i in range(len(categorias))], categorias)
