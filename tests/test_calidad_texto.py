"""Comprobación de que el código del proyecto no contiene caracteres corruptos.

Este test existe porque durante la refactorización aparecieron varias veces
caracteres CJK y cirílicos dentro de docstrings y comentarios en español. Son
invisibles en un diff normal y en la salida de un terminal que no los muestra, y
convierten un docstring en algo que no se puede leer ni revisar.

No es un problema cosmético: varios de esos caracteres estaban en el mensaje de
error de una función, es decir, en la ruta que se lee justo cuando algo ha salido
mal.

El test se aplica también a sí mismo, así que este docstring describe los
caracteres prohibidos sin usarlos: de lo contrario el fichero sería su propio
primer fallo.
"""

import pathlib
import re
import unicodedata

import pytest

# El español usa Greek para notación matemática y el propio símbolo β de la
# F-beta, así que el griego está en la lista de permitidos. Sin esta excepción el
# test marcaría `modelos.py`, que escribe `F-beta` y β, y `docs/03`, que usa β
# en las ecuaciones de métricas.
PERMITIDOS = set("→×°±σμβδλ≈≥≤·—–«»áéíóúñÁÉÍÓÚÑüÜçÇαεθ")

RANGOS_SOSPECHOSOS = [
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x2E80, 0x9FFF),   # CJK
    (0xAC00, 0xD7AF),   # Hangul silábico
    (0xF900, 0xFAFF),   # compatibilidad CJK
    (0xFF00, 0xFFEF),   # formas completas
    (0x0400, 0x04FF),   # cirílico
    (0x0590, 0x05FF),   # hebreo
    (0x0600, 0x06FF),   # árabe
]


def _es_sospechoso(char):
    if char in PERMITIDOS or char in " \n\t":
        return False
    if ord(char) < 0x80:
        return False
    if unicodedata.category(char).startswith("C"):  # control, formato, surrogate
        return False
    cp = ord(char)
    return any(ini <= cp <= fin for ini, fin in RANGOS_SOSPECHOSOS)


def _archivos_fuente():
    raiz = pathlib.Path(__file__).resolve().parent.parent
    return sorted(
        p for ext in ("*.py", "*.md")
        for p in raiz.rglob(ext)
        if ".venv" not in p.parts and "venv" not in p.parts
        and ".git" not in p.parts and "paper" not in p.parts
    )


#: El español usa letras griegas para notación matemática y el propio símbolo β de
#: la F-beta, así que el griego está permitido. Sin esta excepción el test marcaría
#: `modelos.py` y `docs/03-conceptos-ml.md`, que escriben β en las ecuaciones.
PERMITIDOS = set("→×°±σμβδλ≈≥≤·—–«»áéíóúñÁÉÍÓÚÑüÜçÇαεθ")

RANGOS_SOSPECHOSOS = [
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x2E80, 0x9FFF),   # CJK
    (0xAC00, 0xD7AF),   # Hangul silábico
    (0xF900, 0xFAFF),   # compatibilidad CJK
    (0xFF00, 0xFFEF),   # formas completas
    (0x0400, 0x04FF),   # cirílico
    (0x0590, 0x05FF),   # hebreo
    (0x0600, 0x06FF),   # árabe
]


def _es_sospechoso(char):
    if char in PERMITIDOS or char in " \n\t":
        return False
    if ord(char) < 0x80:
        return False
    if unicodedata.category(char).startswith("C"):  # control, formato, surrogate
        return False
    cp = ord(char)
    return any(ini <= cp <= fin for ini, fin in RANGOS_SOSPECHOSOS)


@pytest.mark.parametrize("ruta", _archivos_fuente(), ids=lambda p: p.name)
def test_sin_caracteres_de_otro_alfabeto(ruta):
    """Ningún carácter CJK, cirílico, hebreo o árabe en el código del repo."""
    texto = ruta.read_text(encoding="utf-8", errors="replace")
    malos = {}
    for num_linea, linea in enumerate(texto.splitlines(), 1):
        sospechosos = [c for c in linea if _es_sospechoso(c)]
        if sospechosos:
            malos[num_linea] = [(c, unicodedata.name(c, "?")) for c in sospechosos]
    assert not malos, (
        f"{ruta.name} contiene caracteres de otro alfabeto, normalmente corruptos: "
        + "; ".join(f"linea {n}: {cs}" for n, cs in malos.items()))


#: Sufijos pegados a su raíz, con la forma buena. La lista es cerrada a propósito:
#: un patron abierto ("la" + 3 letras) marcaba "largo", "lanza", "lambda" y "lanzar".
PALABRAS_PEGADAS = [
    ("de", "dis", "dediscordancia", "de discordancia"),
    ("la", "average", "laaverage", "la average precision"),
    ("con", "stante", "constante de", "constante de"),
    ("sin", "complej", "sincomplejidad", "sin complejidad"),
    ("correspond", "ed", "corresponded", "corresponder"),
    ("geometr", "ia", "geometria", "geometria"),
    ("clasificac", "ion", "clasificacion", "clasificacion"),
    ("predicci", "on", "prediccion", "prediccion"),
    ("selecci", "on", "seleccion", "seleccion"),
    ("correlaci", "on", "correlacion", "correlacion"),
    ("definici", "on", "definicion", "definicion"),
    ("informaci", "on", "informacion", "informacion"),
    ("conclusi", "on", "conclusion", "conclusion"),
    ("distribuci", "on", "distribucion", "distribucion"),
    ("precisi", "on", "precision", "precision"),
    ("imputaci", "on", "imputacion", "imputacion"),
]


@pytest.mark.parametrize("ruta", _archivos_fuente(), ids=lambda p: p.name)
def test_palabras_pegado(ruta):
    """Detecta sufijos de palabras latinas pegados al resto, por sustitucion fallida.

    "dediscordancia" y "corresponded" son un sufijo pegado a su raiz. Cada par de la
    lista se busca pegado y se comprueba que la forma buena exista en el fichero: si
    no aparece nunca, la pegada es un texto corrupto y no una palabra suelta.
    """
    texto = ruta.read_text(encoding="utf-8", errors="replace").lower()
    for _raiz, _sufijo, pegada, buena in PALABRAS_PEGADAS:
        n_pegada = texto.count(pegada)
        if not n_pegada:
            continue
        if texto.count(buena) == 0:
            raise AssertionError(
                f"{ruta.name}: {pegada!r} aparece {n_pegada} vez/veces y la forma "
                f"correcta {buena!r} no aparece nunca. Suele ser texto corrupto o una "
                f"palabra escrita sin tilde.")

