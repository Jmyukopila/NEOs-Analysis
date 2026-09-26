"""Utilidades compartidas por la bateria de tests.

Dos cosas se testean en este repo:

1. `scripts/verificar_discrepancias.py`, que se importa por ruta porque
   `scripts/` no es un paquete.
2. Las funciones auxiliares definidas dentro de los notebooks, que se extraen
   del JSON del `.ipynb` con `ast` para poder ejecutarlas sin lanzar el
   pipeline completo (que descarga datos de las APIs de JPL).
"""

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
RUTA_SCRIPT = RAIZ / "scripts" / "verificar_discrepancias.py"
NB_DATOS = RAIZ / "data" / "ProyectoNeoRework_data.ipynb"
NB_ML = RAIZ / "notebooks" / "ProyectoNeoRework_ml.ipynb"

# Nombres de nivel de modulo que las funciones del notebook necesitan para
# ejecutarse (todo lo demas del notebook es codigo de pipeline con efectos).
ASIGNACIONES_NECESARIAS = {"sbdb_fields"}

# Nombres que ademas deben existir en el namespace extraido para que las funciones
# se ejecuten. Se declaran en vez de depender de que `ast` los encuentre por
# casualidad: `descargar_cad` captura `ErrorAPI` en su `except`, y si la clase no
# esta en el namespace el NameError aparece al primer fallo de red simulado, con un
# mensaje que no menciona la clase que falta.
CLASES_NECESARIAS = {"ErrorAPI"}

COLUMNAS_CAD = [
    "Object",
    "Close-Approach (CA) Date",
    "CA DistanceNominal (au)",
    "CA DistanceMinimum (au)",
    "V relative(km/s)",
    "V infinity(km/s)",
    "H(mag)",
    "Diameter(km)",
    "Std Diameter(km)",
    "MOID (au)",
    "H_SBDB(mag)",
    "first_obs_year",
    "data_arc(d)",
    "n_obs_used",
    "PHA_official",
    "post_discovery",
    "PHA_proxy",
]

# Las constantes de física se importan de `neos.constantes` en vez de replicarse. Una
# copia local puede divergir sin que nada lo detecte, y estas dos aparecen en las
# aserciones de los tests precisamente para comprobar que el script y el notebook
# coinciden: si el test usa su propia copia, comprueba que la copia coincide consigo
# misma y el test no vale para nada.
from neos.constantes import AU_KM, FACTOR_H_D, MU_TIERRA  # noqa: E402

UA_KM = AU_KM


def diametro_desde_h(h):
    """Replica la imputacion D = 1329 * 10^(-0.2H) / sqrt(0.14) del notebook."""
    return FACTOR_H_D * 10 ** (-0.2 * h)


def v_rel_desde_v_inf(v_inf, dist_au):
    """Replica la relacion fisica v_rel = sqrt(v_inf^2 + 2*mu/r) del script."""
    r = dist_au * UA_KM
    return (v_inf ** 2 + 2 * MU_TIERRA / r) ** 0.5


def evento(objeto, dist, h, *, dist_min=None, moid=None, h_sbdb=None, pha=1,
           post_discovery=1, v_inf=10.0, v_rel=None, diameter=None,
           fecha="2010-Jan-01 00:00"):
    """Construye una fila del catalogo CAD con valores fisicamente coherentes."""
    dist_min = dist if dist_min is None else dist_min
    h_sbdb = h if h_sbdb is None else h_sbdb
    v_rel = v_rel_desde_v_inf(v_inf, dist) if v_rel is None else v_rel
    diameter = diametro_desde_h(h) if diameter is None else diameter
    return {
        "Object": objeto,
        "Close-Approach (CA) Date": fecha,
        "CA DistanceNominal (au)": dist,
        "CA DistanceMinimum (au)": dist_min,
        "V relative(km/s)": v_rel,
        "V infinity(km/s)": v_inf,
        "H(mag)": h,
        "Diameter(km)": diameter,
        "Std Diameter(km)": diameter * 0.35,
        "MOID (au)": moid,
        "H_SBDB(mag)": h_sbdb,
        "first_obs_year": 2005.0,
        "data_arc(d)": 3000.0,
        "n_obs_used": 100,
        "PHA_official": pha,
        "post_discovery": post_discovery,
        "PHA_proxy": pha,
    }


def frame_cad(filas):
    """DataFrame con el esquema completo del CSV a partir de filas parciales."""
    return pd.DataFrame(filas, columns=COLUMNAS_CAD)


def frame_sano():
    """Catalogo minimo que satisface todas las comprobaciones corregibles.

    Mezcla objetos cercanos y lejanos, PHA y no PHA, y un objeto cuyo MOID no
    coincide con la regla de umbrales, para que ninguna de las igualdades que
    el script vigila (PHA == H<=22, proxy == H<=22) sature.
    """
    return frame_cad([
        evento("cercano-grande", 0.01, 18.0, moid=0.02, pha=1),
        evento("lejano-grande", 0.30, 19.0, moid=0.30, pha=0),
        evento("lejano-pequeno", 0.40, 25.0, moid=0.40, pha=0),
        # H <= 22 pero MOID > 0.05: rompe la equivalencia PHA == (H <= 22)
        evento("cercano-limite", 0.20, 21.0, moid=0.09, pha=0),
        # evento integrado hacia atras: debe quedar fuera de `obs`
        evento("cercano-grande", 0.002, 17.0, moid=0.02, pha=1, post_discovery=0),
    ])


@pytest.fixture
def vd(tmp_path):
    """Importa el script de verificacion aislado del disco y del estado global.

    Cada test recibe un modulo recien importado (la lista global `fallos` empieza
    vacia) y las rutas que usa, apuntando a `tmp_path`.

    Las rutas se pasan como argumentos a `cargar(ruta_cad=...)` en vez de monkeypatchear
    `RAIZ`/`RUTA_CAD` en el modulo. El script ya no define esos nombres: importan de
    `neos.constantes`, y las pruebas llegan a un estado en el que los cuatro tests que
    los tocaban fallaban con `AttributeError` porque `RAIZ` ya no existe ahí. Parchear
    un atributo que el módulo no declara sólo habría movido el fallo.
    """
    spec = importlib.util.spec_from_file_location(
        f"verificar_discrepancias_{id(tmp_path)}", RUTA_SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    modulo.fallos.clear()
    modulo.RUTA_CAD_TEST = tmp_path / "close_approaches.csv"
    modulo.RUTA_SBDB_TEST = tmp_path / "sbdb_neo.csv"
    yield modulo
    sys.modules.pop(spec.name, None)


@pytest.fixture
def escribir_cad(vd):
    """Escribe un DataFrame en la ruta que lee el script y devuelve la ruta."""
    def _escribir(df):
        df.to_csv(vd.RUTA_CAD_TEST, index=False)
        return vd.RUTA_CAD_TEST
    return _escribir


def _sbdb_en_ruta(ruta, filas):
    """Escribe un CSV de SBDB mínimo con las columnas que necesita el bloque D."""
    pd.DataFrame(filas, columns=["pdes", "pha", "moid", "H"]).to_csv(ruta, index=False)
    return ruta


@pytest.fixture
def escribir_sbdb(vd):
    """Escribe el catálogo de la SBDB que lee el bloque D."""
    def _escribir(df):
        df.to_csv(vd.RUTA_SBDB_TEST, index=False)
        return vd.RUTA_SBDB_TEST
    return _escribir


def codigo_notebook(ruta):
    """Codigo fuente concatenado de todas las celdas de codigo del notebook."""
    nb = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return "".join("".join(c["source"]) + "\n"
                   for c in nb["cells"] if c["cell_type"] == "code")


def funciones_notebook(ruta):
    """Ejecuta solo imports, constantes y `def`s del notebook y devuelve el namespace.

    Se descarta el resto del codigo: el notebook, ejecutado tal cual, descarga
    cientos de miles de eventos de las APIs de JPL.
    """
    arbol = ast.parse(codigo_notebook(ruta))
    conservadas = []
    for nodo in arbol.body:
        if isinstance(nodo, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            if isinstance(nodo, ast.ClassDef) and nodo.name not in CLASES_NECESARIAS:
                continue
            conservadas.append(nodo)
        elif isinstance(nodo, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ASIGNACIONES_NECESARIAS
                for t in nodo.targets):
            conservadas.append(nodo)
    ns = {"__name__": "notebook_extraido"}
    exec(compile(ast.Module(body=conservadas, type_ignores=[]), str(ruta), "exec"), ns)

    # Comprobacion de que el namespace tiene lo que las funciones necesitan. Sin
    # esto, un nombre que falta solo se descubre cuando una rama de `except` se
    # ejecuta, es decir, unicamente en los tests que simulan un fallo.
    faltan = sorted(CLASES_NECESARIAS - set(ns))
    assert not faltan, (
        f"{ruta.name}: el namespace extraido no define {faltan}. "
        "Anadelas a CLASES_NECESARIAS en conftest.py si son necesarias.")
    return ns


def fuente_funcion(ruta, nombre, sin_docstring=False):
    """Codigo fuente normalizado de una funcion del notebook (para comparar copias).

    `sin_docstring` permite comparar dos copias de la misma funcion cuyo
    docstring difiere pero cuyo comportamiento debe ser identico.
    """
    arbol = ast.parse(codigo_notebook(ruta))
    for nodo in arbol.body:
        if isinstance(nodo, ast.FunctionDef) and nodo.name == nombre:
            if sin_docstring and ast.get_docstring(nodo) is not None:
                nodo.body = nodo.body[1:]
            return ast.unparse(nodo)
    raise AssertionError(f"{nombre} no esta definida en {ruta}")


@pytest.fixture(scope="module")
def nb_datos():
    """Namespace con las funciones de data/ProyectoNeoRework_data.ipynb."""
    return funciones_notebook(NB_DATOS)
