"""Comprueba que todos los scripts del repositorio parsean.

Un `SyntaxError` en un script no se ve hasta que alguien lo ejecuta, y en este repo
los scripts se ejecutan a mano y con datos reales. El fichero `pipeline_moid_pha.py`
llego a tener un caracter CJK colado en un docstring y otro nombre de etiqueta mal
mapeado: ninguno de los dos impedia que el fichero importara, y los dos rompian mas
tarde, en un punto que no los relacionaba con su causa.

Se comprueba tambien que el modulo se puede *importar* sin ejecutarlo, que es lo que
detecta un `NameError` en un import de nivel superior o una clase base que no existe.
Importar no debe tener efectos: por eso se separa de `main()` en todos los scripts.

Salida: 0 si todo parsea, 1 con la lista de fallos. Pensado para CI.
"""

import ast
import importlib.util
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def comprobar_sintaxis():
    """Devuelve la lista de ficheros con un `SyntaxError`."""
    fallos = []
    for ruta in sorted((RAIZ / "scripts").glob("*.py")):
        try:
            ast.parse(ruta.read_text(encoding="utf-8"), filename=str(ruta))
        except SyntaxError as e:
            fallos.append(f"{ruta.name}:{e.lineno}: {e.msg}")
    return fallos


def comprobar_importables():
    """Devuelve la lista de scripts que no se pueden importar sin ejecutarlos."""
    fallos = []
    for ruta in sorted((RAIZ / "scripts").glob("*.py")):
        if ruta.name.startswith("_") or ruta.name == pathlib.Path(__file__).name:
            continue
        spec = importlib.util.spec_from_file_location(f"ci_{ruta.stem}", ruta)
        modulo = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(modulo)
        except Exception as e:  # noqa: BLE001: aqui el motivo es el error
            fallos.append(f"{ruta.name}: {type(e).__name__}: {e}")
        finally:
            sys.modules.pop(spec.name, None)
    return fallos


def main():
    fallos = comprobar_sintaxis()
    if fallos:
        print("Errores de sintaxis:")
        print("\n".join(f"  {f}" for f in fallos))
        return 1

    n = len(list((RAIZ / "scripts").glob("*.py")))
    print(f"{n} scripts parsean")

    fallos = comprobar_importables()
    if fallos:
        print("Scripts que no importan:")
        print("\n".join(f"  {f}" for f in fallos))
        return 1
    print("todos los scripts importan sin ejecutarse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
