import pkgutil
from importlib import import_module
from pathlib import Path

import dsl


def test_import(pkg=dsl, ignore_deprecated: bool = True):
    stack = [(pkg.__name__, Path(pkg.__file__).parent.absolute())]

    while len(stack) > 0:
        pkgname, pkgpath = stack.pop()
        for m in pkgutil.iter_modules([str(pkgpath)]):
            mname = f"{pkgname}.{m.name}"
            if ignore_deprecated and mname.find("deprecated") != -1:
                continue
            if m.ispkg:
                stack.append((mname, pkgpath / m.name))
            import_module(mname)
