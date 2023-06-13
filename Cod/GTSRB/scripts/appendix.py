#!/bin/python
import os
import pathlib
from pprint import pprint
import sys
from setuptools import find_packages
from pkgutil import iter_modules

current_file_path = pathlib.Path(__file__).parent.resolve()


def find_modules(path):
    modules = list()
    paths = list()
    for pkg in find_packages(path):
        pkgpath = path + "/" + pkg.replace(".", "/")
        if sys.version_info.major == 2 or (
            sys.version_info.major == 3 and sys.version_info.minor < 6
        ):
            for _, name, ispkg in iter_modules([pkgpath]):
                if not ispkg:
                    modules.append(pkg + "." + name)
                    paths.append(pkg + "/" + name + ".py")
        else:
            for info in iter_modules([pkgpath]):
                if not info.ispkg:
                    modules.append(pkg + "." + info.name)
                    paths.append(pkg + "/" + info.name + ".py")

    return modules, paths


def main():
    modules, paths = find_modules(str(current_file_path.parent))
    
    string = """\\CountLinesInFile{{./Cod/GTSRB/{}}}
    \\lstinputlisting[caption={{{}}},linerange={{1-\\LineCount}},label={},captionpos=t]{{./Cod/GTSRB/{}}}
    \\pagebreak
    """
    for module, path in zip(modules, paths):
        print(string.format(path, module, path, path))


if __name__ == "__main__":
    main()
