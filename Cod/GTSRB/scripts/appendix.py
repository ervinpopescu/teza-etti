#!/bin/python
import os
import pathlib
import sys
from pkgutil import iter_modules
from setuptools import find_packages

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
    paths.insert(0, "main.py")
    modules.insert(0, "main")
    string = "\\CountLinesInFile{{./Cod/GTSRB/{}}}\n\\lstinputlisting[caption={{{}}},linerange={{1-\\arabic{{FileLines}}}},label={},captionpos=t]{{./Cod/GTSRB/{}}}\n\\pagebreak\n"
    appendix_tex = os.path.join(current_file_path.parent.parent.parent, "appendix.tex")
    with open(appendix_tex, "w") as f:
        for module, path in zip(modules, paths):
            f.write(
                string.format(
                    path,
                    module.replace("modules.", "").replace("_", " ").capitalize()
                    + " module",
                    path.replace("modules/", ""),
                    path,
                )
            )

if __name__ == "__main__":
    main()
