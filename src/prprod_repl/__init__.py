from importlib.metadata import version

__version__ = version("prprod_repl")


def temp() -> None:
    print("Hello from research-pr-production!")
