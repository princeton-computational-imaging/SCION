"""``python -m scion.eval`` shim → :func:`scion.cli.eval_main`."""
from scion.cli import eval_main

if __name__ == "__main__":
    eval_main()
