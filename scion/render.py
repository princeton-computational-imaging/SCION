"""``python -m scion.render`` shim → :func:`scion.cli.render_main`."""
from scion.cli import render_main

if __name__ == "__main__":
    render_main()
