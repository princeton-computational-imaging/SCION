"""``python -m scion.train`` shim → :func:`scion.cli.train_main`."""
from scion.cli import train_main

if __name__ == "__main__":
    train_main()
