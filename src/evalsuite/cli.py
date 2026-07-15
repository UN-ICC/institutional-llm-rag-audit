"""CLI entrypoint for the evalsuite evaluation suite.

Usage:
  python -m evalsuite --layer 1a --phase run
  python -m evalsuite --layer 1a --phase eval --pattern-only
"""

from evalsuite.runners.run_layer import main

if __name__ == "__main__":
    main()
