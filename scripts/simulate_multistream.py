#!/usr/bin/env python
"""Entry point: compositional dataset generation. Thin wrapper over
hydrabflow.pipeline.simulate_multistream."""

from hydrabflow.pipeline.simulate_multistream import cli

if __name__ == "__main__":
    cli()
