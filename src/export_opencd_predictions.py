#!/usr/bin/env python3
"""Compatibility entry point for OpenPCDet inference export.

The implementation remains isolated while its optional Torch/OpenPCDet runtime
dependencies are deliberately kept out of the lightweight shared package.
"""

from lumpi_repro.legacy.export_opencd_predictions import main


if __name__ == "__main__":
    main()
