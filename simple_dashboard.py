#!/usr/bin/env python3
"""
Run the live GPU dashboard without installing the package.

Same as the ``gpu-dashboard`` command; the dashboard itself lives in
src/python/gpu_optimizer/dashboard/app.py. Pass --port to change the port.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "python"))

from gpu_optimizer.dashboard.app import main

if __name__ == "__main__":
    main()
