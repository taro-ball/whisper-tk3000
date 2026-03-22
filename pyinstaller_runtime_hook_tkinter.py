from __future__ import annotations

import os
import sys
from pathlib import Path


bundle_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
tcl_dir = bundle_dir / "_tcl_data"
tk_dir = bundle_dir / "_tk_data"

if not tcl_dir.is_dir():
    raise FileNotFoundError(f'Tcl data directory "{tcl_dir}" not found.')
if not tk_dir.is_dir():
    raise FileNotFoundError(f'Tk data directory "{tk_dir}" not found.')

os.environ["TCL_LIBRARY"] = str(tcl_dir)
os.environ["TK_LIBRARY"] = str(tk_dir)
