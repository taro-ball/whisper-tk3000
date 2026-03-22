# PyInstaller spec for a Windows onedir build.

import sys
from pathlib import Path

from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import collect_data_files


python_base = Path(sys.base_prefix)
python_dll_dir = python_base / "DLLs"
python_lib_dir = python_base / "Lib"
python_tcl_dir = python_base / "tcl"

datas = collect_data_files("customtkinter")
hiddenimports = [
    "tkinter",
    "tkinter.ttk",
    "tkinter.font",
    "tkinter.filedialog",
    "tkinter.constants",
]
binaries = [
    (str(python_dll_dir / "_tkinter.pyd"), "."),
    (str(python_dll_dir / "tcl86t.dll"), "."),
    (str(python_dll_dir / "tk86t.dll"), "."),
]

block_cipher = None


a = Analysis(
    ["whisper_transcriber.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["pyinstaller_hooks"],
    hooksconfig={},
    runtime_hooks=["pyinstaller_runtime_hook_tkinter.py"],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="whisper_transcriber",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    Tree("bin", prefix="bin"),
    Tree(str(python_lib_dir / "tkinter"), prefix="tkinter"),
    Tree(str(python_tcl_dir / "tcl8.6"), prefix="_tcl_data"),
    Tree(str(python_tcl_dir / "tk8.6"), prefix="_tk_data"),
    Tree(str(python_tcl_dir / "tcl8"), prefix="tcl8"),
    strip=False,
    upx=True,
    upx_exclude=[],
    name="whisper_transcriber",
)
