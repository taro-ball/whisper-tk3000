# PyInstaller spec for a Windows onedir build.

from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import collect_data_files


datas = collect_data_files("customtkinter")

block_cipher = None


a = Analysis(
    ["whisper_tk3000/__main__.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
    strip=False,
    upx=True,
    upx_exclude=[],
    name="whisper_transcriber",
)
