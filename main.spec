# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

datas_imbalanced = collect_data_files('imblearn')

datas = datas_imbalanced + [('minecLogo.ico', '.')]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='minec',                 
    debug=False,
    strip=False,
    upx=True,
    console=False,                
    icon='minecLogo.ico',         
    exclude_binaries=False,      
)
