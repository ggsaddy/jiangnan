# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['application.py'],
    pathex=['./'],
    binaries=[],
    datas=[
    ('element.py', '.'),
    ('utils.py', '.'),
    ('infoextraction2.py', '.'),
    ('plot_geo.py', '.'),
    ('config.py', '.'),
    ('classifier.py', '.'),
    ('draw_dxf.py', '.'),
    ('load.py', '.'),
    ],
    hiddenimports=['sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.tree._utils',
    'sklearn.tree._splitter',
    'sklearn.tree._criterion',
    'sklearn.tree._partitioner'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='application',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
