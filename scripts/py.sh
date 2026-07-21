# source or prefix: keeps uv's caches and managed Pythons off the nearly-full C:
# usage:  UVBIN="$LOCALAPPDATA/Microsoft/WinGet/Links/uv.exe"
#         . scripts/py.sh && "$UVBIN" sync
export UV_CACHE_DIR="D:/Temp/uv-cache"
export UV_PYTHON_INSTALL_DIR="D:/uv-pythons"
