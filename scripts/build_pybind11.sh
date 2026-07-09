#!/bin/bash
# ============================================================================
# build_pybind11.sh — Compile the hydro._core pybind11 extension module
#
# Usage:
#   ./scripts/build_pybind11.sh              # build
#   ./scripts/build_pybind11.sh clean        # remove build artifacts
#   ./scripts/build_pybind11.sh install      # build + install to hydro/ dir
#
# Requirements:
#   pybind11  (pip install pybind11)
#   netcdf C library
#   g++ with C++17 support
# ============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_pybind11"
HYDRO_DIR="${ROOT_DIR}/hydro"
SRC_DIR="${ROOT_DIR}/src"
INC_DIR="${ROOT_DIR}/include"
BINDINGS="${ROOT_DIR}/python/bindings.cpp"

# --- colours ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# --- locate Python ---
PYTHON="${PYTHON:-python3}"
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
PY_INCLUDE=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_SUFFIX=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

info "Python:   $PYTHON (${PY_VER})"
info "Includes: ${PY_INCLUDE}"

# --- locate pybind11 ---
PYBIND11_INCLUDE=$("$PYTHON" -c "import pybind11; print(pybind11.get_include())" 2>/dev/null) || {
    err "pybind11 not found.  Install with:  pip install pybind11"
    exit 1
}
info "pybind11: ${PYBIND11_INCLUDE}"

# --- locate NumPy ---
NUMPY_INCLUDE=$("$PYTHON" -c "import numpy; print(numpy.get_include())" 2>/dev/null) || {
    warn "NumPy not found (required at runtime).  Install with:  pip install numpy"
}
if [ -n "${NUMPY_INCLUDE:-}" ]; then
    info "NumPy:    ${NUMPY_INCLUDE}"
fi

# --- locate NetCDF ---
NETCDF_CFLAGS=$(nc-config --cflags 2>/dev/null || true)
NETCDF_LIBS=$(nc-config --libs 2>/dev/null || true)
if [ -z "$NETCDF_CFLAGS" ]; then
    err "nc-config not found — NetCDF C library required.  Install with:  apt install libnetcdf-dev"
    exit 1
fi
info "NetCDF:   $(echo $NETCDF_LIBS | awk '{print $1}')..."

# --- OpenMP ---
OPENMP_FLAG=""
if echo "int main(){}" | gcc -fopenmp -x c - -o /dev/null 2>/dev/null; then
    OPENMP_FLAG="-fopenmp"
    info "OpenMP:   enabled"
else
    warn "OpenMP not available"
fi

# --- sources (C library core) ---
HYDRO_SOURCES=(
    domain.c mesh.c quantity.c fluxes.c boundaries.c friction.c
    forcing.c operators.c structures.c sww.c geometry.c
    fit_interpolate.c coordinate_transforms.c sparse.c solver.c
    timestepping.c
)

# ============================================================================
# Commands
# ============================================================================

do_clean() {
    info "Cleaning ${BUILD_DIR} ..."
    rm -rf "${BUILD_DIR}"
    rm -f "${HYDRO_DIR}"/_core*.so
    info "Done."
}

do_build() {
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    # --- compile C sources → object files ---
    local OBJ_FILES=()
    for src in "${HYDRO_SOURCES[@]}"; do
        local obj="${src%.c}.o"
        OBJ_FILES+=("$obj")
        if [ "${SRC_DIR}/${src}" -nt "$obj" ] 2>/dev/null || [ ! -f "$obj" ]; then
            info "CC  ${src}"
            gcc -std=c17 -fPIC ${OPENMP_FLAG} \
                -D_DEFAULT_SOURCE \
                -I"${INC_DIR}" \
                ${NETCDF_CFLAGS} \
                -c "${SRC_DIR}/${src}" -o "$obj"
        fi
    done

    # --- link C library (static, PIC) ---
    if [ "${OBJ_FILES[*]}" -nt libhydro_pic.a ] 2>/dev/null || [ ! -f libhydro_pic.a ]; then
        info "AR  libhydro_pic.a"
        ar rcs libhydro_pic.a "${OBJ_FILES[@]}"
    fi

    # --- compile + link pybind11 extension ---
    local TARGET="_core${PY_SUFFIX}"
    if [ "${BINDINGS}" -nt "$TARGET" ] 2>/dev/null || [ ! -f "$TARGET" ]; then
        info "CXX ${TARGET}"
        local NUMPY_ARG=""
        [ -n "${NUMPY_INCLUDE:-}" ] && NUMPY_ARG="-I${NUMPY_INCLUDE}"

        g++ -std=c++17 -fPIC ${OPENMP_FLAG} -O2 \
            -shared \
            -I"${INC_DIR}" \
            -I"${PY_INCLUDE}" \
            -I"${PYBIND11_INCLUDE}" \
            ${NUMPY_ARG} \
            ${NETCDF_CFLAGS} \
            "${BINDINGS}" \
            libhydro_pic.a \
            ${NETCDF_LIBS} \
            -lm \
            -o "$TARGET"

        info "Built ${TARGET}"
    else
        info "UP-TO-DATE  ${TARGET}"
    fi

    echo ""
    info "Build complete — output in ${BUILD_DIR}/"
}

do_install() {
    do_clean
    do_build
    local SRC=$(ls "${BUILD_DIR}"/_core*.so 2>/dev/null | head -1)
    if [ -z "${SRC:-}" ]; then
        err "No _core*.so found in ${BUILD_DIR}"
        exit 1
    fi
    cp "$SRC" "${HYDRO_DIR}/"
    info "Installed $(basename "$SRC") → ${HYDRO_DIR}/"
}

# ============================================================================
# Main
# ============================================================================

case "${1:-build}" in
    clean)   do_clean ;;
    install) do_install ;;
    build)   do_build ;;
    *)
        echo "Usage: $0 {build|install|clean}"
        exit 1
        ;;
esac
