cmake_minimum_required(VERSION 3.16)
project(hydro_core VERSION 0.1.0 LANGUAGES C CXX)

# C17 / C++17 standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(HYDRO_OPENMP "Enable OpenMP parallelism" ON)
option(HYDRO_BUILD_TESTS "Build C tests" ON)
option(HYDRO_BUILD_SHARED "Build shared library" ON)
option(HYDRO_BUILD_STATIC "Build static library" ON)
option(HYDRO_BUILD_PYBIND11 "Build pybind11 Python extension" ON)

# Installation paths
include(GNUInstallDirs)

# Options
option(HYDRO_OPENMP "Enable OpenMP parallelism" ON)
option(HYDRO_BUILD_TESTS "Build C tests" ON)
option(HYDRO_BUILD_SHARED "Build shared library" ON)
option(HYDRO_BUILD_STATIC "Build static library" ON)

# Find dependencies
find_package(OpenMP)
if(OpenMP_FOUND AND HYDRO_OPENMP)
    message(STATUS "OpenMP enabled")
else()
    message(STATUS "OpenMP not found — disabling")
endif()

# NetCDF C library (required for SWW output)
# Use nc-config for discovery
execute_process(
    COMMAND nc-config --cflags
    OUTPUT_VARIABLE NETCDF_CFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NETCDF_CONFIG_RESULT
)
execute_process(
    COMMAND nc-config --libs
    OUTPUT_VARIABLE NETCDF_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT NETCDF_CONFIG_RESULT EQUAL 0)
    message(FATAL_ERROR "nc-config not found — NetCDF C library required for SWW output")
endif()

# Parse include dir from nc-config --cflags (typically -I/path)
string(REGEX REPLACE ".*-I([^ ]+).*" "\\1" NETCDF_INCLUDE_DIR "${NETCDF_CFLAGS}")
message(STATUS "NetCDF include: ${NETCDF_INCLUDE_DIR}")
message(STATUS "NetCDF libs: ${NETCDF_LIBS}")

# Create an imported target for netCDF
add_library(netcdf_imported INTERFACE)
target_include_directories(netcdf_imported INTERFACE ${NETCDF_INCLUDE_DIR})
target_link_libraries(netcdf_imported INTERFACE ${NETCDF_LIBS})

# Library sources
set(HYDRO_SOURCES
    src/domain.c
    src/mesh.c
    src/quantity.c
    src/fluxes.c
    src/boundaries.c
    src/friction.c
    src/forcing.c
    src/operators.c
    src/structures.c
    src/sww.c
    src/geometry.c
    src/fit_interpolate.c
    src/coordinate_transforms.c
    src/sparse.c
    src/solver.c
    src/timestepping.c
)

# Public headers
set(HYDRO_HEADERS
    include/hydro/hydro.h
    include/hydro/config.h
    include/hydro/types.h
    include/hydro/domain.h
    include/hydro/mesh.h
    include/hydro/quantity.h
    include/hydro/fluxes.h
    include/hydro/boundaries.h
    include/hydro/friction.h
    include/hydro/forcing.h
    include/hydro/operators.h
    include/hydro/structures.h
    include/hydro/sww.h
    include/hydro/geometry.h
    include/hydro/fit_interpolate.h
    include/hydro/coordinate_transforms.h
    include/hydro/sparse.h
    include/hydro/solver.h
    include/hydro/timestepping.h
)

# Build shared library
if(HYDRO_BUILD_SHARED)
    add_library(hydro_shared SHARED ${HYDRO_SOURCES})
    target_include_directories(hydro_shared
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(hydro_shared
        PUBLIC
            netcdf_imported
            ${OpenMP_C_LIBRARIES}
            m
    )
    set_target_properties(hydro_shared PROPERTIES
        OUTPUT_NAME hydro
        VERSION ${PROJECT_VERSION}
        SOVERSION 0
    )
    install(TARGETS hydro_shared
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# Build static library
if(HYDRO_BUILD_STATIC)
    add_library(hydro_static STATIC ${HYDRO_SOURCES})
    target_include_directories(hydro_static
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(hydro_static
        PUBLIC
            netcdf_imported
            ${OpenMP_C_LIBRARIES}
            m
    )
    set_target_properties(hydro_static PROPERTIES
        OUTPUT_NAME hydro
    )
    install(TARGETS hydro_static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# Install headers
install(FILES ${HYDRO_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hydro
)

# Tests
if(HYDRO_BUILD_TESTS)
    enable_testing()
    add_executable(test_phase0 tests/test_phase0.c)
    target_include_directories(test_phase0 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(test_phase0
        hydro_static
        netcdf_imported
        ${OpenMP_C_LIBRARIES}
        m
    )
    add_test(NAME phase0 COMMAND test_phase0)

    add_executable(test_phase1 tests/test_phase1.c)
    target_include_directories(test_phase1 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(test_phase1
        hydro_static
        netcdf_imported
        ${OpenMP_C_LIBRARIES}
        m
    )
    add_test(NAME phase1 COMMAND test_phase1)

    add_executable(test_phase1b tests/test_phase1b.c)
    target_include_directories(test_phase1b PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(test_phase1b
        hydro_static
        netcdf_imported
        ${OpenMP_C_LIBRARIES}
        m
    )
    add_test(NAME phase1b COMMAND test_phase1b)

    add_executable(test_phase2 tests/test_phase2.c)
    target_include_directories(test_phase2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(test_phase2
        hydro_static
        netcdf_imported
        ${OpenMP_C_LIBRARIES}
        m
    )
    add_test(NAME phase2 COMMAND test_phase2)

    add_executable(test_phase3 tests/test_phase3.c)
    target_include_directories(test_phase3 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(test_phase3
        hydro_static
        netcdf_imported
        ${OpenMP_C_LIBRARIES}
        m
    )
    add_test(NAME phase3 COMMAND test_phase3)

    add_executable(test_phase4 tests/test_phase4.c)
    target_include_directories(test_phase4 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(test_phase4
        hydro_static
        netcdf_imported
        ${OpenMP_C_LIBRARIES}
        m
    )
    add_test(NAME phase4 COMMAND test_phase4)

    add_executable(test_phase5 tests/test_phase5.c)
    target_include_directories(test_phase5 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(test_phase5
        hydro_static
        netcdf_imported
        ${OpenMP_C_LIBRARIES}
        m
    )
    add_test(NAME phase5 COMMAND test_phase5)

    add_executable(test_phase6 tests/test_phase6.c)
    target_include_directories(test_phase6 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(test_phase6
        hydro_static
        netcdf_imported
        ${OpenMP_C_LIBRARIES}
        m
    )
    add_test(NAME phase6 COMMAND test_phase6)
endif()

# ===========================================================================
# pybind11 Python extension
# ===========================================================================

if(HYDRO_BUILD_PYBIND11)
    find_package(pybind11 REQUIRED)

    pybind11_add_module(_core python/bindings.cpp)

    target_include_directories(_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(_core PRIVATE
        hydro_shared
        netcdf_imported
        ${OpenMP_C_LIBRARIES}
    )

    # Install into the hydro package directory
    install(TARGETS _core
        LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/hydro"
    )
endif()
