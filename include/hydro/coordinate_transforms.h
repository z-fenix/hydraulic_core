#ifndef HYDRO_COORDINATE_TRANSFORMS_H
#define HYDRO_COORDINATE_TRANSFORMS_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Geo-reference: coordinate system origin for numerical stability.
 *
 * All mesh coordinates are stored as offsets from (xllcorner, yllcorner),
 * which is typically a UTM coordinate. This keeps local coordinates small.
 * ========================================================================= */

typedef struct {
    hydro_int zone;          /* UTM zone 1-60, or -1 for non-UTM */
    double xllcorner;        /* X (easting) of local origin in UTM */
    double yllcorner;        /* Y (northing) of local origin in UTM */
    int hemisphere;          /* 0=northern, 1=southern, -1=undefined */
    double false_easting;    /* Usually 500000 */
    double false_northing;   /* 0 (north) or 10,000,000 (south) */
    char datum[32];          /* e.g. "wgs84" */
    char projection[32];     /* e.g. "UTM" */
    char units[16];          /* e.g. "m" */
} hydro_geo_ref_t;

/* Default geo-reference (zone=-1, no UTM framework) */
void hydro_geo_ref_init(hydro_geo_ref_t* gr);

/* Set standard UTM zone with hemisphere */
void hydro_geo_ref_set_utm(hydro_geo_ref_t* gr, hydro_int zone,
                            int hemisphere);

/* Compute EPSG code from UTM zone and hemisphere.
 * WGS84 UTM: 32600 + zone (north), 32700 + zone (south).
 * Returns 0 if zone is -1.
 */
int hydro_geo_ref_get_epsg(const hydro_geo_ref_t* gr);

/* Convert local (offset) coordinates to absolute UTM coordinates.
 * Points: [x0,y0, x1,y1, ...] length 2*N
 * geo_ref: the reference defining xllcorner/yllcorner
 */
void hydro_geo_ref_to_absolute(const hydro_geo_ref_t* gr,
                                double* points, hydro_int N);

/* Convert absolute UTM coordinates to local (offset) coordinates */
void hydro_geo_ref_to_relative(const hydro_geo_ref_t* gr,
                                double* points, hydro_int N);

/* =========================================================================
 * UTM Projection (Redfearn's Formula)
 *
 * Converts latitude/longitude (decimal degrees) to UTM easting/northing.
 * Uses GRS80 ellipsoid parameters (WGS84-compatible).
 * ========================================================================= */

/* Convert a single lat/lon point to UTM.
 * lat, lon: in decimal degrees
 * zone: output UTM zone (1-60)
 * easting, northing: output UTM coordinates in metres
 * false_easting, false_northing: if < 0, auto-determined from hemisphere
 */
void hydro_redfearn_latlon_to_utm(double lat, double lon,
                                   hydro_int* zone,
                                   double* easting, double* northing,
                                   double false_easting,
                                   double false_northing);

/* Batch conversion of lat/lon to UTM.
 * lats, lons: arrays of length N
 * zones, eastings, northings: output arrays of length N
 * If zones is NULL, auto-determine zones from longitude.
 */
void hydro_convert_latlon_to_utm_batch(
    const double* lats, const double* lons, hydro_int N,
    hydro_int* zones, double* eastings, double* northings,
    double false_easting, double false_northing);

/* =========================================================================
 * UTM to Lat/Lon (Inverse Redfearn)
 * ========================================================================= */

/* Convert UTM coordinates back to lat/lon.
 * zone: UTM zone (1-60)
 * easting, northing: UTM coordinates in metres
 * lat, lon: output in decimal degrees
 * is_southern: 1 if southern hemisphere, 0 if northern
 */
void hydro_redfearn_utm_to_latlon(hydro_int zone,
                                   double easting, double northing,
                                   int is_southern,
                                   double* lat, double* lon);

/* =========================================================================
 * DMS (Degrees-Minutes-Seconds) conversion
 * ========================================================================= */

double hydro_dms_to_decimal_degrees(int dd, int mm, double ss);
void hydro_decimal_degrees_to_dms(double dec, int* dd, int* mm, double* ss);

#ifdef __cplusplus
}
#endif

#endif /* HYDRO_COORDINATE_TRANSFORMS_H */
