/**
 * coordinate_transforms.c — UTM projection and Geo-reference
 *
 * Ported from ANUGA:
 *   anuga/coordinate_transforms/redfearn.py
 *   anuga/coordinate_transforms/geo_reference.py
 *
 * Implements Redfearn's formula for lat/lon <-> UTM conversion
 * using GRS80 (WGS84-compatible) ellipsoid.
 */

#include "hydro/coordinate_transforms.h"
#include <math.h>
#include <string.h>

/* =========================================================================
 * Geo-reference
 * ========================================================================= */

void hydro_geo_ref_init(hydro_geo_ref_t* gr)
{
    gr->zone = -1;
    gr->xllcorner = 0.0;
    gr->yllcorner = 0.0;
    gr->hemisphere = -1;
    gr->false_easting = 500000.0;
    gr->false_northing = 0.0;
    strcpy(gr->datum, "wgs84");
    strcpy(gr->projection, "UTM");
    strcpy(gr->units, "m");
}

void hydro_geo_ref_set_utm(hydro_geo_ref_t* gr, hydro_int zone,
                           int hemisphere)
{
    gr->zone = zone;
    gr->hemisphere = hemisphere;
    gr->false_easting = 500000.0;
    gr->false_northing = (hemisphere == 1) ? 10000000.0 : 0.0;
}

int hydro_geo_ref_get_epsg(const hydro_geo_ref_t* gr)
{
    if (gr->zone < 1 || gr->zone > 60) return 0;
    if (gr->hemisphere == 1) return 32700 + (int)gr->zone;
    if (gr->hemisphere == 0) return 32600 + (int)gr->zone;
    return 0;
}

void hydro_geo_ref_to_absolute(const hydro_geo_ref_t* gr,
                               double* points, hydro_int N)
{
    for (hydro_int i = 0; i < N; i++)
    {
        points[2 * i] += gr->xllcorner;
        points[2 * i + 1] += gr->yllcorner;
    }
}

void hydro_geo_ref_to_relative(const hydro_geo_ref_t* gr,
                               double* points, hydro_int N)
{
    for (hydro_int i = 0; i < N; i++)
    {
        points[2 * i] -= gr->xllcorner;
        points[2 * i + 1] -= gr->yllcorner;
    }
}

/* =========================================================================
 * Redfearn's Formula: Lat/Lon -> UTM
 * ========================================================================= */

void hydro_redfearn_latlon_to_utm(double lat, double lon,
                                  hydro_int* zone,
                                  double* easting, double* northing,
                                  double false_easting,
                                  double false_northing)
{
    /* GRS80 / GDA2020 ellipsoid constants */
    const double a = 6378137.0; /* semi-major axis */
    const double inv_f = 298.257222101; /* inverse flattening */
    const double K0 = 0.9996; /* central scale factor */
    const double zone_width = 6.0; /* degrees per zone */
    const double lon_cm_zone0 = -183.0; /* central meridian of zone 0 */
    const double lon_west_zone0 = -186.0; /* west edge of zone 0 */

    /* Derived constants */
    double f = 1.0 / inv_f;
    double b = a * (1.0 - f);
    double e2 = 2.0 * f - f * f;
    double e4 = e2 * e2;
    double e6 = e2 * e4;

    /* n ratio */
    double n = (a - b) / (a + b);
    double n2 = n * n;
    double n4 = n2 * n2;

    /* False easting/northing defaults */
    if (false_easting < 0) false_easting = 500000.0;
    if (false_northing < 0)
    {
        false_northing = (lat < 0) ? 10000000.0 : 0.0;
    }

    /* Determine UTM zone */
    *zone = (hydro_int)((lon - lon_west_zone0) / zone_width);
    if (*zone > 60) *zone -= 60;

    double central_meridian = (*zone) * zone_width + lon_cm_zone0;

    /* Convert to radians */
    double phi = lat * M_PI / 180.0;
    double omega = (lon - central_meridian) * M_PI / 180.0;

    double sinphi = sin(phi);
    double cosphi = cos(phi);
    double cosphi2 = cosphi * cosphi;
    double cosphi3 = cosphi2 * cosphi;
    double cosphi4 = cosphi2 * cosphi2;
    double cosphi5 = cosphi4 * cosphi;
    double cosphi7 = cosphi4 * cosphi3;

    double t = tan(phi);
    double t2 = t * t;
    double t4 = t2 * t2;
    double t6 = t2 * t4;

    /* Radii of curvature */
    double tmp = 1.0 - e2 * sinphi * sinphi;
    double nu = a / sqrt(tmp);
    double psi = nu / (a * (1.0 - e2) / (tmp * sqrt(tmp)));
    /* psi = nu/rho, rho = a*(1-e2)/(1-e2*sinphi^2)^1.5 */
    double psi2 = psi * psi;
    double psi3 = psi2 * psi;
    double psi4 = psi2 * psi2;

    /* Meridian distance */
    double A0 = 1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0;
    double A2 = 3.0 / 8.0 * (e2 + e4 / 4.0 + 15.0 * e6 / 128.0);
    double A4 = 15.0 / 256.0 * (e4 + 3.0 * e6 / 4.0);
    double A6 = 35.0 * e6 / 3072.0;

    double m = a * (A0 * phi - A2 * sin(2.0 * phi) + A4 * sin(4.0 * phi) - A6 * sin(6.0 * phi));

    double omega2 = omega * omega;
    double omega4 = omega2 * omega2;
    double omega6 = omega4 * omega2;
    double omega8 = omega4 * omega4;

    /* Easting */
    double E1 = nu * omega * cosphi;
    double E2 = nu * cosphi3 * (psi - t2) * omega2 * omega / 6.0;
    double E3 = nu * cosphi5 *
    (4.0 * psi3 * (1.0 - 6.0 * t2) + psi2 * (1.0 + 8.0 * t2)
        - 2.0 * psi * t2 + t4) * omega4 * omega / 120.0;
    double E4 = nu * cosphi7 *
        (61.0 - 479.0 * t2 + 179.0 * t4 - t6) * omega6 * omega / 5040.0;
    *easting = false_easting + K0 * (E1 + E2 + E3 + E4);

    /* Northing */
    double N1 = nu * sinphi * cosphi * omega2 / 2.0;
    double N2 = nu * sinphi * cosphi3 *
        (4.0 * psi2 + psi - t2) * omega4 / 24.0;
    double N3 = nu * sinphi * cosphi5 *
    (8.0 * psi4 * (11.0 - 24.0 * t2) - 28.0 * psi3 * (1.0 - 6.0 * t2)
        + psi2 * (1.0 - 32.0 * t2) - psi * 2.0 * t2 + t4 - t2) * omega6 / 720.0;
    double N4 = nu * sinphi * cosphi7 *
        (1385.0 - 3111.0 * t2 + 543.0 * t4 - t6) * omega8 / 40320.0;
    *northing = false_northing + K0 * (m + N1 + N2 + N3 + N4);
}

void hydro_convert_latlon_to_utm_batch(
    const double* lats, const double* lons, hydro_int N,
    hydro_int* zones, double* eastings, double* northings,
    double false_easting, double false_northing)
{
    for (hydro_int i = 0; i < N; i++)
    {
        hydro_int z;
        double e, n;
        hydro_redfearn_latlon_to_utm(lats[i], lons[i], &z, &e, &n,
                                     false_easting, false_northing);
        if (zones) zones[i] = z;
        eastings[i] = e;
        northings[i] = n;
    }
}

/* =========================================================================
 * Inverse Redfearn: UTM -> Lat/Lon
 * ========================================================================= */

void hydro_redfearn_utm_to_latlon(hydro_int zone,
                                  double easting, double northing,
                                  int is_southern,
                                  double* lat, double* lon)
{
    /* GRS80 constants */
    const double a = 6378137.0;
    const double inv_f = 298.257222101;
    const double K0 = 0.9996;
    const double zone_width = 6.0;
    const double lon_cm_zone0 = -183.0;

    double f = 1.0 / inv_f;
    double b = a * (1.0 - f);
    double e2 = 2.0 * f - f * f;
    double e4 = e2 * e2;
    double e6 = e2 * e4;
    double e2_ = e2 / (1.0 - e2);

    double n = (a - b) / (a + b);
    double n2 = n * n;
    double n3 = n * n2;
    double n4 = n2 * n2;

    double false_easting = 500000.0;
    double false_northing = is_southern ? 10000000.0 : 0.0;

    double central_meridian = zone * zone_width + lon_cm_zone0;

    /* Remove false easting/northing and scale factor */
    double x = (easting - false_easting) / K0;
    double y = (northing - false_northing) / K0;

    /* Foot-point latitude */
    double G = a * (1.0 - n) * (1.0 - n2) * (1.0 + 9.0 * n2 / 4.0 + 225.0 * n4 / 64.0);
    double phif = y / G; /* initial estimate (radians) */

    /* Iterate to convergence (foot-point latitude) */
    for (int iter = 0; iter < 5; iter++)
    {
        double A0 = 1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0;
        double A2 = 3.0 / 8.0 * (e2 + e4 / 4.0 + 15.0 * e6 / 128.0);
        double A4 = 15.0 / 256.0 * (e4 + 3.0 * e6 / 4.0);
        double A6 = 35.0 * e6 / 3072.0;
        double mf = a * (A0 * phif - A2 * sin(2.0 * phif) + A4 * sin(4.0 * phif) - A6 * sin(6.0 * phif));
        phif += (y - mf) / G;
    }

    double sinphif = sin(phif);
    double cosphif = cos(phif);
    double t = tan(phif);
    double t2 = t * t;
    double t4 = t2 * t2;

    double tmp = 1.0 - e2 * sinphif * sinphif;
    double nu = a / sqrt(tmp);
    double rho = a * (1.0 - e2) / (tmp * sqrt(tmp));
    double psi = nu / rho;
    double psi2 = psi * psi;

    /* Compute latitude */
    double term1 = t * x * x / (2.0 * rho * nu);
    double term2 = t * x * x * x * x / (24.0 * rho * nu * nu * nu) *
        (5.0 + 3.0 * t2 + e2_ - 9.0 * t2 * e2_ - 4.0 * e2_ * e2_);
    double term3 = t * x * x * x * x * x * x / (720.0 * rho * nu * nu * nu * nu * nu) *
        (61.0 + 90.0 * t2 + 45.0 * t4);
    *lat = (phif - term1 + term2 - term3) * 180.0 / M_PI;

    /* Compute longitude */
    double tmp1 = 1.0 - e2 * sinphif * sinphif;
    double nu2 = a / sqrt(tmp1);
    double secphi = 1.0 / cosphif;
    double L1 = x * secphi / nu2;
    double L2 = x * x * x * secphi / (6.0 * nu2 * nu2 * nu2) *
        (psi + 2.0 * t2);
    double L3 = x * x * x * x * x * secphi / (120.0 * nu2 * nu2 * nu2 * nu2 * nu2) *
        (5.0 + 28.0 * t2 + 24.0 * t4);
    double L4 = x * x * x * x * x * x * x * secphi / (5040.0 * nu2 * nu2 * nu2 * nu2 * nu2 * nu2 * nu2) *
        (61.0 + 662.0 * t2 + 1320.0 * t4 + 720.0 * t4 * t2);
    *lon = central_meridian + (L1 - L2 + L3 - L4) * 180.0 / M_PI;
}

/* =========================================================================
 * DMS Conversion
 * ========================================================================= */

double hydro_dms_to_decimal_degrees(int dd, int mm, double ss)
{
    double sign = (dd < 0) ? -1.0 : 1.0;
    return sign * (fabs((double)dd) + mm / 60.0 + ss / 3600.0);
}

void hydro_decimal_degrees_to_dms(double dec, int* dd, int* mm, double* ss)
{
    double sign = (dec < 0) ? -1.0 : 1.0;
    double adec = fabs(dec);
    *dd = (int)adec;
    double f = adec - (*dd);
    *mm = (int)(f * 60.0);
    *ss = (f * 60.0 - (*mm)) * 60.0;
    *dd = (int)(sign * (*dd));
}
