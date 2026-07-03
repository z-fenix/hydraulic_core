/**
 * mesh.c — Mesh topology: neighbour and boundary structure
 *
 * Adapted from:
 *   anuga/abstract_2d_finite_volumes/neighbour_table.cpp
 *   anuga/abstract_2d_finite_volumes/neighbour_mesh.py
 */

#include "hydro/mesh.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ==========================================================================
 * Simple hash table for edge keys (node_id pairs)
 *
 * Replaces C++ std::unordered_map<edge_key_t, edge_t> from neighbour_table.cpp
 * ========================================================================== */

typedef struct {
    hydro_int i;    /* smaller node id */
    hydro_int j;    /* larger node id */
} edge_key_t;

typedef struct {
    hydro_int vol_id;   /* triangle index */
    hydro_int edge_id;  /* which edge (0,1,2) */
} edge_val_t;

typedef struct {
    edge_key_t key;
    edge_val_t val;
    int        occupied;
} hash_entry_t;

#define HASH_TABLE_SIZE_INIT 65536

static hydro_int edge_hash(const edge_key_t* key, hydro_int table_size) {
    /* FNV-1a hash of two 64-bit ints */
    hydro_uint h = 14695981039346656037ULL;
    h ^= (hydro_uint)key->i;
    h *= 1099511628211ULL;
    h ^= (hydro_uint)key->j;
    h *= 1099511628211ULL;
    return (hydro_int)(h % (hydro_uint)table_size);
}

static edge_key_t make_edge_key(hydro_int a, hydro_int b) {
    edge_key_t key;
    if (a < b) { key.i = a; key.j = b; }
    else       { key.i = b; key.j = a; }
    return key;
}

/* Linear probe hash table */
typedef struct {
    hash_entry_t* entries;
    hydro_int     size;
    hydro_int     count;
} edge_hash_t;

static edge_hash_t* edge_hash_create(hydro_int capacity) {
    edge_hash_t* ht = (edge_hash_t*)calloc(1, sizeof(edge_hash_t));
    if (!ht) return NULL;
    ht->size = capacity;
    ht->entries = (hash_entry_t*)calloc((size_t)capacity, sizeof(hash_entry_t));
    if (!ht->entries) { free(ht); return NULL; }
    return ht;
}

static void edge_hash_destroy(edge_hash_t* ht) {
    if (ht) {
        free(ht->entries);
        free(ht);
    }
}

static int edge_hash_insert(edge_hash_t* ht, edge_key_t key, edge_val_t val) {
    hydro_int idx = edge_hash(&key, ht->size);
    /* Allow duplicates — each triangle sharing an edge creates an entry.
     * No infinite-loop guard needed: table size is 2 * n_edges, so even
     * with all 600 edges as duplicates (300 keys × 2 entries each)
     * we stay well below capacity. */
    while (ht->entries[idx].occupied) {
        idx = (idx + 1) % ht->size;
    }
    ht->entries[idx].key = key;
    ht->entries[idx].val = val;
    ht->entries[idx].occupied = 1;
    ht->count++;
    return 0;
}

static int edge_hash_find(edge_hash_t* ht, edge_key_t key, edge_val_t* val) {
    hydro_int idx = edge_hash(&key, ht->size);
    hydro_int start = idx;
    while (ht->entries[idx].occupied) {
        edge_key_t* ek = &ht->entries[idx].key;
        if (ek->i == key.i && ek->j == key.j) {
            *val = ht->entries[idx].val;
            return 0;  /* found */
        }
        idx = (idx + 1) % ht->size;
        if (idx == start) break;  /* wrap-around guard */
    }
    return -1;  /* not found */
}

/* ==========================================================================
 * Neighbour structure builder
 *
 * Ported from neighbour_table.cpp:_build_neighbour_structure()
 * ========================================================================== */

int hydro_mesh_build_neighbour_structure(hydro_domain_t* d) {
    hydro_int M = d->number_of_elements;
    hydro_int n_edges = d->number_of_edges;
    hydro_int k, k3, edge_id;
    hydro_int n0, n1, n2;
    edge_key_t key;
    edge_val_t val;
    hydro_int boundary_count = 0;
    /* Hash table must hold 3*M entries (one per directed edge).
     * Use 4× to keep linear-probe chains short. */
    hydro_int table_size = HASH_TABLE_SIZE_INIT;
    if (table_size < n_edges * 4) table_size = n_edges * 4;

    /* ---- Step 0: Initialise ---- */
    for (k = 0; k < n_edges; k++) {
        d->neighbours[k]      = -1;
        d->neighbour_edges[k] = -1;
        d->already_computed_flux[k] = 0;
    }
    for (k = 0; k < M; k++) {
        d->number_of_boundaries[k] = 0;
    }

    /* ---- Step 1: Populate hash table with all edges ---- */
    edge_hash_t* ht = edge_hash_create(table_size);
    if (!ht) return -1;

    for (k = 0; k < M; k++) {
        k3 = 3 * k;
        n0 = d->triangles[k3];
        n1 = d->triangles[k3 + 1];
        n2 = d->triangles[k3 + 2];

        /* Edge 0: between n1 and n2 */
        key = make_edge_key(n1, n2);
        val.vol_id = k;
        val.edge_id = 0;
        edge_hash_insert(ht, key, val);

        /* Edge 1: between n2 and n0 */
        key = make_edge_key(n2, n0);
        val.vol_id = k;
        val.edge_id = 1;
        edge_hash_insert(ht, key, val);

        /* Edge 2: between n0 and n1 */
        key = make_edge_key(n0, n1);
        val.vol_id = k;
        val.edge_id = 2;
        edge_hash_insert(ht, key, val);
    }

    /* ---- Step 2: Find neighbours ---- */
    for (k = 0; k < M; k++) {
        k3 = 3 * k;
        n0 = d->triangles[k3];
        n1 = d->triangles[k3 + 1];
        n2 = d->triangles[k3 + 2];

        for (edge_id = 0; edge_id < 3; edge_id++) {
            hydro_int ni;

            if (edge_id == 0) {
                key = make_edge_key(n1, n2);
            } else if (edge_id == 1) {
                key = make_edge_key(n2, n0);
            } else {
                key = make_edge_key(n0, n1);
            }

            if (edge_hash_find(ht, key, &val) == 0) {
                /* Check all entries with this key (linear probe) */
                hydro_int idx = edge_hash(&key, ht->size);
                hydro_int found_count = 0;
                hydro_int my_idx = k3 + edge_id;

                while (ht->entries[idx].occupied) {
                    edge_key_t* ek = &ht->entries[idx].key;
                    if (ek->i == key.i && ek->j == key.j) {
                        hydro_int other_k = ht->entries[idx].val.vol_id;
                        hydro_int other_e = ht->entries[idx].val.edge_id;
                        if (other_k != k) {
                            d->neighbours[my_idx] = other_k;
                            d->neighbour_edges[my_idx] = other_e;
                            found_count++;
                        }
                    }
                    idx = (idx + 1) % ht->size;
                }

                /* If this is the only triangle with this edge, it's a boundary */
                /* We count boundary edges in Step 3 */
            }

            /* If neighbour is still -1 after the lookup above,
             * this edge belongs to only one triangle → boundary.
             * The -1 marker is handled in Step 3 (surrogate neighbours). */
        }
    }

    edge_hash_destroy(ht);

    /* ---- Step 3: Build surrogate neighbours and count boundaries ---- */
    for (k = 0; k < M; k++) {
        hydro_int boundary_count_k = 0;
        k3 = 3 * k;
        for (edge_id = 0; edge_id < 3; edge_id++) {
            hydro_int ni = k3 + edge_id;
            if (d->neighbours[ni] < 0) {
                /* Boundary edge: surrogate neighbour = self */
                d->surrogate_neighbours[ni] = k;
                boundary_count_k++;
            } else {
                d->surrogate_neighbours[ni] = d->neighbours[ni];
            }
        }
        d->number_of_boundaries[k] = boundary_count_k;
        d->boundary_length += boundary_count_k;
    }

    return 0;
}

/* ==========================================================================
 * Boundary structure builder
 *
 * Ported from neighbour_mesh.py
 * ========================================================================== */

int hydro_mesh_build_boundary_structure(hydro_domain_t* d) {
    hydro_int M = d->number_of_elements;
    hydro_int bl = d->boundary_length;
    hydro_int k, k3, edge_id, bi;

    if (bl == 0) {
        return 0;  /* no boundaries */
    }

    /* Allocate boundary arrays */
    d->boundary_tags  = (hydro_int*)calloc((size_t)bl, sizeof(hydro_int));
    d->boundary_edges = (hydro_int*)calloc((size_t)bl, sizeof(hydro_int));

    d->stage_boundary_values    = (double*)calloc((size_t)bl, sizeof(double));
    d->xmom_boundary_values     = (double*)calloc((size_t)bl, sizeof(double));
    d->ymom_boundary_values     = (double*)calloc((size_t)bl, sizeof(double));
    d->bed_boundary_values      = (double*)calloc((size_t)bl, sizeof(double));
    d->height_boundary_values   = (double*)calloc((size_t)bl, sizeof(double));
    d->xvelocity_boundary_values = (double*)calloc((size_t)bl, sizeof(double));
    d->yvelocity_boundary_values = (double*)calloc((size_t)bl, sizeof(double));

    if (!d->boundary_tags || !d->boundary_edges) {
        fprintf(stderr, "hydro: boundary allocation failed\n");
        return -1;
    }

    /* Enumerate boundary edges and assign unique negative neighbour indices.
       The convention: neighbours[edge] = -(bi + 1) where bi = boundary index.
       This allows flux computation to compute bnd_idx = -neighbour - 1. */
    bi = 0;
    for (k = 0; k < M; k++) {
        k3 = 3 * k;
        for (edge_id = 0; edge_id < 3; edge_id++) {
            hydro_int ni = k3 + edge_id;
            if (d->neighbours[ni] < 0) {
                d->neighbours[ni] = -(bi + 1);  /* unique negative index */
                d->boundary_edges[bi] = ni;
                /* Use user-provided tag from boundary_tag_map, or default to 1 */
                d->boundary_tags[bi] = (d->boundary_tag_map && d->boundary_tag_map[ni] > 0)
                    ? d->boundary_tag_map[ni] : 1;
                bi++;
            }
        }
    }

    return 0;
}
