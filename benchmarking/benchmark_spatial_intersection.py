"""
Benchmark: GeoPandas vs DuckDB for Spatial Intersection
========================================================
Compares two approaches for reading an AOI and a dataset, then performing
a spatial intersection to identify geometries that intersect the AOI.

Scenario 1: GeoPandas reads AOI (.shp) and dataset (.gdb) directly.
Scenario 2: DuckDB reads the dataset from GeoParquet + AOI via GeoPandas.
Scenario 3: DuckDB stateless – ST_Read the .gdb/.shp directly, query inline, no tables.

Requirements:
    pip install geopandas duckdb pyarrow fiona shapely

Usage:
    python benchmark_spatial_intersection.py

Notes:
    - Update the paths in main() to match your environment.
    - The AOI and dataset must be in the same CRS (or reprojection is applied).
    - The GeoParquet file should be an export of the same .gdb dataset.
    - To create the GeoParquet file from your feature class:
        import geopandas as gpd
        gdf = gpd.read_file("data.gdb", layer="my_layer")
        gdf.to_parquet("dataset.parquet")
"""

import time
import tracemalloc
import warnings
from dataclasses import dataclass
from typing import Optional
import geopandas as gpd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    scenario: str
    read_time_s: float
    intersect_time_s: float
    total_time_s: float
    peak_memory_mb: float
    result_count: int
    dataset_count: int
    aoi_count: int


# ---------------------------------------------------------------------------
# Scenario 1a – GeoPandas (full read + sjoin)
# ---------------------------------------------------------------------------
def scenario_geopandas(
    aoi_path: str,
    dataset_path: str,
    aoi_layer: Optional[str] = None,
    dataset_layer: Optional[str] = None,
) -> BenchmarkResult:
    """
    Read the entire dataset into memory, then filter with sjoin.
    No spatial pre-filtering at the GDAL level.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 1a: GeoPandas (full read + sjoin)")
    print("=" * 60)

    tracemalloc.start()

    # -- Read --
    t0 = time.perf_counter()
    aoi_gdf = gpd.read_file(aoi_path, layer=aoi_layer)
    dataset_gdf = gpd.read_file(dataset_path, layer=dataset_layer)
    read_time = time.perf_counter() - t0

    # Ensure matching CRS
    if aoi_gdf.crs != dataset_gdf.crs:
        print(f"  Reprojecting AOI from {aoi_gdf.crs} → {dataset_gdf.crs}")
        aoi_gdf = aoi_gdf.to_crs(dataset_gdf.crs)

    aoi_count = len(aoi_gdf)
    dataset_count = len(dataset_gdf)
    print(f"  AOI features:     {aoi_count:,}")
    print(f"  Dataset features: {dataset_count:,}")

    # -- Intersect --
    t1 = time.perf_counter()
    result_gdf = gpd.sjoin(dataset_gdf, aoi_gdf, how="inner", predicate="intersects")
    result_gdf = result_gdf.drop_duplicates(subset=result_gdf.columns.difference(["index_right"]))
    intersect_time = time.perf_counter() - t1

    total_time = read_time + intersect_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result_count = len(result_gdf)
    print(f"  Intersecting features: {result_count:,}")
    print(f"  Read time:      {read_time:.4f} s")
    print(f"  Intersect time: {intersect_time:.4f} s")
    print(f"  Total time:     {total_time:.4f} s")
    print(f"  Peak memory:    {peak_mem / 1024 / 1024:.2f} MB")

    return BenchmarkResult(
        scenario="GPD full+sjoin",
        read_time_s=round(read_time, 4),
        intersect_time_s=round(intersect_time, 4),
        total_time_s=round(total_time, 4),
        peak_memory_mb=round(peak_mem / 1024 / 1024, 2),
        result_count=result_count,
        dataset_count=dataset_count,
        aoi_count=aoi_count,
    )


# ---------------------------------------------------------------------------
# Scenario 1b – GeoPandas (mask-filtered read + sjoin)
# ---------------------------------------------------------------------------
def scenario_geopandas_mask(
    aoi_path: str,
    dataset_path: str,
    aoi_layer: Optional[str] = None,
    dataset_layer: Optional[str] = None,
) -> BenchmarkResult:
    """
    Read only features whose bbox intersects the AOI using the mask parameter
    (GDAL spatial filter), then do exact intersect with sjoin.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 1b: GeoPandas (mask-filtered read + sjoin)")
    print("=" * 60)

    tracemalloc.start()

    # -- Read AOI --
    t0 = time.perf_counter()
    aoi_gdf = gpd.read_file(aoi_path, layer=aoi_layer)
    aoi_count = len(aoi_gdf)

    # Union AOI into a single geometry for the mask
    if aoi_count == 1:
        aoi_geom = aoi_gdf.geometry.iloc[0]
    else:
        aoi_geom = aoi_gdf.geometry.union_all()

    # Read dataset with mask — GDAL only returns features whose bbox
    # overlaps the mask geometry, dramatically reducing I/O for large datasets
    dataset_gdf = gpd.read_file(
        dataset_path, layer=dataset_layer, mask=aoi_geom
    )
    read_time = time.perf_counter() - t0

    # Ensure matching CRS
    if aoi_gdf.crs != dataset_gdf.crs:
        print(f"  Reprojecting AOI from {aoi_gdf.crs} → {dataset_gdf.crs}")
        aoi_gdf = aoi_gdf.to_crs(dataset_gdf.crs)

    dataset_count = len(dataset_gdf)
    print(f"  AOI features:     {aoi_count:,}")
    print(f"  Dataset features (after mask): {dataset_count:,}")

    # -- Intersect (sjoin, same as Scenario 1a for consistency) --
    t1 = time.perf_counter()
    result_gdf = gpd.sjoin(dataset_gdf, aoi_gdf, how="inner", predicate="intersects")
    result_gdf = result_gdf.drop_duplicates(subset=result_gdf.columns.difference(["index_right"]))
    intersect_time = time.perf_counter() - t1

    total_time = read_time + intersect_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result_count = len(result_gdf)
    print(f"  Intersecting features: {result_count:,}")
    print(f"  Read time:      {read_time:.4f} s")
    print(f"  Intersect time: {intersect_time:.4f} s")
    print(f"  Total time:     {total_time:.4f} s")
    print(f"  Peak memory:    {peak_mem / 1024 / 1024:.2f} MB")

    return BenchmarkResult(
        scenario="GPD mask+sjoin",
        read_time_s=round(read_time, 4),
        intersect_time_s=round(intersect_time, 4),
        total_time_s=round(total_time, 4),
        peak_memory_mb=round(peak_mem / 1024 / 1024, 2),
        result_count=result_count,
        dataset_count=dataset_count,
        aoi_count=aoi_count,
    )


# ---------------------------------------------------------------------------
# Scenario 1c – GeoPandas (bbox-filtered read + sjoin)
# ---------------------------------------------------------------------------
def scenario_geopandas_bbox(
    aoi_path: str,
    dataset_path: str,
    aoi_layer: Optional[str] = None,
    dataset_layer: Optional[str] = None,
) -> BenchmarkResult:
    """
    Read only features within the AOI's bounding box using the bbox parameter,
    then do exact intersect with sjoin.
    bbox is a simpler/faster filter than mask (just a rectangle, no geometry test).
    """
    print("\n" + "=" * 60)
    print("SCENARIO 1c: GeoPandas (bbox-filtered read + sjoin)")
    print("=" * 60)

    tracemalloc.start()

    # -- Read AOI --
    t0 = time.perf_counter()
    aoi_gdf = gpd.read_file(aoi_path, layer=aoi_layer)
    aoi_count = len(aoi_gdf)

    # Get the bounding box of the AOI (minx, miny, maxx, maxy)
    aoi_bbox = tuple(aoi_gdf.total_bounds)

    # Read dataset with bbox — GDAL only returns features whose bbox
    # overlaps the given bounding box rectangle
    dataset_gdf = gpd.read_file(
        dataset_path, layer=dataset_layer, bbox=aoi_bbox
    )
    read_time = time.perf_counter() - t0

    # Ensure matching CRS
    if aoi_gdf.crs != dataset_gdf.crs:
        print(f"  Reprojecting AOI from {aoi_gdf.crs} → {dataset_gdf.crs}")
        aoi_gdf = aoi_gdf.to_crs(dataset_gdf.crs)

    dataset_count = len(dataset_gdf)
    print(f"  AOI features:     {aoi_count:,}")
    print(f"  Dataset features (after bbox): {dataset_count:,}")

    # -- Intersect (sjoin, same as other scenarios for consistency) --
    t1 = time.perf_counter()
    result_gdf = gpd.sjoin(dataset_gdf, aoi_gdf, how="inner", predicate="intersects")
    result_gdf = result_gdf.drop_duplicates(subset=result_gdf.columns.difference(["index_right"]))
    intersect_time = time.perf_counter() - t1

    total_time = read_time + intersect_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result_count = len(result_gdf)
    print(f"  Intersecting features: {result_count:,}")
    print(f"  Read time:      {read_time:.4f} s")
    print(f"  Intersect time: {intersect_time:.4f} s")
    print(f"  Total time:     {total_time:.4f} s")
    print(f"  Peak memory:    {peak_mem / 1024 / 1024:.2f} MB")

    return BenchmarkResult(
        scenario="GPD bbox+sjoin",
        read_time_s=round(read_time, 4),
        intersect_time_s=round(intersect_time, 4),
        total_time_s=round(total_time, 4),
        peak_memory_mb=round(peak_mem / 1024 / 1024, 2),
        result_count=result_count,
        dataset_count=dataset_count,
        aoi_count=aoi_count,
    )


# ---------------------------------------------------------------------------
# Scenario 2 – DuckDB + GeoParquet
# ---------------------------------------------------------------------------
def scenario_duckdb(
    aoi_path: str,
    parquet_path: str,
    aoi_layer: Optional[str] = None,
) -> BenchmarkResult:
    """
    Read the AOI from a Feature Class (via GeoPandas, since DuckDB can't
    read .gdb natively), load the dataset from GeoParquet into DuckDB,
    and perform the spatial intersection using DuckDB's spatial extension.
    """
    import duckdb

    print("\n" + "=" * 60)
    print("SCENARIO 2: DuckDB + GeoParquet")
    print("=" * 60)

    tracemalloc.start()

    # -- Read --
    t0 = time.perf_counter()

    # Read AOI via GeoPandas (DuckDB can't read .gdb directly)
    aoi_gdf = gpd.read_file(aoi_path, layer=aoi_layer)

    # Initialize DuckDB with spatial extension
    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")

    # Read GeoParquet into DuckDB
    # Try st_read first (handles GeoParquet geometry natively), fall back to read_parquet
    parquet_path_safe = parquet_path.replace("\\", "/")
    try:
        con.execute(f"""
            CREATE TABLE dataset AS
            SELECT * FROM st_read('{parquet_path_safe}')
        """)
        print("  Read method: st_read (native spatial)")
    except Exception:
        con.execute(f"""
            CREATE TABLE dataset AS
            SELECT * FROM read_parquet('{parquet_path_safe}')
        """)
        print("  Read method: read_parquet")

    dataset_count = con.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
    read_time = time.perf_counter() - t0

    aoi_count = len(aoi_gdf)
    print(f"  AOI features:     {aoi_count:,}")
    print(f"  Dataset features: {dataset_count:,}")

    # -- Intersect --
    # Convert AOI geometries to WKT and register as a plain DataFrame in DuckDB
    # (DuckDB cannot ingest GeoPandas geometry dtype directly)
    t1 = time.perf_counter()

    aoi_df = aoi_gdf.drop(columns="geometry").copy()
    aoi_df["wkt"] = aoi_gdf.geometry.to_wkt()
    con.execute("CREATE TABLE aoi AS SELECT * FROM aoi_df")

    # Check the geometry column name and type in the parquet dataset
    cols_info = con.execute("DESCRIBE dataset").fetchdf()
    print(f"  Dataset columns: {list(cols_info['column_name'])}")
    
    # Identify the geometry column - common names: geometry, geom, SHAPE, wkb_geometry
    geom_candidates = ["geometry", "geom", "SHAPE", "wkb_geometry", "shape"]
    geom_col = None
    for candidate in geom_candidates:
        if candidate in cols_info["column_name"].values:
            geom_col = candidate
            break
    
    if geom_col is None:
        # Fallback: look for BLOB or GEOMETRY typed columns
        for _, row in cols_info.iterrows():
            if row["column_type"] in ("BLOB", "GEOMETRY", "WKB_BLOB"):
                geom_col = row["column_name"]
                break
    
    if geom_col is None:
        raise ValueError(
            f"Could not identify geometry column. Columns: {list(cols_info['column_name'])} "
            f"Types: {list(cols_info['column_type'])}"
        )
    
    geom_type = cols_info.loc[cols_info["column_name"] == geom_col, "column_type"].values[0]
    print(f"  Geometry column: '{geom_col}' (type: {geom_type})")

    # Build the ST conversion based on the column type
    if geom_type == "GEOMETRY":
        # Already a native GEOMETRY type - use directly
        geom_expr = f"d.\"{geom_col}\""
    else:
        # WKB blob - convert to geometry
        geom_expr = f"ST_GeomFromWKB(d.\"{geom_col}\")"

    # Perform spatial intersection using DuckDB spatial functions.
    query = f"""
        SELECT DISTINCT d.*
        FROM dataset d, aoi a
        WHERE ST_Intersects(
            {geom_expr},
            ST_GeomFromText(a.wkt)
        )
    """
    print(f"  Running intersection query...")
    result = con.execute(query).fetchdf()

    intersect_time = time.perf_counter() - t1

    total_time = read_time + intersect_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result_count = len(result)
    print(f"  Intersecting features: {result_count:,}")
    print(f"  Read time:      {read_time:.4f} s")
    print(f"  Intersect time: {intersect_time:.4f} s")
    print(f"  Total time:     {total_time:.4f} s")
    print(f"  Peak memory:    {peak_mem / 1024 / 1024:.2f} MB")

    con.close()

    return BenchmarkResult(
        scenario="DuckDB",
        read_time_s=round(read_time, 4),
        intersect_time_s=round(intersect_time, 4),
        total_time_s=round(total_time, 4),
        peak_memory_mb=round(peak_mem / 1024 / 1024, 2),
        result_count=result_count,
        dataset_count=dataset_count,
        aoi_count=aoi_count,
    )


# ---------------------------------------------------------------------------
# Scenario 3 – DuckDB Stateless (ST_Read from source file directly)
# ---------------------------------------------------------------------------
def scenario_duckdb_stateless(
    aoi_path: str,
    dataset_path: str,
    aoi_layer: Optional[str] = None,
    dataset_layer: Optional[str] = None,
) -> BenchmarkResult:
    """
    Stateless DuckDB: read the dataset directly from its source file (.gdb, .shp)
    via ST_Read, run the spatial intersection inline — no tables, no indexes,
    no persistent database.
    """
    import duckdb

    print("\n" + "=" * 60)
    print("SCENARIO 3: DuckDB Stateless (ST_Read from source)")
    print("=" * 60)

    tracemalloc.start()

    t0 = time.perf_counter()

    # Read AOI via GeoPandas to get WKT for the query
    aoi_gdf = gpd.read_file(aoi_path, layer=aoi_layer)
    aoi_count = len(aoi_gdf)

    # Build a single WKT from all AOI geometries (union if multiple)
    if aoi_count == 1:
        aoi_wkt = aoi_gdf.geometry.iloc[0].wkt
    else:
        aoi_wkt = aoi_gdf.geometry.union_all().wkt

    # Initialize DuckDB in-memory (stateless — no file)
    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")

    # Build the ST_Read call
    dataset_path_safe = dataset_path.replace("\\", "/")
    st_read_args = [f"'{dataset_path_safe}'"]

    # Detect driver from file extension
    if dataset_path.lower().endswith(".gdb"):
        st_read_args.append("allowed_drivers => ['OpenFileGDB']")
    elif dataset_path.lower().endswith(".shp"):
        st_read_args.append("allowed_drivers => ['ESRI Shapefile']")

    if dataset_layer:
        st_read_args.append(f"layer => '{dataset_layer}'")

    st_read_expr = f"ST_Read({', '.join(st_read_args)})"

    # Detect geometry column name
    cols_info = con.execute(f"DESCRIBE SELECT * FROM {st_read_expr}").fetchdf()
    geom_candidates = ["geom", "geometry", "SHAPE", "shape", "wkb_geometry"]
    geom_col = None
    for candidate in geom_candidates:
        if candidate in cols_info["column_name"].values:
            geom_col = candidate
            break
    if geom_col is None:
        # Fallback: look for GEOMETRY typed columns
        for _, row in cols_info.iterrows():
            if row["column_type"] in ("GEOMETRY", "BLOB", "WKB_BLOB"):
                geom_col = row["column_name"]
                break
    if geom_col is None:
        raise ValueError(
            f"Could not identify geometry column. Columns: {list(cols_info['column_name'])} "
            f"Types: {list(cols_info['column_type'])}"
        )
    print(f"  Geometry column: '{geom_col}'")

    # Get dataset count (this also warms up the read path)
    dataset_count = con.execute(f"SELECT COUNT(*) FROM {st_read_expr}").fetchone()[0]
    read_time = time.perf_counter() - t0

    print(f"  AOI features:     {aoi_count:,}")
    print(f"  Dataset features: {dataset_count:,}")

    # -- Intersect --
    t1 = time.perf_counter()

    query = f"""
        SELECT *
        FROM {st_read_expr}
        WHERE ST_Intersects(
            "{geom_col}",
            ST_GeomFromText('{aoi_wkt}')
        )
    """
    result = con.execute(query).fetchdf()
    intersect_time = time.perf_counter() - t1

    total_time = read_time + intersect_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result_count = len(result)
    print(f"  Intersecting features: {result_count:,}")
    print(f"  Read time:      {read_time:.4f} s")
    print(f"  Intersect time: {intersect_time:.4f} s")
    print(f"  Total time:     {total_time:.4f} s")
    print(f"  Peak memory:    {peak_mem / 1024 / 1024:.2f} MB")

    con.close()

    return BenchmarkResult(
        scenario="DuckDB Stateless",
        read_time_s=round(read_time, 4),
        intersect_time_s=round(intersect_time, 4),
        total_time_s=round(total_time, 4),
        peak_memory_mb=round(peak_mem / 1024 / 1024, 2),
        result_count=result_count,
        dataset_count=dataset_count,
        aoi_count=aoi_count,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a comparison table of all benchmark runs."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    header = (
        f"{'Scenario':<25} {'Read (s)':>10} {'Intersect (s)':>14} "
        f"{'Total (s)':>10} {'Memory (MB)':>12} {'Results':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.scenario:<25} {r.read_time_s:>10.4f} {r.intersect_time_s:>14.4f} "
            f"{r.total_time_s:>10.4f} {r.peak_memory_mb:>12.2f} {r.result_count:>10,}"
        )

    # Pairwise speedup comparisons
    if len(results) >= 2:
        fastest = min(results, key=lambda r: r.total_time_s)
        leanest = min(results, key=lambda r: r.peak_memory_mb)
        print(f"\n  → Fastest: {fastest.scenario} ({fastest.total_time_s:.4f}s)")
        print(f"  → Leanest: {leanest.scenario} ({leanest.peak_memory_mb:.2f} MB)")
        for r in results:
            if r is not fastest and fastest.total_time_s > 0:
                ratio = r.total_time_s / fastest.total_time_s
                print(f"  → {fastest.scenario} is {ratio:.2f}x faster than {r.scenario}")


def main():
    # -----------------------------------------------------------------------
    # Dataset paths
    # -----------------------------------------------------------------------
    AOI_PATH = r"W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework\test_parquet\test_aoi_cariboo.shp"

    DATASET_PATH = (
        r"W:\srm\wml\Workarea\!Cariboo_Data_Warehouse\fish_wildlife_and_plant_species"
        r"\fisher_2021\Cariboo_region_fisher_habitat_retention_2021.gdb"
    )
    DATASET_LAYER = "Cariboo_region_fisher_habitat_retention"

    PARQUET_PATH = (
        r"W:\srm\gss\sandbox\mlabiadh\workspace\20251203_ast_rework"
        r"\test_parquet\Cariboo_region_fisher_habitat_retention.parquet"
    )

    RUNS = 1  # Number of runs to average
    # -----------------------------------------------------------------------

    print("Configuration:")
    print(f"  AOI:      {AOI_PATH}")
    print(f"  Dataset:  {DATASET_PATH} (layer={DATASET_LAYER})")
    print(f"  Parquet:  {PARQUET_PATH}")
    print(f"  Runs:     {RUNS}")

    all_results = []

    for run_num in range(1, RUNS + 1):
        if RUNS > 1:
            print(f"\n{'#' * 60}")
            print(f"  RUN {run_num} / {RUNS}")
            print(f"{'#' * 60}")

        #r1 = scenario_geopandas(AOI_PATH, DATASET_PATH, dataset_layer=DATASET_LAYER)
        r2 = scenario_geopandas_mask(AOI_PATH, DATASET_PATH, dataset_layer=DATASET_LAYER)
        r3 = scenario_geopandas_bbox(AOI_PATH, DATASET_PATH, dataset_layer=DATASET_LAYER)
        r4 = scenario_duckdb(AOI_PATH, PARQUET_PATH)
        r5 = scenario_duckdb_stateless(
            AOI_PATH, DATASET_PATH, dataset_layer=DATASET_LAYER
        )
        all_results.append((r2, r3, r4, r5))

    # Average results across runs
    scenario_count = 5
    scenario_names = [
        "GPD mask+sjoin (avg)",
        "GPD bbox+sjoin (avg)",
        "DuckDB+Parquet (avg)",
        "DuckDB Stateless (avg)",
    ]

    if RUNS > 1:
        averages = []
        for i in range(scenario_count):
            averages.append(BenchmarkResult(
                scenario=scenario_names[i],
                read_time_s=round(sum(r[i].read_time_s for r in all_results) / RUNS, 4),
                intersect_time_s=round(sum(r[i].intersect_time_s for r in all_results) / RUNS, 4),
                total_time_s=round(sum(r[i].total_time_s for r in all_results) / RUNS, 4),
                peak_memory_mb=round(sum(r[i].peak_memory_mb for r in all_results) / RUNS, 2),
                result_count=all_results[-1][i].result_count,
                dataset_count=all_results[-1][i].dataset_count,
                aoi_count=all_results[-1][i].aoi_count,
            ))
        print_summary(averages)
    else:
        print_summary(list(all_results[0]))


if __name__ == "__main__":
    main()
