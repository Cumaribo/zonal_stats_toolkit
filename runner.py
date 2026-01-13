from __future__ import annotations
from datetime import datetime
from pathlib import Path
import argparse
import collections
import configparser
import csv
import glob
import logging
import os
import shutil
import tempfile
import time

from ecoshard import taskgraph, geoprocessing
from osgeo import gdal, ogr, osr
import fiona
import numpy as np
import shapely

logger = logging.getLogger(__name__)

_LOGGING_PERIOD = 10.0
VALID_OPERATIONS = {
    "avg",
    "stdev",
    "min",
    "max",
    "sum",
    "total_count",
    "valid_count",
    "median",
    "p5",
    "p10",
    "p25",
    "p75",
    "p90",
    "p95",
}


def _make_logger_callback(message):
    """Build a timed logger callback that prints ``message`` replaced.

    Args:
        message (string): a string that expects 2 placement %% variables,
            first for % complete from ``df_complete``, second from
            ``p_progress_arg[0]``.

    Return:
        Function with signature:
            logger_callback(df_complete, psz_message, p_progress_arg)

    """

    def logger_callback(df_complete, _, p_progress_arg):
        """Argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if (current_time - logger_callback.last_time) > 5.0 or (
                df_complete == 1.0 and logger_callback.total_time >= 5.0
            ):
                # In some multiprocess applications I was encountering a
                # ``p_progress_arg`` of None. This is unexpected and I suspect
                # was an issue for some kind of GDAL race condition. So I'm
                # guarding against it here and reporting an appropriate log
                # if it occurs.
                if p_progress_arg:
                    logger.info(message, df_complete * 100, p_progress_arg[0])
                else:
                    logger.info(message, df_complete * 100, "")
                logger_callback.last_time = current_time
                logger_callback.total_time += current_time
        except AttributeError:
            logger_callback.last_time = time.time()
            logger_callback.total_time = 0.0
        except Exception:
            logger.exception(
                "Unhandled error occurred while logging "
                "progress.  df_complete: %s, p_progress_arg: %s",
                df_complete,
                p_progress_arg,
            )

    return logger_callback


def parse_and_validate_config(cfg_path: Path) -> dict:
    """Parse and validate a zonal-stats runner INI configuration file.

    The configuration file is expected to contain a `[project]` section and one or
    more `[job:<tag>]` sections. This function validates required keys, enforces
    naming constraints, checks file existence, verifies vector layer/field
    presence, and validates the `operations` list against `VALID_OPERATIONS`.

    Validation rules:
      1) `[project].name` must equal the configuration file stem.
      2) `[project].log_level` must be a valid `logging` level name.
      3) Job section tags (`job:<tag>`) must be unique.
      4) For each job, `agg_vector` must exist; `base_raster` must exist if set.
      5) `agg_layer` must exist in `agg_vector`, and `agg_field` must exist in
         that layer schema.
      6) `operations` must be present and all entries must be in
         `VALID_OPERATIONS`.

    The returned dictionary contains:
      - `project`: global configuration values.
      - `job_list`: a list of per-job dictionaries.

    Args:
        cfg_path: Path to the INI configuration file.

    Returns:
        A dictionary with keys:
          - `project`: Dict containing `name`, `global_work_dir`, and `log_level`.
          - `job_list`: List of dicts describing each job. Each job dict includes
            `tag`, `agg_vector`, `agg_layer`, `agg_field`, `base_raster`,
            `workdir`, `output_csv`, and `operations`.

    Raises:
        ValueError: If required sections/keys are missing, if values are invalid,
            if job tags are duplicated, if a layer/field is missing, or if any
            operation is not recognized.
        FileNotFoundError: If `agg_vector` does not exist, or if `base_raster` is
            provided but does not exist.
    """
    stem = cfg_path.stem

    config = configparser.ConfigParser(interpolation=None)
    config.read(cfg_path)
    cfg_dir = cfg_path.resolve().parent

    if "project" not in config:
        raise ValueError("Missing [project] section")

    project_name = config["project"].get("name", "").strip()
    if project_name != stem:
        raise ValueError(
            f"[project].name must equal config stem: expected {stem}, got {project_name}"
        )

    log_level_str = config["project"].get("log_level", "INFO").strip().upper()
    try:
        _ = getattr(logging, log_level_str)
    except AttributeError:
        raise ValueError(f"Invalid log_level: {log_level_str}")

    global_work_dir = Path(config["project"]["global_work_dir"].strip())
    global_output_dir = Path(config["project"]["global_output_dir"].strip())

    job_tags = []
    jobs_sections = []
    for section in config.sections():
        if section.startswith("job:"):
            tag = section.split(":", 1)[1].strip()
            if not tag:
                raise ValueError(f"Invalid job section name: [{section}]")
            job_tags.append(tag)
            jobs_sections.append((tag, config[section]))

    if len(job_tags) != len(set(job_tags)):
        seen = set()
        dups = []
        for t in job_tags:
            if t in seen:
                dups.append(t)
            seen.add(t)
        raise ValueError(f"Duplicate job tags found: {sorted(set(dups))}")

    job_list = []
    for tag, job in jobs_sections:
        agg_vector_raw = job.get("agg_vector", "").strip()
        if not agg_vector_raw:
            raise ValueError(f"[job:{tag}] missing agg_vector")

        agg_vector = Path(agg_vector_raw)
        if not agg_vector.is_absolute():
            agg_vector = cfg_dir / agg_vector

        if not agg_vector.exists():
            raise FileNotFoundError(
                f"[job:{tag}] agg_vector not found: {agg_vector}"
            )

        if not agg_vector.is_file():
            raise IsADirectoryError(f"[job:{tag}] agg_vector is not a file: {agg_vector}")

        if not os.access(agg_vector, os.R_OK):
            stat_info = agg_vector.stat()
            raise PermissionError(
                f"[job:{tag}] Permission denied reading agg_vector: {agg_vector}. "
                f"Current user: {os.getuid()}, File owner: {stat_info.st_uid}, Mode: {oct(stat_info.st_mode)}. "
                "Check file permissions."
            )

        base_raster_pattern = job.get("base_raster_pattern", "").strip()
        if base_raster_pattern in [None, ""]:
            raise FileNotFoundError(
                f"[job:{tag}] base_raster_pattern tag not found"
            )

        base_raster_path_list = []
        for pattern in base_raster_pattern.split(","):
            pattern = pattern.strip()
            if not pattern:
                continue
            if not Path(pattern).is_absolute():
                full_pattern = str(cfg_dir / pattern)
            else:
                full_pattern = pattern
            base_raster_path_list.extend(
                [Path(p) for p in glob.glob(full_pattern)]
            )

        if not base_raster_path_list:
            raise FileNotFoundError(
                f"[job:{tag}] no files found at {base_raster_pattern}"
            )

        agg_field = job.get("agg_field", "").strip()
        if not agg_field:
            raise ValueError(f"[job:{tag}] missing agg_field")

        ops_raw = job.get("operations", "").strip()
        if not ops_raw:
            raise ValueError(f"[job:{tag}] missing operations")
        operations = [
            o.strip().lower() for o in ops_raw.split(",") if o.strip()
        ]
        if not operations:
            raise ValueError(f"[job:{tag}] operations is empty")

        invalid_ops = sorted(set(operations) - VALID_OPERATIONS)
        if invalid_ops:
            raise ValueError(
                f"[job:{tag}] invalid operations: {invalid_ops}. "
                f"Valid operations: {sorted(VALID_OPERATIONS)}"
            )

        layers = fiona.listlayers(str(agg_vector))

        agg_layer = job.get("agg_layer", "").strip()
        if agg_layer is None or not str(agg_layer).strip():
            if not layers:
                raise ValueError(f"[job:{tag}] no layers found in {agg_vector}")
            agg_layer = layers[0]

        if agg_layer not in layers:
            raise ValueError(
                f'[job:{tag}] agg_layer "{agg_layer}" not found in {agg_vector}. '
                f"Available layers: {layers}"
            )

        with fiona.open(str(agg_vector), layer=agg_layer) as src:
            props = src.schema.get("properties", {})
            if agg_field not in props:
                raise ValueError(
                    f'[job:{tag}] agg_field "{agg_field}" not found in layer "{agg_layer}" of {agg_vector}. '
                    f"Available fields: {sorted(props.keys())}"
                )
        outdir = global_output_dir
        workdir = global_work_dir / Path(tag)
        try:
            outdir.mkdir(parents=True, exist_ok=True)
            workdir.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            failed_path = Path(e.filename) if e.filename else workdir
            # Determine which path to inspect for permissions
            if failed_path.exists():
                target_path = failed_path
                msg_prefix = "Permission denied accessing existing directory"
            else:
                target_path = failed_path.resolve().parent
                msg_prefix = "Permission denied creating directory in"

            try:
                stat_info = target_path.stat()
                diag_info = (
                    f"Current user: {os.getuid()}, "
                    f"Target owner: {stat_info.st_uid}, "
                    f"Mode: {oct(stat_info.st_mode)}"
                )
            except Exception:
                diag_info = "Could not stat target path."

            raise PermissionError(
                f"{msg_prefix}: {failed_path}. {diag_info}. "
                "Check directory permissions."
            ) from e

        job_list.append(
            {
                "tag": tag,
                "agg_vector": agg_vector,
                "agg_layer": agg_layer,
                "agg_field": agg_field,
                "base_raster_path_list": base_raster_path_list,
                "operations": operations,
                "row_col_order": job["row_col_order"],
                "workdir": workdir,
                "output_csv": outdir / f"{tag}.csv",
            }
        )

    return {
        "project": {
            "name": project_name,
            "global_work_dir": global_work_dir,
            "log_level": log_level_str,
        },
        "job_list": job_list,
    }


def fast_zonal_statistics(
    base_raster_path_band,
    aggregate_vector_path,
    aggregate_vector_field,
    aggregate_layer_name=None,
    ignore_nodata=True,
    polygons_might_overlap=True,
    working_dir=None,
    clean_working_dir=True,
    percentile_list=None,
):
    logger.info(
        "fast_zonal_statistics start | raster=%s band=%s | vector=%s layer=%s field=%s | ignore_nodata=%s overlap=%s | working_dir=%s clean=%s | percentiles=%s",
        base_raster_path_band[0],
        base_raster_path_band[1],
        str(aggregate_vector_path),
        aggregate_layer_name,
        aggregate_vector_field,
        ignore_nodata,
        polygons_might_overlap,
        working_dir,
        clean_working_dir,
        percentile_list,
    )

    percentile_list = [] if percentile_list is None else list(percentile_list)
    percentile_list = sorted(set(float(p) for p in percentile_list))
    percentile_keys = [
        f"p{int(p) if float(p).is_integer() else p}" for p in percentile_list
    ]

    raster_info = geoprocessing.get_raster_info(base_raster_path_band[0])
    raster_nodata = raster_info["nodata"][base_raster_path_band[1] - 1]
    pixel_width = abs(raster_info["pixel_size"][0])
    tolerance = pixel_width * 0.5

    logger.info(
        "raster loaded | nodata=%s | pixel_size=%s | bbox=%s",
        raster_nodata,
        raster_info["pixel_size"],
        raster_info["bounding_box"],
    )

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_info["projection_wkt"])
    raster_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    logger.info("opening source vector: %s", str(aggregate_vector_path))
    src_vector = gdal.OpenEx(str(aggregate_vector_path), gdal.OF_VECTOR)
    if src_vector is None:
        raise RuntimeError(
            "Could not open aggregate vector at %s" % str(aggregate_vector_path)
        )

    if aggregate_layer_name is not None:
        logger.info("selecting vector layer by name: %s", aggregate_layer_name)
        src_layer = src_vector.GetLayerByName(aggregate_layer_name)
    else:
        logger.info("selecting default vector layer")
        src_layer = src_vector.GetLayer()

    if src_layer is None:
        raise RuntimeError(
            "Could not open layer %s on %s"
            % (aggregate_layer_name, str(aggregate_vector_path))
        )

    src_srs = src_layer.GetSpatialRef()
    needs_reproject = True
    if src_srs is not None:
        src_srs = src_srs.Clone()
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        needs_reproject = not src_srs.IsSame(raster_srs)
        logger.info("vector SRS detected | needs_reproject=%s", needs_reproject)
    else:
        logger.info(
            "vector SRS missing/unknown | forcing reprojection to raster SRS"
        )

    if working_dir and not os.access(working_dir, os.W_OK):
        stat_info = os.stat(working_dir)
        raise PermissionError(
            f"Permission denied writing to working_dir: {working_dir}. "
            f"Current user: {os.getuid()}, Directory owner: {stat_info.st_uid}, Mode: {oct(stat_info.st_mode)}. "
            "Check directory permissions."
        )

    temp_working_dir = tempfile.mkdtemp(dir=working_dir)
    projected_vector_path = os.path.join(
        temp_working_dir, "projected_vector.gpkg"
    )
    logger.info("created temp working dir: %s", temp_working_dir)

    translate_kwargs = {
        "simplifyTolerance": tolerance,
        "format": "GPKG",
    }
    if needs_reproject:
        translate_kwargs["dstSRS"] = raster_info["projection_wkt"]

    logger.info(
        "vector translate start | output=%s | simplifyTolerance=%s | reproject=%s",
        projected_vector_path,
        tolerance,
        needs_reproject,
    )
    gdal.VectorTranslate(
        projected_vector_path, str(aggregate_vector_path), **translate_kwargs
    )
    logger.info("vector translate done | output=%s", projected_vector_path)

    src_layer = None
    src_vector = None

    logger.info("opening projected vector: %s", projected_vector_path)
    aggregate_vector = gdal.OpenEx(projected_vector_path, gdal.OF_VECTOR)
    if aggregate_vector is None:
        raise RuntimeError(
            "Could not open aggregate vector at %s" % projected_vector_path
        )

    if aggregate_layer_name is not None:
        aggregate_layer = aggregate_vector.GetLayerByName(aggregate_layer_name)
    else:
        aggregate_layer = aggregate_vector.GetLayer()

    if aggregate_layer is None:
        raise RuntimeError(
            "Could not open layer %s on %s"
            % (aggregate_layer_name, projected_vector_path)
        )

    logger.info(
        "scanning vector for grouping field values: %s", aggregate_vector_field
    )
    aggregate_layer_fid_set = set()
    fid_to_group_value = {}
    unique_group_values = set()
    aggregate_layer.ResetReading()
    for feat in aggregate_layer:
        fid = feat.GetFID()
        group_value = feat.GetField(aggregate_vector_field)
        aggregate_layer_fid_set.add(fid)
        fid_to_group_value[fid] = group_value
        unique_group_values.add(group_value)
    aggregate_layer.ResetReading()
    logger.info(
        "vector scan done | features=%d | unique %s=%d",
        len(aggregate_layer_fid_set),
        aggregate_vector_field,
        len(unique_group_values),
    )

    raster_bbox = raster_info["bounding_box"]
    vec_extent = aggregate_layer.GetExtent()
    logger.info(
        "extent check | raster_bbox=%s | vector_extent=%s",
        raster_bbox,
        vec_extent,
    )

    v_minx, v_maxx, v_miny, v_maxy = vec_extent
    r_minx, r_miny, r_maxx, r_maxy = raster_bbox

    no_intersection = (
        v_maxx < r_minx or v_minx > r_maxx or v_maxy < r_miny or v_miny > r_maxy
    )

    if no_intersection:
        logger.error(
            f"aggregate vector {aggregate_vector_path} does not intersect with "
            f"the raster {base_raster_path_band[0]}: vector extent {vec_extent} vs raster bounding box {raster_bbox}"
        )
        group_stats = collections.defaultdict(
            lambda: {
                "min": None,
                "max": None,
                "count": 0,
                "nodata_count": 0,
                "valid_count": 0,
                "sum": 0.0,
                "stdev": None,
                **{k: None for k in percentile_keys},
            }
        )
        for group_value in unique_group_values:
            _ = group_stats[group_value]
        logger.info(
            "returning empty stats for %d groups (no intersection)",
            len(unique_group_values),
        )
        if clean_working_dir:
            logger.info("cleaning temp working dir: %s", temp_working_dir)
            shutil.rmtree(temp_working_dir)
        return dict(group_stats)

    clipped_raster_path = base_raster_path_band[0]
    logger.info("opening raster for read: %s", clipped_raster_path)
    clipped_raster = gdal.OpenEx(clipped_raster_path, gdal.OF_RASTER)
    clipped_band = clipped_raster.GetRasterBand(base_raster_path_band[1])
    logger.info(
        "raster opened | size=%dx%d | band=%d",
        clipped_band.XSize,
        clipped_band.YSize,
        base_raster_path_band[1],
    )

    local_aggregate_field_name = "original_fid"
    rasterize_layer_args = {
        "options": [
            "ALL_TOUCHED=FALSE",
            "ATTRIBUTE=%s" % local_aggregate_field_name,
        ]
    }

    driver = ogr.GetDriverByName("MEMORY")
    disjoint_vector = driver.CreateDataSource("disjoint_vector")
    spat_ref = aggregate_layer.GetSpatialRef()

    logger.info(
        "building disjoint polygon sets | polygons_might_overlap=%s",
        polygons_might_overlap,
    )
    if polygons_might_overlap:
        disjoint_fid_sets = geoprocessing.calculate_disjoint_polygon_set(
            projected_vector_path, bounding_box=raster_bbox
        )
    else:
        disjoint_fid_sets = [aggregate_layer_fid_set]
    logger.info(
        "disjoint sets ready | sets=%d | total_features=%d",
        len(disjoint_fid_sets),
        len(aggregate_layer_fid_set),
    )

    aggregate_stats = collections.defaultdict(
        lambda: {
            "min": None,
            "max": None,
            "count": 0,
            "nodata_count": 0,
            "sum": 0.0,
            "sumsq": 0.0,
        }
    )

    fid_value_chunks = None
    if percentile_list:
        fid_value_chunks = collections.defaultdict(list)
        logger.info(
            "percentiles enabled | collecting values in memory | percentiles=%s",
            percentile_list,
        )

    last_time = time.time()
    logger.info("processing %d disjoint polygon sets", len(disjoint_fid_sets))
    for set_index, disjoint_fid_set in enumerate(disjoint_fid_sets):
        logger.info(
            "set %d/%d start | polygons_in_set=%d",
            set_index + 1,
            len(disjoint_fid_sets),
            len(disjoint_fid_set),
        )

        last_time = _invoke_timed_callback(
            last_time,
            lambda: logger.info(
                "zonal stats approximately %.1f%% complete on %s",
                100.0 * float(set_index + 1) / len(disjoint_fid_sets),
                os.path.basename(projected_vector_path),
            ),
            _LOGGING_PERIOD,
        )

        agg_fid_raster_path = os.path.join(
            temp_working_dir, f"agg_fid_{set_index}.tif"
        )
        agg_fid_nodata = -1
        logger.info("creating agg fid raster: %s", agg_fid_raster_path)
        geoprocessing.new_raster_from_base(
            clipped_raster_path,
            agg_fid_raster_path,
            gdal.GDT_Int32,
            [agg_fid_nodata],
        )

        agg_fid_offset_list = list(
            geoprocessing.iterblocks((agg_fid_raster_path, 1), offset_only=True)
        )
        logger.info("iterblocks prepared | blocks=%d", len(agg_fid_offset_list))

        agg_fid_raster = gdal.OpenEx(
            agg_fid_raster_path, gdal.GA_Update | gdal.OF_RASTER
        )
        agg_fid_band = agg_fid_raster.GetRasterBand(1)

        disjoint_layer = disjoint_vector.CreateLayer(
            "disjoint_vector", spat_ref, ogr.wkbPolygon
        )
        disjoint_layer.CreateField(
            ogr.FieldDefn(local_aggregate_field_name, ogr.OFTInteger)
        )
        disjoint_layer_defn = disjoint_layer.GetLayerDefn()

        logger.info("populating disjoint layer features (transaction start)")
        disjoint_layer.StartTransaction()
        for index, feature_fid in enumerate(disjoint_fid_set):
            last_time = _invoke_timed_callback(
                last_time,
                lambda i=index: logger.info(
                    "polygon set %d/%d approximately %.1f%% processed on %s",
                    set_index + 1,
                    len(disjoint_fid_sets),
                    100.0 * float(i + 1) / len(disjoint_fid_set),
                    os.path.basename(projected_vector_path),
                ),
                _LOGGING_PERIOD,
            )
            agg_feat = aggregate_layer.GetFeature(feature_fid)
            agg_geom_ref = agg_feat.GetGeometryRef()
            disjoint_feat = ogr.Feature(disjoint_layer_defn)
            disjoint_feat.SetGeometry(agg_geom_ref.Clone())
            agg_geom_ref = None
            disjoint_feat.SetField(local_aggregate_field_name, feature_fid)
            disjoint_layer.CreateFeature(disjoint_feat)

        agg_feat = None
        disjoint_layer.CommitTransaction()
        logger.info(
            "populating disjoint layer features done (transaction commit)"
        )

        rasterize_callback = _make_logger_callback(
            "rasterizing polygon "
            + str(set_index + 1)
            + " of "
            + str(len(disjoint_fid_set))
            + " set %.1f%% complete %s"
        )

        logger.info(
            "rasterize start | set %d/%d", set_index + 1, len(disjoint_fid_sets)
        )
        gdal.RasterizeLayer(
            agg_fid_raster,
            [1],
            disjoint_layer,
            callback=rasterize_callback,
            **rasterize_layer_args,
        )
        agg_fid_raster.FlushCache()
        logger.info(
            "rasterize done | set %d/%d", set_index + 1, len(disjoint_fid_sets)
        )

        disjoint_layer = None
        disjoint_vector.DeleteLayer(0)

        logger.info(
            "gathering stats from raster blocks | set %d/%d",
            set_index + 1,
            len(disjoint_fid_sets),
        )
        block_last_time = time.time()
        for block_index, agg_fid_offset in enumerate(agg_fid_offset_list):
            block_last_time = _invoke_timed_callback(
                block_last_time,
                lambda bi=block_index: logger.info(
                    "block processing | set %d/%d | %.1f%% (%d/%d blocks)",
                    set_index + 1,
                    len(disjoint_fid_sets),
                    100.0 * float(bi + 1) / len(agg_fid_offset_list),
                    bi + 1,
                    len(agg_fid_offset_list),
                ),
                _LOGGING_PERIOD,
            )

            agg_fid_block = agg_fid_band.ReadAsArray(**agg_fid_offset)
            clipped_block = clipped_band.ReadAsArray(**agg_fid_offset)
            valid_mask = agg_fid_block != agg_fid_nodata
            valid_agg_fids = agg_fid_block[valid_mask]
            valid_clipped = clipped_block[valid_mask]

            for agg_fid in np.unique(valid_agg_fids):
                masked_clipped_block = valid_clipped[valid_agg_fids == agg_fid]
                total_count = masked_clipped_block.size

                if raster_nodata is not None:
                    clipped_nodata_mask = np.isclose(
                        masked_clipped_block, raster_nodata
                    )
                else:
                    clipped_nodata_mask = np.zeros(
                        masked_clipped_block.shape, dtype=bool
                    )

                if np.issubdtype(masked_clipped_block.dtype, np.floating):
                    clipped_nodata_mask |= np.isnan(masked_clipped_block)

                nodata_count = np.count_nonzero(clipped_nodata_mask)
                aggregate_stats[agg_fid]["count"] += total_count
                aggregate_stats[agg_fid]["nodata_count"] += nodata_count

                if ignore_nodata:
                    masked_clipped_block = masked_clipped_block[
                        ~clipped_nodata_mask
                    ]
                if masked_clipped_block.size == 0:
                    continue

                if fid_value_chunks is not None:
                    fid_value_chunks[agg_fid].append(
                        masked_clipped_block.astype(np.float32, copy=False)
                    )

                if aggregate_stats[agg_fid]["min"] is None:
                    aggregate_stats[agg_fid]["min"] = masked_clipped_block[0]
                    aggregate_stats[agg_fid]["max"] = masked_clipped_block[0]

                aggregate_stats[agg_fid]["min"] = min(
                    np.min(masked_clipped_block),
                    aggregate_stats[agg_fid]["min"],
                )
                aggregate_stats[agg_fid]["max"] = max(
                    np.max(masked_clipped_block),
                    aggregate_stats[agg_fid]["max"],
                )
                aggregate_stats[agg_fid]["sum"] += np.sum(masked_clipped_block)
                aggregate_stats[agg_fid]["sumsq"] += np.sum(
                    masked_clipped_block * masked_clipped_block,
                    dtype=np.float64,
                )

        logger.info(
            "set %d/%d done | fids_with_any_stats_so_far=%d",
            set_index + 1,
            len(disjoint_fid_sets),
            len(aggregate_stats),
        )

        agg_fid_band = None
        agg_fid_raster = None

    unset_fids = aggregate_layer_fid_set.difference(aggregate_stats)
    logger.info("unset fid pass start | unset_fids=%d", len(unset_fids))

    clipped_gt = np.array(clipped_raster.GetGeoTransform(), dtype=np.float32)
    for unset_fid in unset_fids:
        unset_feat = aggregate_layer.GetFeature(unset_fid)
        unset_geom_ref = unset_feat.GetGeometryRef()
        if unset_geom_ref is None:
            logger.warning(
                "no geometry in %s FID: %s", projected_vector_path, unset_fid
            )
            continue

        shapely_geom = shapely.wkb.loads(bytes(unset_geom_ref.ExportToWkb()))
        try:
            shapely_geom_list = list(shapely_geom)
        except TypeError:
            shapely_geom_list = [shapely_geom]
        unset_geom_ref = None

        for shapely_geom in shapely_geom_list:
            single_geom = ogr.CreateGeometryFromWkt(shapely_geom.wkt)
            unset_geom_envelope = list(single_geom.GetEnvelope())
            single_geom = None

            if clipped_gt[1] < 0:
                unset_geom_envelope[0], unset_geom_envelope[1] = (
                    unset_geom_envelope[1],
                    unset_geom_envelope[0],
                )
            if clipped_gt[5] < 0:
                unset_geom_envelope[2], unset_geom_envelope[3] = (
                    unset_geom_envelope[3],
                    unset_geom_envelope[2],
                )

            xoff = int((unset_geom_envelope[0] - clipped_gt[0]) / clipped_gt[1])
            yoff = int((unset_geom_envelope[2] - clipped_gt[3]) / clipped_gt[5])
            win_xsize = (
                int(
                    np.ceil(
                        (unset_geom_envelope[1] - clipped_gt[0]) / clipped_gt[1]
                    )
                )
                - xoff
            )
            win_ysize = (
                int(
                    np.ceil(
                        (unset_geom_envelope[3] - clipped_gt[3]) / clipped_gt[5]
                    )
                )
                - yoff
            )

            if xoff < 0:
                win_xsize += xoff
                xoff = 0
            if yoff < 0:
                win_ysize += yoff
                yoff = 0
            if xoff + win_xsize > clipped_band.XSize:
                win_xsize = clipped_band.XSize - xoff
            if yoff + win_ysize > clipped_band.YSize:
                win_ysize = clipped_band.YSize - yoff
            if win_xsize <= 0 or win_ysize <= 0:
                continue

            unset_fid_block = clipped_band.ReadAsArray(
                xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize
            )

            if raster_nodata is not None:
                unset_fid_nodata_mask = np.isclose(
                    unset_fid_block, raster_nodata
                )
            else:
                unset_fid_nodata_mask = np.zeros(
                    unset_fid_block.shape, dtype=bool
                )

            if np.issubdtype(unset_fid_block.dtype, np.floating):
                unset_fid_nodata_mask |= np.isnan(unset_fid_block)

            if ignore_nodata:
                valid_unset_fid_block = unset_fid_block[~unset_fid_nodata_mask]
            else:
                valid_unset_fid_block = unset_fid_block

            aggregate_stats[unset_fid]["count"] = unset_fid_block.size
            aggregate_stats[unset_fid]["nodata_count"] = np.count_nonzero(
                unset_fid_nodata_mask
            )

            if valid_unset_fid_block.size == 0:
                aggregate_stats[unset_fid]["min"] = 0.0
                aggregate_stats[unset_fid]["max"] = 0.0
                aggregate_stats[unset_fid]["sum"] = 0.0
                aggregate_stats[unset_fid]["sumsq"] = 0.0
            else:
                aggregate_stats[unset_fid]["min"] = np.min(
                    valid_unset_fid_block
                )
                aggregate_stats[unset_fid]["max"] = np.max(
                    valid_unset_fid_block
                )
                aggregate_stats[unset_fid]["sum"] = np.sum(
                    valid_unset_fid_block
                )
                aggregate_stats[unset_fid]["sumsq"] = np.sum(
                    valid_unset_fid_block * valid_unset_fid_block,
                    dtype=np.float64,
                )

            if fid_value_chunks is not None and valid_unset_fid_block.size:
                fid_value_chunks[unset_fid].append(
                    valid_unset_fid_block.astype(np.float32, copy=False)
                )

    unset_fids = aggregate_layer_fid_set.difference(aggregate_stats)
    for fid in unset_fids:
        _ = aggregate_stats[fid]

    logger.info(
        "unset fid pass done | remaining_unset=%d | total_fids=%d",
        len(unset_fids),
        len(aggregate_layer_fid_set),
    )

    if fid_value_chunks is not None:
        logger.info(
            "computing per-fid percentiles | percentiles=%s", percentile_list
        )
        fid_percentiles = {}
        for fid, chunks in fid_value_chunks.items():
            if not chunks:
                fid_percentiles[fid] = [None] * len(percentile_list)
                continue
            vals = np.concatenate(chunks)
            fid_percentiles[fid] = np.percentile(vals, percentile_list).tolist()
        logger.info(
            "computing per-fid percentiles done | fids=%d", len(fid_percentiles)
        )
    else:
        fid_percentiles = None

    spat_ref = None
    clipped_band = None
    clipped_raster = None
    disjoint_layer = None
    disjoint_vector = None
    aggregate_layer = None
    aggregate_vector = None

    logger.info("grouping fid stats -> %s values", aggregate_vector_field)
    grouped_stats = collections.defaultdict(
        lambda: {
            "min": None,
            "max": None,
            "count": 0,
            "nodata_count": 0,
            "valid_count": 0,
            "sum": 0.0,
            "sumsq": 0.0,
            "stdev": None,
            "avg": None,
            **{k: None for k in percentile_keys},
        }
    )

    group_value_chunks = None
    if percentile_list:
        group_value_chunks = collections.defaultdict(list)

    for fid in aggregate_layer_fid_set:
        group_value = fid_to_group_value[fid]
        fid_stats = aggregate_stats[fid]
        g = grouped_stats[group_value]

        g["count"] += fid_stats["count"]
        g["nodata_count"] += fid_stats["nodata_count"]
        g["sum"] += fid_stats["sum"]
        g["sumsq"] += fid_stats["sumsq"]

        fid_valid_count = fid_stats["count"] - fid_stats["nodata_count"]
        if fid_valid_count > 0:
            if g["min"] is None:
                g["min"] = fid_stats["min"]
                g["max"] = fid_stats["max"]
            else:
                g["min"] = min(g["min"], fid_stats["min"])
                g["max"] = max(g["max"], fid_stats["max"])

        if group_value_chunks is not None:
            chunks = fid_value_chunks.get(fid)
            if chunks:
                group_value_chunks[group_value].extend(chunks)

    if group_value_chunks is not None:
        logger.info(
            "computing grouped percentiles | groups=%d | percentiles=%s",
            len(group_value_chunks),
            percentile_list,
        )
        for group_value, chunks in group_value_chunks.items():
            if not chunks:
                continue
            vals = np.concatenate(chunks)
            pct_vals = np.percentile(vals, percentile_list)
            for k, v in zip(percentile_keys, pct_vals.tolist()):
                grouped_stats[group_value][k] = v
        logger.info("computing grouped percentiles done")

    for group_value, g in grouped_stats.items():
        valid_count = g["count"] - g["nodata_count"]
        g["valid_count"] = valid_count
        if valid_count > 0:
            mean = g["sum"] / valid_count
            g["avg"] = mean
            var = g["sumsq"] / valid_count - mean * mean
            if var < 0:
                var = 0.0
            g["stdev"] = float(np.sqrt(var))
        else:
            g["stdev"] = None
            g["avg"] = None
        del g["sumsq"]

    logger.info("grouping done | groups=%d", len(grouped_stats))

    if clean_working_dir:
        logger.info("cleaning temp working dir: %s", temp_working_dir)
        shutil.rmtree(temp_working_dir)

    logger.info("fast_zonal_statistics done")
    return dict(grouped_stats)


def run_zonal_stats_job(
    base_raster_path_list: list[Path],
    agg_vector: Path,
    agg_layer: str,
    agg_field: str,
    operations: list[str],
    output_csv: Path,
    workdir: Path,
    tag: str,
    row_col_order: str,
):
    raster_stems = []
    raster_stats_by_stem = {}
    all_groups = set()

    percentile_list = [
        float(op[1:])
        for op in operations
        if op.startswith("p") and op[1:].replace(".", "", 1).isdigit()
    ]

    op_to_key = {
        "avg": "avg",
        "stdev": "stdev",
        "min": "min",
        "max": "max",
        "sum": "sum",
        "total_count": "count",
        "valid_count": "valid_count",
        "nodata_count": "nodata_count",
    }
    stat_fields = []
    for op in operations:
        if op in op_to_key:
            stat_fields.append(op_to_key[op])
        elif op.startswith("p") and op[1:].replace(".", "", 1).isdigit():
            val = float(op[1:])
            key = f"p{int(val) if val.is_integer() else val}"
            stat_fields.append(key)

    for raster_path in base_raster_path_list:
        stem = raster_path.stem
        raster_stems.append(stem)
        stats = fast_zonal_statistics(
            (str(raster_path), 1),
            str(agg_vector),
            agg_field,
            aggregate_layer_name=agg_layer,
            ignore_nodata=True,
            polygons_might_overlap=False,
            working_dir=str(workdir),
            clean_working_dir=False,
            percentile_list=percentile_list,
        )
        raster_stats_by_stem[stem] = stats
        all_groups.update(stats.keys())

    parts = [p.strip() for p in row_col_order.split(",") if p.strip()]
    if parts == ["agg_field", "base_raster"]:
        first_col = agg_field
        columns = [
            f"{field}_{stem}" for stem in raster_stems for field in stat_fields
        ]

        def row_iter():
            for group_value in sorted(
                all_groups, key=lambda v: (v is None, str(v))
            ):
                row = {
                    first_col: "" if group_value is None else str(group_value)
                }
                for stem in raster_stems:
                    s = raster_stats_by_stem[stem][group_value]
                    for field in stat_fields:
                        row[f"{field}_{stem}"] = s[field]
                yield row

    elif parts == ["base_raster", "agg_field"]:
        first_col = "base_raster"
        columns = [
            f'{field}_{"" if gv is None else str(gv)}'
            for gv in sorted(all_groups, key=lambda v: (v is None, str(v)))
            for field in stat_fields
        ]

        ordered_groups = sorted(all_groups, key=lambda v: (v is None, str(v)))

        def row_iter():
            for stem in raster_stems:
                row = {first_col: stem}
                stats = raster_stats_by_stem[stem]
                for group_value in ordered_groups:
                    s = stats[group_value]
                    group_label = (
                        "" if group_value is None else str(group_value)
                    )
                    for field in stat_fields:
                        row[f"{field}_{group_label}"] = s[field]
                yield row

    else:
        raise ValueError(
            "row_col_order must be 'agg_field,base_raster' or 'base_raster,agg_field'"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[first_col] + columns)
        writer.writeheader()
        writer.writerows(row_iter())


def _invoke_timed_callback(reference_time, callback_lambda, callback_period):
    """Invoke callback if a certain amount of time has passed.

    This is a convenience function to standardize update callbacks from the
    module.

    Args:
        reference_time (float): time to base ``callback_period`` length from.
        callback_lambda (lambda): function to invoke if difference between
            current time and ``reference_time`` has exceeded
            ``callback_period``.
        callback_period (float): time in seconds to pass until
            ``callback_lambda`` is invoked.

    Return:
        ``reference_time`` if ``callback_lambda`` not invoked, otherwise the
        time when ``callback_lambda`` was invoked.

    """
    current_time = time.time()
    if current_time - reference_time > callback_period:
        callback_lambda()
        return current_time
    return reference_time


def main():
    """CLI entrypoint for validating a zonal-stats runner configuration.

    Parses a single positional argument pointing to an INI configuration file,
    validates it via `parse_and_validate_config`, configures logging based on the
    `[project].log_level` setting, and logs a validation summary for each job.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to INI configuration file")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = parse_and_validate_config(cfg_path)

    log_level = getattr(logging, cfg["project"]["log_level"])
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d: %(message)s",
    )
    logger = logging.getLogger(cfg["project"]["name"])
    logger.info("Loaded config %s", str(cfg_path))
    task_graph = taskgraph.TaskGraph(
        cfg["project"]["global_work_dir"], len(cfg["job_list"]) + 1, 15.0
    )

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    for job in cfg["job_list"]:
        logger.info(
            "Validated job:%s (operations=%s)",
            job["tag"],
            ",".join(job["operations"]),
        )
        output_path = job["output_csv"]
        output_path_timestamped = output_path.with_name(
            f"{output_path.stem}_{timestamp}{output_path.suffix}"
        )
        job["output_csv"] = output_path_timestamped

        task_graph.add_task(
            func=run_zonal_stats_job,
            kwargs=job,
            target_path_list=[output_path_timestamped],
            task_name=f"zonal stats {job['tag']}",
        )
    logger.info("All jobs validated (%d)", len(cfg["job_list"]))
    task_graph.join()
    task_graph.close()


if __name__ == "__main__":
    main()
