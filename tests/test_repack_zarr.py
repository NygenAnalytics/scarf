import numpy as np
import pytest
import zarr
from zarr.codecs import ZstdCodec

from scarf.tools.repack_zarr import repack_store


def test_repack_store_round_trip(toy_crdir_writer, tmp_path):
    output = tmp_path / "repacked.zarr"
    repack_store(toy_crdir_writer, str(output), profile="fast_local", shard_counts=True)

    src = zarr.open_group(toy_crdir_writer, mode="r")
    dst = zarr.open_group(str(output), mode="r")

    assert set(src.keys()) == set(dst.keys())
    assay_names = [name for name in src.keys() if src[name].attrs.get("is_assay")]
    for assay_name in assay_names:
        src_assay = src[assay_name]
        dst_assay = dst[assay_name]
        assert src_assay.attrs.get("is_assay") is True
        assert "counts" in dst_assay
        assert src_assay["counts"].shape == dst_assay["counts"].shape
        assert (src_assay["counts"][...] == dst_assay["counts"][...]).all()
        assert dst_assay["countsT"].attrs["complete"] is True
        np.testing.assert_array_equal(
            dst_assay["countsT"][:],
            np.asarray(dst_assay["counts"][:]).T,
        )


def test_repack_store_without_sharding(toy_crdir_writer, tmp_path):
    output = tmp_path / "repacked_plain.zarr"
    repack_store(
        toy_crdir_writer,
        str(output),
        profile="fast_local",
        shard_counts=False,
    )
    dst = zarr.open_group(str(output), mode="r")
    assert "RNA" in dst
    assert "counts" in dst["RNA"]


def test_repack_v2_without_counts_t_builds_complete_transpose(tmp_path):
    source = tmp_path / "source_v2.zarr"
    output = tmp_path / "output_v3.zarr"
    root = zarr.open_group(str(source), mode="w", zarr_format=2)
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    values = np.arange(12, dtype=np.uint32).reshape(3, 4)
    assay.create_array("counts", data=values, chunks=(2, 2))

    repack_store(str(source), str(output), shard_counts=True)

    result = zarr.open_group(str(output), mode="r")
    assert result.metadata.zarr_format == 3
    assert result["RNA/countsT"].attrs["complete"] is True
    np.testing.assert_array_equal(result["RNA/counts"][:], values)
    np.testing.assert_array_equal(result["RNA/countsT"][:], values.T)


def test_repack_rebuilds_incorrect_source_counts_t(tmp_path):
    source = tmp_path / "source_wrong.zarr"
    output = tmp_path / "output_fixed.zarr"
    root = zarr.open_group(str(source), mode="w")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    values = np.arange(15, dtype=np.uint32).reshape(5, 3)
    assay.create_array("counts", data=values, chunks=(2, 3))
    stale = assay.create_array(
        "countsT",
        data=np.zeros((3, 5), dtype=np.uint32),
    )
    stale.attrs["complete"] = True

    repack_store(str(source), str(output), shard_counts=True)

    result = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(result["RNA/countsT"][:], values.T)
    assert result["RNA/countsT"].attrs["complete"] is True


def test_repack_workspace_counts_uses_requested_profile(
    tmp_path,
):
    source = tmp_path / "source_workspace.zarr"
    output = tmp_path / "output_workspace.zarr"
    root = zarr.open_group(str(source), mode="w")
    metadata = root.create_group("workspace/RNA")
    metadata.attrs["is_assay"] = True
    counts_group = root.create_group("matrices/RNA")
    values = np.arange(8, dtype=np.uint32).reshape(4, 2)
    counts_group.create_array("counts", data=values, chunks=(2, 2))

    repack_store(
        str(source),
        str(output),
        profile="cloud",
        shard_counts=True,
    )

    result = zarr.open_group(str(output), mode="r")
    assay = result["matrices/RNA"]
    counts = assay["counts"]
    counts_t = assay["countsT"]
    np.testing.assert_array_equal(counts_t[:], values.T)
    assert counts_t.attrs["complete"] is True
    assert assay.attrs["scarf:zarr_spec"]["profile"] == "cloud"
    assert isinstance(counts.compressors[0], ZstdCodec)
    assert isinstance(counts_t.compressors[0], ZstdCodec)


def test_repack_rejects_source_destination_alias_before_overwrite(tmp_path):
    source = tmp_path / "source.zarr"
    root = zarr.open_group(str(source), mode="w")
    root.create_array("sentinel", data=np.array([1, 2, 3]))
    equivalent_path = source / ".." / source.name

    with pytest.raises(ValueError, match="different stores"):
        repack_store(str(source), str(equivalent_path))

    reopened = zarr.open_group(str(source), mode="r")
    np.testing.assert_array_equal(reopened["sentinel"][:], [1, 2, 3])


@pytest.mark.parametrize("destination_relation", ["child", "parent"])
def test_repack_rejects_overlapping_local_paths_before_overwrite(
    tmp_path,
    destination_relation,
):
    if destination_relation == "child":
        source = tmp_path / "source.zarr"
        output = source / "nested.zarr"
    else:
        output = tmp_path / "container.zarr"
        source = output / "source.zarr"

    root = zarr.open_group(str(source), mode="w")
    root.create_array("sentinel", data=np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="must not overlap"):
        repack_store(str(source), str(output))

    reopened = zarr.open_group(str(source), mode="r")
    np.testing.assert_array_equal(reopened["sentinel"][:], [1, 2, 3])


@pytest.mark.parametrize(
    ("source", "output"),
    [
        ("s3://bucket/source.zarr", "s3://bucket/source.zarr/nested.zarr"),
        ("s3://bucket/container.zarr/source.zarr", "s3://bucket/container.zarr"),
    ],
)
def test_repack_rejects_overlapping_uri_paths_before_open(
    source,
    output,
    monkeypatch,
):
    import scarf.tools.repack_zarr as repack_module

    monkeypatch.setattr(
        repack_module,
        "open_store",
        lambda *_args, **_kwargs: pytest.fail(
            "overlap must be rejected before opening either store"
        ),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        repack_store(source, output)
