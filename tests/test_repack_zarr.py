import zarr

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
