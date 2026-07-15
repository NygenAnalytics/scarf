import modal


MODAL_APP_NAME = "scarf-profiling"

app = modal.App(MODAL_APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential",
        "git",
        "libfftw3-dev",
        "libmetis-dev",
        "libtbb-dev",
    )
    .uv_sync(
        groups=["profiling"],
        frozen=True,
        extra_options="--no-default-groups",
    )
    .add_local_python_source("scarf", "profiling", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .add_local_file("VERSION", "/root/VERSION", copy=True)
)


COMMON_FUNCTION_OPTIONS = {
    "image": image,
    "retries": 0,
    "single_use_containers": True,
}
