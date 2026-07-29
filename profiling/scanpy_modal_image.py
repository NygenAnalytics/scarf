import modal


MODAL_APP_NAME = "scarf-profiling-scanpy"

app = modal.App(MODAL_APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential",
        "git",
    )
    .uv_sync(
        groups=["profiling-scanpy"],
        frozen=True,
        extra_options="--no-default-groups --no-install-project",
    )
    .add_local_python_source("profiling", copy=True)
    .add_local_file("uv.lock", "/root/uv.lock", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
)


COMMON_FUNCTION_OPTIONS = {
    "image": image,
    "retries": 0,
    "single_use_containers": True,
}
