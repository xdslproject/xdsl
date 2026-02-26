import os  # noqa


def build_xdsl_wheel(config: dict[str, str], **kwargs: object):
    if os.environ.get("SKIP_BUILD_WHEEL") == "1":
        return

    site_dir = config["site_dir"]

    os.system(
        f"SETUPTOOLS_SCM_PRETEND_VERSION='0.0.0' uv build --package xdsl --no-build-logs --no-verify-hashes -o {site_dir}"
    )
