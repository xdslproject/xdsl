async def _():
    # Get the current notebook URL, drop the 'blob' URL components that seem to be added,
    # and add the buildnumber that a makethedocs PR build seems to add. This allows to load
    # the wheel both locally and when deployed to makethedocs.
    async def import_toy():
        import re
        from urllib.parse import urlparse

        import micropip
        from marimo import notebook_location

        url = str(notebook_location()).replace("blob:", "")
        print(f"DEBUG: notebook url (full): {url}")

        url_parsed = urlparse(url)
        scheme = url_parsed.scheme
        netloc = url_parsed.netloc

        print(f"DEBUG: notebook url (parsed): {url_parsed}")

        url = re.sub(
            "([^/])/([a-f0-9-]+-[a-f0-9-]+-[a-f0-9-]+-[a-f0-9-]+)", "\\1/", url, count=1
        )
        buildnumber = re.sub(".*--([0-9+]+).*", "\\1", url, count=1)

        new_url = scheme + "://" + netloc

        if buildnumber != url:
            new_url = new_url + "/" + buildnumber + "/"

        print(f"DEBUG: notebook url (trimmed): {new_url}")

        await micropip.install("xdsl @ " + new_url + "/xdsl-0.0.0-py3-none-any.whl")
        # deps=False so micropip does not pull xDSL from PyPI; use the local wheel above.
        await micropip.install(
            "toy @ " + new_url + "/toy-0.0.0-py3-none-any.whl", deps=False
        )
        await micropip.install("riscemu==2.2.7")

    await import_toy()
    from xdsl.utils import marimo as xmo

    return xmo
