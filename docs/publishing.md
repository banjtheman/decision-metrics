# Publishing DecisionMetrics

The distribution name is `decision-metrics`; the import is `decision_metrics`.
The project uses semantic versions and is currently an alpha release. Keep the
version in `pyproject.toml` and the Git tag (`v0.1.0`, for example) in agreement.

## Set up GitHub Trusted Publishing

The release workflow uses PyPI's Trusted Publishing, with short-lived credentials
issued to the publishing job. Register these fields in your PyPI account's
[Publishing settings](https://pypi.org/manage/account/publishing/) for a new
project, or in the project's Publishing settings once it exists:

| Field | Value |
| --- | --- |
| PyPI project name | `decision-metrics` |
| GitHub owner | `banjtheman` |
| GitHub repository | `decision-metrics` |
| Workflow filename | `publish.yml` |
| Environment name | `pypi` |

A pending publisher creates the PyPI project on its first successful upload; it
does not reserve the name. See [PyPI's setup guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Release a version

1. Update `pyproject.toml` and `src/decision_metrics/types.py` to the same version,
   and document changes in `CHANGELOG.md` or the release notes.
2. Push the changes and wait for the tests workflow to pass. CI builds both
   distributions and runs the tests against the installed wheel on Python
   3.11–3.13.
3. Tag that commit with the matching `v` version and push the tag.
4. Publish a GitHub release for that tag. `publish.yml` checks the version, builds
   and validates the distributions, runs the tests, and uploads through the
   `pypi` environment. OIDC permission is limited to the upload job.
5. Check the PyPI listing and install the version in a fresh environment.

The workflow can also be run manually with an existing version tag, which is
useful if account setup was completed after creating the release. Uploaded
versions cannot be overwritten; use a new version for changed package contents.

## Build and check locally

With a Python 3.11+ environment active:

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```

`python -m build` builds the wheel from the source distribution, checking that the
source package includes the files needed to build it. The wheel contains the
library, CLI, metadata, README description, and MIT license. Sample configuration
and replay packets are available from the GitHub repository.

If bootstrapping a first upload with an existing local PyPI credential instead
of Trusted Publishing, upload the checked distributions with
`python -m twine upload --non-interactive dist/*`. Configure the GitHub publisher
on the new project's Publishing page before using automated releases.
