# Building and contributing

Use Python 3.12 for the documentation toolchain, matching CI:

```console
python -m pip install -e ".[docs,test]"
python scripts/check_examples.py
python -m pytest
python -m sphinx -W --keep-going -b html docs docs/_build/html
python scripts/check_docs_discovery.py
```

Open `docs/_build/html/index.html` to preview the result. The site uses the same
Sphinx, MyST, and PyData theme conventions as TreeIG and CBaseline. Examples are
included directly from runnable files so the guide and checked code stay aligned.

## GitHub Pages

In repository Settings → Pages, select GitHub Actions as the source. The docs
workflow builds and checks examples on pushes and pull requests. Successful main
branch builds upload a Pages artifact and deploy through the `github-pages`
environment. Ensure that environment permits main-branch deployments.

The expected address is `https://ludgerhentschel.github.io/skgrad/`.
Configuring this workflow does not itself enable Pages or make the repository
public. A pull request build never deploys the site.

## Package release

Update the version in `pyproject.toml` and release notes together.
The release workflow calls the same complete test workflow used on branches:
Python 3.9–3.13, macOS and Windows, a pinned older dependency stack, strict docs,
and installed-wheel tests and examples. Only that workflow's tested distribution
artifact can proceed to PyPI publishing; failed or cancelled checks block it. Run tests,
examples, strict docs, package builds, and `twine check` before creating a tag.
The release guard requires the tag to equal `v` plus the package version.
Use a fresh tag; do not move the historical 0.1.2–0.1.4 tags. Verify the Trusted
Publisher and required CI checks before pushing a release tag.

The project is licensed under BSD-3-Clause. New backends must define the output
scale and compare derivatives with independent calculations.

The package job builds a wheel from the source distribution, installs it, and
runs `python scripts/check_installed_distribution.py` using Python 3.12. This
copies tests/examples outside the checkout and removes PYTHONPATH so an editable
source installation cannot masquerade as the wheel. Test dependencies include
pandas to exercise DataFrame feature-name validation.

## Release checklist for 0.1.6

1. Push the release commit to `main` and wait for Tests and Documentation to
   succeed, including the Pages deployment.
2. In GitHub Settings → Pages, select **GitHub Actions**. In Settings →
   Environments, ensure `github-pages` permits `main` and `pypi` permits the
   `v0.1.6` tag (and approve deployments if protection rules require it).
3. In PyPI's skgrad project → Publishing, verify a GitHub Trusted Publisher with
   owner `LudgerHentschel`, repository `skgrad`, workflow `release.yml`, and
   environment `pypi`. The workflow filename and environment must match exactly.
4. After branch checks and publisher settings are verified, create and push
   `v0.1.6` at that same commit. This tag starts publication automatically after
   the release checks pass. Do not push the tag while checks are unresolved.
5. Verify PyPI displays version 0.1.6 and BSD-3-Clause, install the published
   package in a fresh environment, and check the documentation link.

Changing repository visibility is a separate action. The workflow does not
change it. Release skgrad before releasing UnifiedIG's dependency update.

## Discovery files

Maintain the repository-root `llms.txt` as an annotated guide to the published
site. Sphinx copies this one source through `html_extra_path`. Link to rendered
API pages and complete example files; raw Sphinx Markdown can contain unexpanded
autodoc or literalinclude instructions.

`sphinx-sitemap` generates `sitemap.xml` using `html_baseurl`, which also supplies
canonical page URLs. Important pages define concise descriptions in
`myst.html_meta` YAML front matter. The discovery check runs after HTML builds
in both documentation workflows and checks the index copy, local targets,
sitemap URLs, canonical links, descriptions, and expanded API/example content.
After deployment, verify `/skgrad/llms.txt` and `/skgrad/sitemap.xml` on the public
site and check external links in the index.
