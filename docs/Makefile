# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
SPHINXPROJ    = Scarf

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile execute-docs execute-page execute-vignette execute-vignettes prune-stale-cache html execute-notebooks-all nitpick

# Execute one doc page locally and update docs/.jupyter_cache.
# Usage: make execute-page PAGE=scrna_seq
# Alias: make execute-vignette VIGNETTE=scrna_seq (legacy)
execute-page:
	@uv run python execute_vignette.py $(or $(PAGE),$(VIGNETTE),scrna_seq)

execute-vignette: execute-page

# Execute all executable MyST pages; merge into docs/.jupyter_cache.
execute-vignettes execute-docs:
	@set -o pipefail; uv run python execute_all_vignettes.py -j $(or $(JOBS),1)

# Remove cache entries whose source .md files were deleted or moved.
prune-stale-cache:
	@uv run python -c "from docs.execute_vignette import prune_stale_cache; removed = prune_stale_cache(); print(f'Pruned {len(removed)} entries')"

# Build HTML locally (uses committed docs/.jupyter_cache; does not re-execute).
html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Nitpicky HTML build used by docs CI (missing refs/toctree fail the build).
nitpick:
	@$(SPHINXBUILD) -b html -n "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Wipe cache and rebuild HTML (sequential re-execution via Sphinx; slower than execute-docs).
execute-notebooks-all:
	@rm -rf .jupyter_cache
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
