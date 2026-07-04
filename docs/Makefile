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

.PHONY: help Makefile execute-docs execute-vignette execute-vignettes prune-stale-cache html execute-notebooks-all

# Execute one doc page locally and update docs/.jupyter_cache.
execute-vignette:
	@uv run python execute_vignette.py $(VIGNETTE)

# Execute quickstart + all vignettes in parallel; merge into docs/.jupyter_cache.
execute-vignettes execute-docs:
	@uv run python execute_all_vignettes.py -j $(or $(JOBS),1)

# Remove cache entries whose source .md files were deleted.
prune-stale-cache:
	@uv run python -c "from docs.execute_vignette import prune_stale_cache; removed = prune_stale_cache(); print(f'Pruned {len(removed)} entries')"

# Build HTML locally (uses committed docs/.jupyter_cache; does not re-execute).
html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Wipe cache and rebuild HTML (sequential re-execution via Sphinx; slower than execute-docs).
execute-notebooks-all:
	@rm -rf .jupyter_cache
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
