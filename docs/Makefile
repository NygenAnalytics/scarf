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

.PHONY: help Makefile execute-docs execute-docs-modal execute-page execute-vignette execute-vignettes resume-docs resume-docs-modal validate-cache prune-stale-cache html execute-notebooks-all nitpick coverage check-reference


# Execute one doc page locally and update docs/.jupyter_cache.
# Usage: make execute-page PAGE=scrna_seq
# Alias: make execute-vignette VIGNETTE=scrna_seq (legacy)
execute-page:
	@uv run python execute_all_vignettes.py $(or $(PAGE),$(VIGNETTE),scrna_seq) -j $(or $(JOBS),1)

execute-vignette: execute-page

# Force all executable MyST pages and publish a validated cache.
execute-vignettes execute-docs:
	@uv run python execute_all_vignettes.py --full -j $(or $(JOBS),1)

# Resume the last failed run when source and execution fingerprints still match.
resume-docs:
	@uv run python execute_all_vignettes.py --resume -j $(or $(JOBS),1)

# Execute all pages concurrently on Modal and publish the cache locally.
# Set PAGES="cite_seq batch_correction" for a partial run.
execute-docs-modal:
	@cd .. && MODAL_ENVIRONMENT=scarf_profiling uv run --group docs-modal --extra agent --extra docs --extra extra modal run -m docs.modal_docs $(if $(strip $(PAGES)),--pages "$(strip $(PAGES))",)

# Resume the failed Modal run scope using matching staged page results.
resume-docs-modal:
	@cd .. && MODAL_ENVIRONMENT=scarf_profiling uv run --group docs-modal --extra agent --extra docs --extra extra modal run -m docs.modal_docs --resume $(if $(strip $(PAGES)),--pages "$(strip $(PAGES))",)

# Validate the committed cache without changing it.
validate-cache:
	@uv run python execute_all_vignettes.py --validate

# Rebuild the cache from outputs matching current sources only.
prune-stale-cache:
	@uv run python execute_all_vignettes.py --prune

# Build HTML locally (uses committed docs/.jupyter_cache; does not re-execute).
html: validate-cache
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Nitpicky HTML build used by docs CI (missing refs/toctree fail the build).
nitpick: validate-cache
	@$(SPHINXBUILD) -b html -n -W "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Sphinx coverage report for documented Python objects (sphinx.ext.coverage).
# Use after expanding public __all__ and explicit autosummary pages; do not treat
# package-wide undocumented internals as a release blocker until those boundaries
# are complete. Report: build/coverage/python.txt
coverage:
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS) $(O)
	@echo "Coverage report written to $(BUILDDIR)/coverage/python.txt"

# Fail when a public symbol is missing from, or duplicated in, the built reference.
# Requires a prior HTML build because it reads build/html/objects.inv.
check-reference: nitpick
	@uv run python check_reference_coverage.py


# Force all notebooks through the validated runner, then build strictly.
execute-notebooks-all:
	@uv run python execute_all_vignettes.py --full -j $(or $(JOBS),1)
	@$(SPHINXBUILD) -b html -n -W "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
