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

.PHONY: help Makefile execute-notebooks execute-notebooks-all execute-vignette execute-vignettes

# Execute one vignette locally and update docs/.jupyter_cache.
execute-vignette:
	@uv run python execute_vignette.py $(VIGNETTE)

# Execute all vignettes in parallel and merge into docs/.jupyter_cache.
execute-vignettes:
	@uv run python execute_all_vignettes.py -j $(or $(JOBS),4)

# Execute vignettes locally and refresh docs/.jupyter_cache (commit that directory).
execute-notebooks:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Re-execute every vignette from scratch.
execute-notebooks-all:
	@rm -rf .jupyter_cache
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
