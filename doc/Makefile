# makefile for Progressivis documentation
#
ifndef ENVNAME
override ENVNAME = progressivis-doc
endif

SHELL = /bin/bash
CONDA = conda
PYTHON = python
PRINTF = printf

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = progressivis
SOURCEDIR     = .
BUILDDIR      = _build

# Colors
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[0;33m
NC=\033[0m


.PHONY: help Makefile

# Put it first so that "make" without argument is like "make help".
help:
	@$(PRINTF) "Please use 'make <target>' where <target> is one of:\n"
	@$(PRINTF) "  $(GREEN)clean$(NC) to remove generated doc files and start fresh\n"
	@$(PRINTF) "  $(GREEN)docenv$(NC) [ENVNAME=<custom-name>] make the conda environment in which to build docs (default name: progressivis-doc)\n"
	@$(PRINTF) "  $(GREEN)docreq$(NC) add required packages to build docs to your current conda environment (which have to contain progressivis)\n"
	@$(PRINTF) "  $(GREEN)html$(NC) to make standalone HTML files\n"
	@$(PRINTF) "$(YELLOW)NB:$(NC)\n"
	@$(PRINTF) "To $(GREEN)activate$(NC) the $(YELLOW)$(ENVNAME)$(NC) environment, use:\n"
	@$(PRINTF) "    $(YELLOW)$$ conda activate $(ENVNAME)$(NC)\n"
	@$(PRINTF) "To $(YELLOW)deactivate$(NC) an active environment, use:\n"
	@$(PRINTF) "    $(YELLOW)$$ conda deactivate$(NC)\n"
	@$(PRINTF) "To $(RED)remove$(NC) the $(YELLOW)$(ENVNAME)$(NC) environment, use:\n"
	@$(PRINTF) "    $(YELLOW)$$ conda remove -n $(ENVNAME) --all$(NC)\n"

docenv:
	@$(CONDA) env create -f  ../environment_ci.yml -n create -n $(ENVNAME)
	@$(CONDA) install --file=requirements_doc.txt -c conda-forge  -n $(ENVNAME) -y
	@$(PRINTF) "To $(GREEN)activate$(NC) the $(YELLOW)$(ENVNAME)$(NC) environment, use:\n"
	@$(PRINTF) "    $(YELLOW)$$ conda activate $(ENVNAME)$(NC)\n"
	@$(PRINTF) "To $(YELLOW)deactivate$(NC) an active environment, use:\n"
	@$(PRINTF) "    $(YELLOW)$$ conda deactivate$(NC)\n"
	@$(PRINTF) "To $(RED)remove$(NC) the $(YELLOW)$(ENVNAME)$(NC) environment, use:\n"
	@$(PRINTF) "    $(YELLOW)$$ conda remove -n $(ENVNAME) --all$(NC)\n"

docreq:
	@$(PRINTF) "$(RED)WARNING:$(NC) your current environment have to contain the $(YELLOW)progressivis$(NC) package!\n"
	@$(PYTHON) -c "import progressivis"
	@$(CONDA) install --file=requirements_doc.txt -c conda-forge

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
