# -------- LaTeX Makefile -------------------------------------------------
# Usage:
#   make              # build PDF into build/
#   make watch        # live compile (requires latexmk)
#   make view         # open the PDF
#   make clean        # remove aux files
#   make distclean    # remove build/ and PDF
# -------------------------------------------------------------------------

# Core settings
PROJECT ?= main                 # name without .tex
OUTDIR  ?= build
ENGINE  ?= pdflatex             # pdflatex | xelatex | lualatex
BIBTOOL ?= bibtex               # bibtex | biber
SHELL   := /bin/bash

# Detect latexmk
USE_LATEXMK := $(shell command -v latexmk >/dev/null 2>&1 && echo yes || echo no)

# Map ENGINE to latexmk switch
ifeq ($(ENGINE),pdflatex)
  LATEXMK_SWITCH := -pdf
else ifeq ($(ENGINE),xelatex)
  LATEXMK_SWITCH := -pdfxe
else ifeq ($(ENGINE),lualatex)
  LATEXMK_SWITCH := -pdflua
else
  $(error ENGINE must be one of: pdflatex, xelatex, lualatex)
endif

# Common flags
LATEX_FLAGS := -halt-on-error -file-line-error -interaction=nonstopmode -shell-escape
MKDIR_P     := mkdir -p

# File groups for basic dependency tracking
TEX_SRC := $(shell find . -maxdepth 2 -type f -name "*.tex")
BIB_SRC := $(shell find . -maxdepth 2 -type f -name "*.bib")
STY_SRC := $(shell find . -maxdepth 2 -type f \( -name "*.sty" -o -name "*.cls" -o -name "*.cfg" \))
IMG_SRC := $(shell find . -maxdepth 3 -type f \( -name "*.pdf" -o -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \))

# Default target
.PHONY: all
all: $(OUTDIR)/$(PROJECT).pdf

# -----------------------------------------------------------------------------
# Build with latexmk if present (recommended)
# -----------------------------------------------------------------------------
ifeq ($(USE_LATEXMK),yes)

$(OUTDIR)/$(PROJECT).pdf: $(PROJECT).tex $(TEX_SRC) $(BIB_SRC) $(STY_SRC) $(IMG_SRC)
	@$(MKDIR_P) $(OUTDIR)
	latexmk $(LATEXMK_SWITCH) $(LATEX_FLAGS) -quiet -outdir=$(OUTDIR) $(PROJECT).tex

.PHONY: watch
watch:
	@$(MKDIR_P) $(OUTDIR)
	latexmk $(LATEXMK_SWITCH) $(LATEX_FLAGS) -pvc -outdir=$(OUTDIR) $(PROJECT).tex

else
# -----------------------------------------------------------------------------
# Fallback: manual passes with chosen engine + bibtex/biber
# -----------------------------------------------------------------------------

# One engine run
define RUN_LATEX
$(ENGINE) $(LATEX_FLAGS) -output-directory=$(OUTDIR) $(PROJECT).tex
endef

# Bib pass (only if there are .bib files)
ifeq ($(BIBTOOL),biber)
  BIB_CMD = biber --quiet --output-directory $(OUTDIR) $(PROJECT)
else
  BIB_CMD = (cd $(OUTDIR) && bibtex $(PROJECT))
endif

$(OUTDIR)/$(PROJECT).pdf: $(PROJECT).tex $(TEX_SRC) $(BIB_SRC) $(STY_SRC) $(IMG_SRC)
	@$(MKDIR_P) $(OUTDIR)
	@echo ">> latexmk not found; using manual passes with $(ENGINE) + $(BIBTOOL)"
	$(RUN_LATEX) || (echo "LaTeX error. Check logs in $(OUTDIR)/$(PROJECT).log"; exit 1)
ifneq ($(strip $(BIB_SRC)),)
	$(BIB_CMD) || true
	$(RUN_LATEX)
	$(RUN_LATEX)
else
	# no bibliography; a second pass usually resolves refs/tocs
	$(RUN_LATEX)
endif

.PHONY: watch
watch:
	@echo "watch: requires latexmk. Install it or use: while inotifywait -e modify -r .; do make; done"
	@exit 1
endif

# -----------------------------------------------------------------------------
# Convenience targets
# -----------------------------------------------------------------------------
.PHONY: view
view: $(OUTDIR)/$(PROJECT).pdf
	@if command -v xdg-open >/dev/null 2>&1; then xdg-open $(OUTDIR)/$(PROJECT).pdf >/dev/null 2>&1 & else echo "Open $(OUTDIR)/$(PROJECT).pdf"; fi

.PHONY: clean
clean:
	@rm -f $(OUTDIR)/*.{aux,bbl,bcf,blg,idx,ilg,ind,lof,log,lot,out,run.xml,toc,fdb_latexmk,fls,synctex.gz}

.PHONY: distclean
distclean: clean
	@rm -f $(OUTDIR)/$(PROJECT).pdf
	@rmdir --ignore-fail-on-non-empty $(OUTDIR) >/dev/null 2>&1 || true


# Describe targets with "## ..." and they'll show up in `make help`

.PHONY: help
help: ## Show this help and current config
	@echo "LaTeX Makefile — targets"; \
	awk 'BEGIN {FS":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST); \
	echo ""; \
	echo "Variables (override like VAR=value make):"; \
	printf "  %-15s %s\n" "PROJECT" "$(PROJECT)"; \
	printf "  %-15s %s\n" "OUTDIR"  "$(OUTDIR)"; \
	printf "  %-15s %s\n" "ENGINE"  "$(ENGINE)"; \
	printf "  %-15s %s\n" "BIBTOOL" "$(BIBTOOL)"

# Add "## ..." after your targets so help can find them:
all: ## Build PDF into $(OUTDIR)/
	@:

watch: ## Live-compile (latexmk -pvc)
	@:

view: ## Open the built PDF
	@:

clean: ## Remove aux files from $(OUTDIR)/
	@:

distclean: ## Remove aux + PDF and maybe $(OUTDIR)/
	@:
