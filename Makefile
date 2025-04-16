AUX_DIR := aux-dir
PDF_DIR := pdf-dir
TEX_DIR := tex-dir
LATEXMK := latexmk
OPTIONS := -pdflua -outdir=$(PDF_DIR) -auxdir=$(AUX_DIR)
HWFILES := $(patsubst %.tex, %.pdf, $(notdir $(wildcard $(TEX_DIR)/*.tex)))
CUPDATE ?= 0
IMG_TAG := cs6140

CMD_CHECK := podman image exists $(IMG_TAG)
CMD_BUILD := podman build $(PWD) -t $(IMG_TAG) -f Dockerfile
CMD_RUN := podman run --rm -v $(PWD):/work -w /work $(IMG_TAG)
CMD_RMI := podman rmi $(IMG_TAG)

ifeq ($(CUPDATE), 1)
	OPTIONS += -pvc
endif

all: $(HWFILES)

build:
	@$(CMD_CHECK) || $(CMD_BUILD)

%.pdf: $(TEX_DIR)/%.tex build
	@$(CMD_RUN) $(LATEXMK) $(OPTIONS) $<

clean: build
	@$(CMD_RUN) $(LATEXMK) $(OPTIONS) -f -c $(AUX_DIR)/*.aux

distclean: build
	@$(CMD_RUN) $(LATEXMK) $(OPTIONS) -f -C $(PDF_DIR)/*.pdf

remove:
	@! $(CMD_CHECK) || $(CMD_RMI)

.PRECIOUS: $(PDF_DIR)/%.pdf

.PHONY: all build clean distclean remove
