.PHONY: all figs pdf clean

LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIB   = bibtex
MAIN  = main

all: figs pdf

figs:
	mkdir -p figs/wu
	cp -f ../wu/steps/05_wu_pass_ab/wu_mean_pv_3levels.png figs/wu/ 2>/dev/null || true
	cp -f ../wu/steps/05_wu_pass_ab/wu_pv_anomaly_3levels.png figs/wu/ 2>/dev/null || true
	cp -f ../wu/steps/05_wu_pass_ab/wu_mean_psi_500hpa.png figs/wu/ 2>/dev/null || true
	cp -f ../wu/steps/06_wu_pass_c/pass_c_psi_all_levels.png figs/wu/ 2>/dev/null || true
	cp -f ../wu/steps/07_wu_pass_d/pass_d_psi_3pieces_250hpa.png figs/wu/ 2>/dev/null || true
	cp -f ../wu/steps/08_parse_outputs/pv_comparison_250hPa.png figs/wu/ 2>/dev/null || true
	cp -f ../wu/steps/10_fig8/fig8_replica.png figs/wu/ 2>/dev/null || true
	cp -f ../wu/steps/11_xsection/xsection_vinduced_3panel.png figs/wu/ 2>/dev/null || true
	cp -f ../wu/steps/11_xsection/xsection_pv_psi_4panel.png figs/wu/ 2>/dev/null || true
	cp -f ../wu/steps/11_xsection/closure_check.png figs/wu/ 2>/dev/null || true
	@echo "Wu step figures copied into figs/wu/"

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex chapters/*.tex refs.bib
	$(LATEX) $(MAIN)
	-$(BIB) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk
	rm -f chapters/*.aux

distclean: clean
	rm -f $(MAIN).pdf
	rm -rf figs/*.png figs/*.pdf
