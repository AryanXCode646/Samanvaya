# Samanvaya Makefile: Single-Command Automation for ISRO Chandrayaan-2 Registration

.PHONY: install run test clean info help pipeline metrics evaluate verify-raster report-pdf

PYTHON := python3
PIP := pip

help:
	@echo "Samanvaya: Lunar Optical Image Registration Framework (SIH PS 26166)"
	@echo "Usage:"
	@echo "  make install      - Install python dependencies and register package"
	@echo "  make run          - Launch interactive Streamlit demo portal (port 8501)"
	@echo "  make test         - Run full automated verification test suite"
	@echo "  make pipeline     - Run end-to-end lunar registration pipeline"
	@echo "  make metrics      - Generate quantitative evaluation metrics report (JSON & CSV)"
	@echo "  make verify-raster- Real raster out-of-core windowed tile verification"
	@echo "  make report-pdf   - Generate executive ReportLab PDF mission report"
	@echo "  make clean        - Remove build artifacts and caches"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

run:
	@echo "🚀 Launching Samanvaya Streamlit Portal on http://localhost:8501 ..."
	streamlit run app.py --server.port 8501

test:
	@echo "🧪 Running automated verification test suite..."
	pytest tests/ -v

pipeline:
	@echo "🛰️ Running end-to-end registration pipeline..."
	$(PYTHON) run_pipeline.py --scenario scenario_a

metrics:
	@echo "📊 Computing evaluation metrics (RMSE, Inlier Ratio, Uniformity)..."
	$(PYTHON) metrics.py

evaluate: metrics

verify-raster:
	@echo "🛰️ Running real raster out-of-core verification harness..."
	$(PYTHON) verify_raster_run.py --scenario scenario_a

report-pdf:
	@echo "📄 Generating ISRO Mission Registration PDF Report..."
	$(PYTHON) pdf_reporter.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__ */__pycache__ */*/__pycache__
	@echo "🧹 Cleaned repository build artifacts."
