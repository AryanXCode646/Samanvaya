# Samanvaya Makefile: Single-Command Automation for ISRO Chandrayaan-2 Registration

.PHONY: install run test clean info help pipeline metrics evaluate verify-raster report-pdf prefetch-weights dataset-manifest validate-datasets preprocess-real-data run-real-benchmark run-ablation generate-report validate-real-pair

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
	@echo "  make prefetch-weights - Download LoFTR weights before an offline demo"
	@echo "  make clean        - Remove build artifacts and caches"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

run:
	@echo "🚀 Launching Samanvaya Streamlit Portal on http://localhost:8501 ..."
	$(PYTHON) -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

test:
	@echo "🧪 Running automated verification test suite..."
	$(PYTHON) -m pytest tests/ -v

pipeline:
	@echo "🛰️ Running end-to-end registration pipeline..."
	$(PYTHON) run_pipeline.py --scenario scenario_a

metrics:
	@echo "📊 Computing evaluation metrics (RMSE, Inlier Ratio, Uniformity)..."
	$(PYTHON) -m lunar_core.evaluation.metrics

evaluate: metrics

verify-raster:
	@echo "🛰️ Running real raster out-of-core verification harness..."
	$(PYTHON) verify_raster_run.py --scenario scenario_a

report-pdf:
	@echo "📄 Generating ISRO Mission Registration PDF Report..."
	$(PYTHON) -m lunar_core.evaluation.pdf_reporter

# Real-data validation workflow scaffold. This is intentionally conservative and does not fabricate mission results.
dataset-manifest:
	@echo "📦 Validating mission dataset manifest..."
	$(PYTHON) -c "from lunar_core.data_io.dataset_manifest import load_dataset_manifest; print(load_dataset_manifest('datasets/manifest.yaml'))"

validate-datasets:
	@echo "🧪 Checking dataset readiness status..."
	$(PYTHON) -c "from lunar_core.data_io.dataset_manifest import summarize_dataset_status; import json; print(json.dumps(summarize_dataset_status('datasets/manifest.yaml'), indent=2))"

preprocess-real-data:
	@echo "📥 Real-data preprocessing is data-dependent and remains pending external mission products."
	@echo "This step prepares the pipeline for externally supplied mission data without fabricating results."

run-real-benchmark:
	@echo "🛰️ Real benchmark execution requires checked-in mission products and independent ground truth."
	@echo "No real-mission validation is claimed until those datasets are provided."

run-ablation:
	@echo "📊 Ablation study entry point is prepared for real-data runs; current execution is blocked by missing external datasets."
	@echo "Synthetic and real results remain explicitly separated."

generate-report:
	@echo "📄 Generating the real-data validation report scaffold..."
	@echo "This repository is intentionally limited to evidence-backed synthetic + manifest validation until real dataset inputs exist."

validate-real-pair:
	@test -n "$(OHRC)" && test -n "$(LROC)" || { echo "Usage: make validate-real-pair OHRC=/path/to/ohrc LROC=/path/to/lroc"; exit 1; }
	@echo "📥 Importing real OHRC ↔ LROC pair..."
	$(PYTHON) -m samanvaya.data_io.import_real_pair --ohrc "$(OHRC)" --lroc "$(LROC)" --manifest data/real/manifest.json --artifact-root artifacts/real_validation
	@PAIR_ID=$$($(PYTHON) -c "import json, pathlib; p=pathlib.Path('data/real/manifest.json'); data=json.loads(p.read_text()) if p.exists() else {}; pairs=data.get('pairs',[]); print(pairs[-1]['pair_id'] if pairs else 'PAIR_MISSING')"); \
	if [ "$$PAIR_ID" = "PAIR_MISSING" ]; then echo "No imported pair found in manifest."; exit 1; fi; \
	$(PYTHON) -m samanvaya.validation.run_real_pair --pair-id "$$PAIR_ID" --manifest data/real/manifest.json --artifact-root artifacts/real_validation; \
	$(PYTHON) -m samanvaya.validation.evaluate_real_pair --pair-id "$$PAIR_ID" --manifest data/real/manifest.json
	@echo "📄 Benchmark report template: docs/REAL_PAIR_BENCHMARK.md"
	@echo "Current ground-truth status is reported as PENDING unless independent control points are supplied."

prefetch-weights:
	@echo "⬇️ Prefetching LoFTR pretrained weights..."
	$(PYTHON) -c "from kornia.feature import LoFTR; LoFTR(pretrained='outdoor'); print('LoFTR weights ready.')"

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__ */__pycache__ */*/__pycache__
	@echo "🧹 Cleaned repository build artifacts."
