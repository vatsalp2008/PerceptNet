.DEFAULT_GOAL := help
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
KITTI := data/kitti/training

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## create venv (py3.11), install package, check dataset
	python3.11 -m venv $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(MAKE) data

data: ## verify KITTI layout; print manual download steps if absent (KITTI needs an account)
	@if [ -d "$(KITTI)/velodyne" ] && [ -d "$(KITTI)/image_2" ] && [ -d "$(KITTI)/calib" ] && [ -d "$(KITTI)/label_2" ]; then \
		echo "KITTI found under $(KITTI):"; \
		echo "  velodyne: $$(ls $(KITTI)/velodyne | wc -l | tr -d ' ') files"; \
		echo "  image_2:  $$(ls $(KITTI)/image_2 | wc -l | tr -d ' ') files"; \
	else \
		echo "KITTI not found under $(KITTI)."; \
		echo "KITTI requires a free account, so it cannot be auto-downloaded:"; \
		echo "  1. Register at https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d"; \
		echo "  2. Download: left color (image_2), Velodyne (velodyne), calib, training labels (label_2)"; \
		echo "  3. Unzip into data/kitti/training/{image_2,velodyne,calib,label_2} (+ testing/)"; \
		echo "  (tests/fixtures/mini_kitti lets the test suite run with no download.)"; \
		exit 1; \
	fi

test: ## run the CPU test suite (excludes gpu/ros/network)
	$(PY) -m pytest

test-all: ## run ALL tests incl. gpu/ros/network (cloud box)
	$(PY) -m pytest -m ""

test-network: ## run network-gated tests (downloads YOLO weights)
	$(PY) -m pytest -m network

train-lidar: ## train PointPillars on KITTI (needs CUDA)
	$(PY) scripts/train_lidar.py

train-lidar-smoke: ## 1-step LiDAR training wiring check on the fixture (CPU OK)
	$(PY) scripts/train_lidar.py --smoke

train-fusion: ## train the ROI fusion head (needs trained LiDAR + CUDA)
	$(PY) scripts/train_fusion.py

train-fusion-smoke: ## fusion-head training wiring check on the fixture (CPU OK)
	$(PY) scripts/train_fusion.py --smoke

evaluate: ## run the KITTI mAP evaluation table
	$(PY) scripts/evaluate.py

evaluate-smoke: ## run evaluation on the fixture (CPU OK)
	$(PY) scripts/evaluate.py --smoke

benchmark: ## latency benchmark (p50/p95/p99 + FPS) on the available device
	$(PY) scripts/benchmark.py

robustness: ## modality-dropout robustness study (meaningful with a trained checkpoint)
	$(PY) scripts/robustness.py

visualize: ## render a fixture frame overlay PNG (works on CPU/Mac)
	$(PY) scripts/visualize.py

export-onnx: ## export backbone+head to ONNX and verify vs ONNX Runtime (CPU OK)
	$(PY) scripts/export_onnx.py

build-trt: ## build a TensorRT engine from the ONNX (CUDA box only)
	$(PY) -c "from perceptnet.optimization.build_trt import build_engine; build_engine('outputs/perceptnet_backbone.onnx','outputs/perceptnet.trt','fp16')"

ros2-build: ## build the ROS 2 Humble docker image
	docker build -f docker/Dockerfile.ros2 -t perceptnet:ros2 .

ros2-run: ## run the ROS 2 perception node container
	docker compose run --rm ros2

docker-train: ## build + run training in the CUDA container
	docker compose run --rm train

clean: ## remove caches / build artifacts
	rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info outputs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: help setup data test test-all test-network train-lidar train-lidar-smoke \
	train-fusion train-fusion-smoke evaluate evaluate-smoke benchmark robustness \
	visualize export-onnx build-trt ros2-build ros2-run docker-train clean
