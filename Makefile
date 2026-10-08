.PHONY: bootstrap check doctor smoke connectivity public-review

bootstrap:
	bash scripts/start_here.sh

check:
	bash scripts/check.sh

doctor:
	uv run home-credit doctor

smoke:
	uv run home-credit dataframe-smoke
	uv run home-credit model-smoke
	uv run home-credit metric-smoke

connectivity:
	bash scripts/connectivity_check.sh

public-review:
	python3 scripts/review_public_evidence.py --output artifacts/public-review
