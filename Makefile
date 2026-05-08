#-------------------------------- VARIABLES ----------------------------------#

NAME		=	call-me-maybe

SRC			=	src
VENV		=	.venv
MODEL		=	llm_sdk

FUNCIONS	?=	"data/input/functions_definition.json"
INPUT		?=	"data/input/function_calling_tests.json"
OUTPUT		?=	"data/ouput/output.json"

EXCLUDE				=	--exclude $(VENV),$(MODEL)
EXCLUDE_MYPY		=	--exclude $(VENV) --exclude $(MODEL)

#-------------------------------- RULES --------------------------------------#

.PHONY: all install run debug clean fclean re reset lint lint-strict help

all: check_uv install run

install: check_uv
	uv sync

run: check_uv
	@uv run python -m $(SRC) --functions_definition $(FUNCIONS) --input $(INPUT) --output $(OUTPUT)

debug: check_uv
	@uv run python -m pdb $(SRC) --functions_definition $(FUNCIONS) --input $(INPUT) --output $(OUTPUT)

test: check_uv
	uv run pytest tests/ -v

test-cov: check_uv
	uv run pytest tests/ -v --cov=. --cov-report=html
	@echo "Coverage report generated in htmlcov/"

lint: check_uv
	uv run flake8 . $(EXCLUDE)
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs $(EXCLUDE_MYPY)

lint-strict: check_uv
	uv run flake8 . $(EXCLUDE)
	uv run mypy . --strict $(EXCLUDE_MYPY)

clean:
	find . -name "__pycache__" -type d -exec rm -rf "{}" +
	find . -name "*.pyc" -delete
	find . -name ".mypy_cache" -type d -exec rm -rf "{}" +
	find . -name ".pytest_cache" -type d -exec rm -rf "{}" +
	find . -name ".coverage" -delete
	find . -name "htmlcov" -type d -exec rm -rf "{}" +

fclean: clean
	rm -rf .venv

re: check_uv fclean install

reset: check_uv fclean
	rm -f uv.lock
	uv sync

check_uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Error: uv is not installed."; \
		echo "Install it via: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	}

help:
	@echo "Available rules:"
	@echo "  all        	- Install and run"
	@echo "  install    	- Install dependencies"
	@echo "  run        	- Run the project (MAP=... to override map)"
	@echo "  debug      	- Run with pdb debugger"
	@echo "  test       	- Run tests"
	@echo "  test-cov   	- Run tests with coverage report"
	@echo "  format     	- Format code with black and isort"
	@echo "  lint       	- Run flake8 and mypy"
	@echo "  lint-strict	- Run flake8 and mypy in strict mode"
	@echo "  check      	- Format then lint"
	@echo "  clean      	- Remove cache files"
	@echo "  fclean     	- clean + remove venv"
	@echo "  re         	- fclean + install"
	@echo "  reset      	- fclean + remove uv.lock + install"
