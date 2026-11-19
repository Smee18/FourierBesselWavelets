set shell := ["powershell.exe", "-NoProfile", "-Command"]

default: check

fmt:
    ruff format .
    ruff check . --fix

types:
    mypy src/

test:
    pytest -v

# Everything, in order
check: fmt types test