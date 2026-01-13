import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "xray_image_classifier"
PYTHON_VERSION = "3.13"


# Project commands
@task
def lint(ctx: Context) -> None:
    """Run linter (ruff)."""
    ctx.run("uv run ruff .", echo=True, pty=not WINDOWS)


@task
def typecheck(ctx: Context) -> None:
    """Run static type checks (mypy)."""
    ctx.run("uv run mypy src", echo=True, pty=not WINDOWS)


@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context, backbone: str = "densenet121", profile: bool = False) -> None:
    """Train model."""

    cmd = f"uv run src/{PROJECT_NAME}/train.py " f"--data-dir data/processed/train " f"--backbone {backbone}"
    if profile:
        cmd += " --profile"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context, backbone: str = "densenet121") -> None:
    """Evaluate model."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/evaluate.py " f"--data-dir data/processed/test " f"--backbone {backbone}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
