from typer.testing import CliRunner

from whoa_llm.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "whoa-llm" in result.stdout.lower()


def test_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "CUDA available" in result.stdout
