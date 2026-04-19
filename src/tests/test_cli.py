"""Tests for the CLI interface."""

from crf.cli import main


def test_cli_main(capsys):
    """Test the main function of the CLI."""
    main()
    captured = capsys.readouterr()
    assert "Hello from crf!" in captured.out

