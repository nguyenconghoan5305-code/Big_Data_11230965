from bft65b.cli import main


def test_help(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "show-config" in capsys.readouterr().out


def test_show_config(capsys):
    assert main(["show-config"]) == 0
    assert '"financial_start": "1984-03-31"' in capsys.readouterr().out
