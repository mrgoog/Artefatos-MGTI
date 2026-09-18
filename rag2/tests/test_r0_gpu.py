"""R0 roda de ponta a ponta pela CLI em um subconjunto (GPU) e respeita a forma da saída."""

import pytest
import torch

from rag2.cli import main

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sem CUDA")
def test_degrau_r0_limite(capsys):
    assert main(["degrau", "R0", "--limite", "5", "--b", "200"]) == 0
    out = capsys.readouterr().out
    assert "degrau=R0  itens=5" in out
    assert "artefatos em" in out and "debug-temp" in out
