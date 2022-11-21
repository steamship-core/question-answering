"""Tests."""
import os
from typing import List

from steamship import Steamship

from src.api import QuestionAnsweringPackage

__copyright__ = "Steamship"
__license__ = "MIT"


def _get_test_facts() -> List[str]:
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "..", "test_data", "facts.txt"), "r") as f:
        return f.read().split("\n")


def test_basic_similarity_lookup():
    """Tests that our embedder properly associates certain sentences nearby known facts in embedding space."""
    client = Steamship()
    qa = QuestionAnsweringPackage(client)

    facts = _get_test_facts()
    for fact in facts:
        response = qa.learn(fact=fact)
        assert response is not None

    tests = [
        ("What does Ted think about eggs?", "Ted thinks eggs are good."),
        ("Can armadillos eat everything?", "Armadillos are allergic to cake."),
        ("Who should I give this apple to?", "Jerry likes to eat apples."),
    ]

    for test in tests:
        response = qa.query(query=test[0], k=1)
        assert response.items is not None
        assert len(response.items) == 1
        assert response.items[0].value.value == test[1]


def test_lookup_with_metadata():
    """Tests that our embedder properly associates certain sentences nearby known facts in embedding space."""
    client = Steamship()
    qa = QuestionAnsweringPackage(client)

    qa.learn(
        fact="Firm shall repay on the fourth of the month.",
        metadata={"paragraph": 1, "filename": "contract.txt"},
    )

    qa.learn(
        fact="Client shall henceforth be referred to as THE CLIENT.",
        metadata={"paragraph": 2, "filename": "some_other_contract.txt"},
    )

    resp = qa.query("You need to repay on the first of the month", k=1)

    assert resp.items is not None
    assert len(resp.items) == 1
    assert resp.items[0].value.metadata is not None
    assert resp.items[0].value.metadata.get("paragraph") == 1
    assert resp.items[0].value.metadata.get("filename") == "contract.txt"
