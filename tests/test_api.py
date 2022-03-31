import json
from steamship import Steamship
from src.api import QuestionAnswer
from steamship.data.embedding import EmbedAndSearchResponse

import os
from typing import List

__copyright__ = "Steamship"
__license__ = "MIT"

def _get_test_facts() -> List[str]:
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', 'facts.txt'), 'r') as f:
        return f.read().split("\n")

def test_embedder():
    """Tests that our embedder properly associates certain sentences nearby known facts in embedding space."""
    client = Steamship()
    qa = QuestionAnswer(client)

    facts = _get_test_facts()
    for fact in facts:
      response = qa.learn(fact=fact)
      assert(response.error is None)
      assert(response.http is not None)
      assert(response.http.status == 200)
      assert(response.body is not None)


    tests = [
        ("What does Ted think about eggs?", "Ted thinks eggs are good."),
        ("Can everyone eat cake?", "Armadillos are allergic to cake."),
        ("Can armadillos eat everything?", "Armadillos are allergic to cake."),
        ("Who should I give this apple to?", "Jerry likes to eat apples.")
    ]

    for test in tests:
        query_response = qa.query(query=test[0], k=1)
        assert (query_response.error is None)
        assert (query_response.body is not None)

        response = EmbedAndSearchResponse.from_dict(json.loads(query_response.body))
        assert(response.hits is not None)
        assert(len(response.hits) == 1)
        assert(response.hits[0].value == test[1])

