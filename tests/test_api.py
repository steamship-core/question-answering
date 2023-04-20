from steamship import Steamship
from steamship_langchain.vectorstores import SteamshipVectorStore
from src.api import QuestionAnswering

import os
from typing import List


def _get_test_facts() -> List[str]:
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "..", "test_data", "facts.txt"), "r") as f:
        return f.read().split("\n")


def test_answer():
    """Tests that our embedder properly associates certain sentences nearby known facts in embedding space."""

    with Steamship.temporary_workspace() as client:
        qa = QuestionAnswering(client=client, config={"index_name": "test-index"})

        vectorstore = SteamshipVectorStore(client=client,
                                           index_name="test-index",
                                           embedding="text-embedding-ada-002")

        facts = _get_test_facts()
        metadatas = [{"source": f"test-{i}"} for i, _ in enumerate(facts)]
        vectorstore.add_texts(facts, metadatas)

        tests = [
            ("What does Ted think about eggs?", "Ted thinks eggs are good."),
            ("Can all animals eat cake?", "Armadillos are allergic to cake."),
            ("Can armadillos eat everything?", "Armadillos are allergic to cake."),
            ("Who would like to eat this apple?", "Jerry likes to eat apples.")
        ]

        for test in tests:
            response = qa.answer(question=test[0], k=1)
            print(response["answer"])
            assert(response["answer"] is not None)
            assert(response["answer"].strip() != "I don't know.")
            assert(response["source_documents"][0].page_content == test[1])
