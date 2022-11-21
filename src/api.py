"""Question Answering Package."""
from typing import Dict

from steamship import EmbeddingIndex, PluginInstance, Steamship, SteamshipError
from steamship.data.embeddings import IndexInsertResponse, QueryResults
from steamship.invocable import PackageService, create_handler, post


class QuestionAnsweringPackage(PackageService):
    """Simple question answering class that works with different embedding models."""

    embedder: PluginInstance
    index: EmbeddingIndex

    def __init__(self, client: Steamship, **kwargs):
        super().__init__(client, **kwargs)

        # First we create an embedder. For this example package we'll use
        # OpenAI's Davinci 001
        self.embedder = self.client.use_plugin(
            "openai-embedder",
            config={"model": "text-similarity-davinci-001", "dimensionality": 12288},
        )

        self.index = EmbeddingIndex.create(
            client=self.client,
            handle="my-qa-index",
            plugin_instance=self.embedder.handle,
            fetch_if_exists=True,
        )

    @post("learn")
    def learn(self, fact: str = None, metadata: Dict = None) -> IndexInsertResponse:
        """Learns a new fact."""
        if fact is None:
            raise SteamshipError(message="Empty fact provided to learn.")

        # Reindex is usually good to call right away -- this will make sure that the embedding
        # gets created.
        res = self.index.insert(fact, metadata=metadata, reindex=True)

        # This is also good to do as it will help your index scale. This creates an AKNN
        # structure on disk, as opposed to the KNN structure that would have otherwise been used.
        self.index.create_snapshot()

        return res

    @post("query")
    def query(self, query: str = None, k: int = 1) -> QueryResults:
        """Learns a new fact."""
        if query is None:
            raise SteamshipError(message="Empty query provided.")

        res = self.index.search(query=query, k=k, include_metadata=True)
        res.wait()
        return res.output


handler = create_handler(QuestionAnsweringPackage)
