from typing import Dict, Type

from steamship.invocable import Config, post, PackageService
from steamship_langchain.cache import SteamshipCache
from steamship_langchain.llms import OpenAI
from steamship_langchain.vectorstores import SteamshipVectorStore

import langchain
from langchain.chains import VectorDBQAWithSourcesChain


class QuestionAnswering(PackageService):

    class QuestionAnsweringConfig(Config):
        index_name: str

    config: QuestionAnsweringConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        langchain.verbose = True
        langchain.llm_cache = SteamshipCache(self.client)
        self._index = SteamshipVectorStore(client=self.client,
                                           index_name=self.config.index_name,
                                           embedding="text-embedding-ada-002")

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return cls.QuestionAnsweringConfig

    @post("answer", public=True)
    def answer(self, question: str, k: int = 4) -> Dict[str, str]:
        chain = VectorDBQAWithSourcesChain.from_chain_type(
            OpenAI(client=self.client, temperature=0),
            chain_type="stuff",
            vectorstore=self._index,
            return_source_documents=True,
            verbose=True,
            k=k,
        )
        return chain({"question": question})
