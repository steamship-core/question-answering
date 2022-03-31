from typing import Dict
from steamship import Steamship
from steamship.app import App, Response, Error, post, create_handler

class QuestionAnswer(App):
  def __init__(self, client: Steamship):
    # In production, the lambda handler will provide a Steamship client:
    # - Authenticated to the appropriate user
    # - Bound to the appropriate space
    self.client = client

    # Create an embedding index using (for now!) our
    # mock embedder.
    #
    # Note that the *scope* of this index is limited to the space
    # this app is executing within. Each new instance of the app
    # will resultingly have a fresh index.
    self.index = self.client.create_index(
      handle="qa-index",
      plugin="test-embedder-v1"     
    ).data
  
  @post('learn')
  def learn(self, fact: str = None) -> Response:
    """Learns a new fact."""
    if fact is None:
      return Error(message="Empty fact provided to learn.")
   
    if self.index is None:
      return Error(message="Unable to initialize QA index.")

    res = self.index.insert(fact, reindex=True)

    if res.error:
      # Steamship error messages can be passed straight
      # back to the user
      return Error(
        message = res.error.message,
        suggestion = res.error.suggestion,
        code = res.error.code
      )
    
    return Response(json=res.data)

  @post('query')
  def query(self, query: str = None, k: int = 1) -> Response:
    """Learns a new fact."""
    if query is None:
      return Error(message="Empty query provided.")
    
    if self.index is None:
      return Error(message="Unable to initialize QA index.")

    res = self.index.search(query=query, k=k)

    if res.error:
      # Steamship error messages can be passed straight
      # back to the user
      return Error(
        message = res.error.message,
        suggestion = res.error.suggestion,
        code = res.error.code
      )
    return Response(json=res.data)


handler = create_handler(QuestionAnswer)



