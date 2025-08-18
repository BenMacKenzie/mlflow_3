from typing import Any, Generator, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

mlflow.openai.autolog()

# Optional: Replace with any model serving endpoint
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"


class SimpleChatAgent(ChatAgent):
    def __init__(self):
        self.workspace_client = WorkspaceClient()
        self.client = self.workspace_client.serving_endpoints.get_open_ai_client()
        self.llm_endpoint = LLM_ENDPOINT_NAME

        # Fake documents to simulate the retriever
        self.documents = [
            mlflow.entities.Document(
                metadata={"doc_uri": "uri1.txt"},
                page_content="""Lakehouse Monitoring for GenAI helps you monitor the quality, cost, and latency of production GenAI apps.  Lakehouse Monitoring for GenAI allows you to:\n- Track quality and operational performance (latency, request volume, errors, etc.).\n- Run LLM-based evaluations on production traffic to detect drift or regressions using Agent Evaluation's LLM judges.\n- Deep dive into individual requests to debug and improve agent responses.\n- Transform real-world logs into evaluation sets to drive continuous improvements.""",
            ),
            # This is a new document about spark.
            mlflow.entities.Document(
                metadata={"doc_uri": "uri2.txt"},
                page_content="The latest spark version in databricks in 3.5.0",
            ),
        ]

        # Tell Agent Evaluation's judges and review app about the schema of your retrieved documents
        mlflow.models.set_retriever_schema(
            name="fake_vector_search",
            primary_key="doc_uri",
            text_column="page_content",
            doc_uri="doc_uri"
            # other_columns=["column1", "column2"],
        )

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    def dummy_retriever(self):
      # Fake retriever
      return self.documents
  

    def prepare_messages_for_llm(
        self, messages: list[ChatAgentMessage]
    ) -> list[dict[str, Any]]:
        """Filter out ChatAgentMessage fields that are not compatible with LLM message formats"""
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        return [
            {
                k: v
                for k, v in m.model_dump_compat(exclude_none=True).items()
                if k in compatible_keys
            }
            for m in messages
        ]

    @mlflow.trace(span_type=SpanType.PARSER)
    def prepare_rag_prompt(self, messages):

        docs = self.dummy_retriever()

        messages = self.prepare_messages_for_llm(messages)

        messages[-1]['content'] = f"Answer the user's question based on the documents.\nDocuments: <documents>{docs}</documents>.\nUser's question: <user_question>{messages[-1]['content']}</user_question>"

        return messages

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        
        messages = self.prepare_rag_prompt(messages)

        resp = self.client.chat.completions.create(
            model=self.llm_endpoint,
            messages=messages,
        )

        return ChatAgentResponse(
            messages=[
                ChatAgentMessage(**resp.choices[0].message.to_dict(), id=resp.id)
            ],
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        
        messages = self.prepare_rag_prompt(messages)

        for chunk in self.client.chat.completions.create(
            model=self.llm_endpoint,
            messages=messages,
            stream=True,
        ):
            if not chunk.choices or not chunk.choices[0].delta.content:
                continue

            yield ChatAgentChunk(
                delta=ChatAgentMessage(
                    **{
                        "role": "assistant",
                        "content": chunk.choices[0].delta.content,
                        "id": chunk.id,
                    }
                )
            )


from mlflow.models import set_model

AGENT = SimpleChatAgent()
set_model(AGENT)
