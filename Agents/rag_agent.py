
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

# TODO: Replace with your model serving endpoint
LLM_ENDPOINT_NAME = "databricks-llama-4-maverick"


class SimpleChatAgent(ChatAgent):
    def __init__(self):
        self.workspace_client = WorkspaceClient()
        self.client = self.workspace_client.serving_endpoints.get_open_ai_client()
        self.llm_endpoint = LLM_ENDPOINT_NAME
    
    def prepare_messages_for_llm(self, messages: list[ChatAgentMessage]) -> list[dict[str, Any]]:
        """Filter out ChatAgentMessage fields that are not compatible with LLM message formats"""
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        m = [
            {k: v for k, v in m.model_dump_compat(exclude_none=True).items() if k in compatible_keys} for m in messages
        ]

        print(m)
        return m

    @mlflow.trace(span_type=SpanType.AGENT)
    def retrieve_context(self, question):
        if "mlflow" in question.lower():
            return ["MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides tools for experiment tracking, model packaging, and deployment."]
        else:
            return ["General information about machine learning and data science."]

    @mlflow.trace(span_type=SpanType.AGENT)
    def augment_query(self, question, question_context):
        return f"""

    You are a question answering bot. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    ======================================================================

    Context:{'''
    ----------------------------------------------------------------------
    '''.join([c for c in question_context])}.
    ======================================================================

    Question: {question}

    """

    def predict_common(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        
        question_context = self.retrieve_context(question=messages[-1].content)
        augmented_query = self.augment_query(
            question=messages[-1].content, question_context=question_context
        )
        messages[-1].content = augmented_query

        return messages

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        
        messages = self.predict_common(messages, context, custom_inputs)

        resp = self.client.chat.completions.create(
            model=self.llm_endpoint,
            messages=self.prepare_messages_for_llm(messages),
        )

        return ChatAgentResponse(
            messages=[ChatAgentMessage(**resp.choices[0].message.to_dict(), id=resp.id)],
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        
        messages = self.predict_common(messages, context, custom_inputs)

        for chunk in self.client.chat.completions.create(
            model=self.llm_endpoint,
            messages=self.prepare_messages_for_llm(messages),
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
