# Databricks notebook source
# MAGIC %md 
# MAGIC # Mosaic AI Agent Framework: Author and deploy a simple OpenAI agent
# MAGIC
# MAGIC This notebook demonstrates how to author a OpenAI agent that's compatible with Mosaic AI Agent Framework features. In this notebook you learn to:
# MAGIC - Author a OpenAI agent with `ChatAgent`
# MAGIC - Manually test the agent's output
# MAGIC - Log and deploy the agent
# MAGIC
# MAGIC To learn more about authoring an agent using Mosaic AI Agent Framework, see Databricks documentation ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/author-agent) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/create-chat-model)).
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq openai databricks-agents==0.21.1 uv databricks-vectorsearch mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Define the agent in code
# MAGIC Define the agent code in a single cell below. This lets you easily write the agent code to a local Python file, using the `%%writefile` magic command, for subsequent logging and deployment.
# MAGIC

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC from typing import Any, Generator, Optional
# MAGIC import os
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc.model import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC
# MAGIC
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC
# MAGIC mlflow.openai.autolog()
# MAGIC
# MAGIC # TODO: Replace with your model serving endpoint
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC
# MAGIC
# MAGIC VECTOR_ENDPOINT_NAME = "dbdemos_vs_endpoint"
# MAGIC VECTOR_INDEX_NAME = "main.morley_demo.databricks_documentation_vs_index"
# MAGIC WORKSPACE_URL = "https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints"
# MAGIC
# MAGIC
# MAGIC class SimpleChatAgent(ChatAgent):
# MAGIC     def __init__(self):
# MAGIC         self.workspace_client = WorkspaceClient()
# MAGIC         self.client = self.workspace_client.serving_endpoints.get_open_ai_client()
# MAGIC         self.llm_endpoint = LLM_ENDPOINT_NAME
# MAGIC
# MAGIC         vsc = VectorSearchClient(
# MAGIC         )
# MAGIC         self.vector_idx = vsc.get_index(
# MAGIC             endpoint_name=VECTOR_ENDPOINT_NAME,
# MAGIC             index_name=VECTOR_INDEX_NAME,
# MAGIC         )
# MAGIC     
# MAGIC     def prepare_messages_for_llm(self, messages: list[ChatAgentMessage]) -> list[dict[str, Any]]:
# MAGIC         """Filter out ChatAgentMessage fields that are not compatible with LLM message formats"""
# MAGIC         compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
# MAGIC         return [
# MAGIC             {k: v for k, v in m.model_dump_compat(exclude_none=True).items() if k in compatible_keys} for m in messages
# MAGIC         ]
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.AGENT)
# MAGIC     def retrieve_context(self, question):
# MAGIC         results = self.vector_idx.similarity_search(
# MAGIC             query_text=question,
# MAGIC             columns=["id", "url", "content"],
# MAGIC             num_results=10,
# MAGIC             query_type="hybrid",
# MAGIC             # score_threshold=0.2,
# MAGIC             # filters=filters,
# MAGIC         )
# MAGIC         result_list = [
# MAGIC             {"url": row[1], "content": row[2], "score": row[3]}
# MAGIC             for row in results["result"]["data_array"]
# MAGIC         ]
# MAGIC
# MAGIC         return result_list
# MAGIC     
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.AGENT)
# MAGIC     def augment_query(self, question, question_context):
# MAGIC         return f"""
# MAGIC
# MAGIC You are a question answering bot. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
# MAGIC
# MAGIC ======================================================================
# MAGIC
# MAGIC Context:{'''
# MAGIC ----------------------------------------------------------------------
# MAGIC '''.join([c['content'] for c in question_context])}.
# MAGIC ======================================================================
# MAGIC
# MAGIC Question: {question}
# MAGIC
# MAGIC """
# MAGIC
# MAGIC
# MAGIC     def predict_common(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         
# MAGIC         question_context = self.retrieve_context(question=messages[-1].content)
# MAGIC         augmented_query = self.augment_query(
# MAGIC             question=messages[-1].content, question_context=question_context
# MAGIC         )
# MAGIC         messages[-1].content = augmented_query
# MAGIC
# MAGIC         return messages
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.AGENT)
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         
# MAGIC         messages = self.predict_common(messages, context, custom_inputs)
# MAGIC
# MAGIC         resp = self.client.chat.completions.create(
# MAGIC             model=self.llm_endpoint,
# MAGIC             messages=self.prepare_messages_for_llm(messages),
# MAGIC         )
# MAGIC
# MAGIC         return ChatAgentResponse(
# MAGIC             messages=[ChatAgentMessage(**resp.choices[0].message.to_dict(), id=resp.id)],
# MAGIC         )
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.AGENT)
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         
# MAGIC         messages = self.predict_common(messages, context, custom_inputs)
# MAGIC
# MAGIC         for chunk in self.client.chat.completions.create(
# MAGIC             model=self.llm_endpoint,
# MAGIC             messages=self.prepare_messages_for_llm(messages),
# MAGIC             stream=True,
# MAGIC         ):
# MAGIC             if not chunk.choices or not chunk.choices[0].delta.content:
# MAGIC                 continue
# MAGIC
# MAGIC             yield ChatAgentChunk(
# MAGIC                 delta=ChatAgentMessage(
# MAGIC                     **{
# MAGIC                         "role": "assistant",
# MAGIC                         "content": chunk.choices[0].delta.content,
# MAGIC                         "id": chunk.id,
# MAGIC                     }
# MAGIC                 )
# MAGIC             )
# MAGIC
# MAGIC
# MAGIC from mlflow.models import set_model
# MAGIC
# MAGIC AGENT = SimpleChatAgent()
# MAGIC set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. 
# MAGIC
# MAGIC Since you manually traced methods within `ChatAgent`, you can view the trace for each step the agent takes, with any LLM calls made via the OpenAI SDK automatically traced by autologging.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT

# COMMAND ----------

AGENT.predict({"messages": [{"role": "user", "content": "How do I get the current timezone in spark sql?"}]})

# COMMAND ----------

for event in AGENT.predict_stream(
    {"messages": [{"role": "user", "content": "How do I get the current timezone in spark sql?"}]}
):
    print(event, "-----------\n")


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

import mlflow
from agent import LLM_ENDPOINT_NAME, VECTOR_INDEX_NAME
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex
from pkg_resources import get_distribution

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        extra_pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"databricks-vectorsearch=={get_distribution('databricks-vectorsearch').version}",
            f"openai=={get_distribution('openai').version}",
            f"mlflow=={get_distribution('mlflow').version}",
        ],
        resources=[
            DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
            DatabricksVectorSearchIndex(index_name=VECTOR_INDEX_NAME),
        ],
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-debug.html#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks)).

# COMMAND ----------

# mlflow.models.predict(
#     model_uri=f"runs:/{logged_agent_info.run_id}/agent",
#     input_data={"messages": [{"role": "user", "content": "Hello!"}]},
#     env_manager="uv",
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Before you deploy the agent, you must register the agent to Unity Catalog.
# MAGIC
# MAGIC - **TODO** Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "bryn"
schema = "nos"
model_name = "openai-pyfunc-rag"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags={"endpointSource": "docs"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See Databricks documentation ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)).
