# Databricks notebook source
# MAGIC %pip install -U -qqqq backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://docs.databricks.com/mlflow3/genai/eval-monitor)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.
# MAGIC
# MAGIC Evaluate your agent with one of our [predefined LLM scorers](https://docs.databricks.com/mlflow3/genai/eval-monitor/predefined-judge-scorers), or try adding [custom metrics](https://docs.databricks.com/mlflow3/genai/eval-monitor/custom-scorers).

# COMMAND ----------

# MAGIC %md
# MAGIC ##Now lets use our synthetic eval data set
# MAGIC

# COMMAND ----------

mlflow.set_experiment("/Users/ben.mackenzie@databricks.com/rag_agent")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from benmackenzie_catalog.mlflow3.gisa_manual_evalaution_s

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety, RetrievalRelevance, RetrievalGroundedness, Correctness

# COMMAND ----------

# MAGIC %md
# MAGIC ##Skip next cell if already complete

# COMMAND ----------

eval_dataset = mlflow.genai.create_dataset('benmackenzie_catalog.mlflow3.gisa_manual_evalaution_ds')
data = spark.table('benmackenzie_catalog.mlflow3.gisa_manual_evalaution_s')
eval_dataset = eval_dataset.merge_records(data) #does not update in place

# COMMAND ----------

# MAGIC %md
# MAGIC ##Nota Bene
# MAGIC
# MAGIC Schema in dataset contains additional info

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from benmackenzie_catalog.mlflow3.gisa_manual_evalaution_ds;

# COMMAND ----------

import mlflow

catalog = "benmackenzie_catalog"
schema = "mlflow3"
model_name = "gisa_rag_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

model = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}/1")

# COMMAND ----------

# we need to convert formats to new ReponseAgent.

#eval data data is in this format:

{"messages": [{"content": "What was the former name of Kemper Canada?", "role": "user"}]}

#model wants this format:

{"input": [{"role": "user", "content": "Hello!"}]}

#basially we need to swamp "messages" for "input"


# COMMAND ----------



with mlflow.start_run(run_name="eval on dataset 1"):
    eval_results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=lambda messages: model.predict({"input": messages}),
        scorers=[RelevanceToQuery(), Safety(), Correctness(), RetrievalRelevance(), RetrievalGroundedness()]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ###Alternative to inline

# COMMAND ----------

def predict_fn(messages):
    # If messages is a numpy array, convert to list
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    print({"input": messages})
    return model.predict({"input": messages})

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=[RelevanceToQuery(), Safety(), Correctness(), RetrievalRelevance(), RetrievalGroundedness()]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Human Feedback

# COMMAND ----------

import mlflow
from mlflow.genai.label_schemas import create_label_schema, InputCategorical, InputText
from mlflow.genai.labeling import create_labeling_session

# Define what feedback to collect
accuracy_schema = create_label_schema(
    name="response_accuracy",
    type="feedback",
    title="Is the response factually accurate?",
    input=InputCategorical(options=["Accurate", "Partially Accurate", "Inaccurate"]),
    overwrite=True
)

ideal_response_schema = create_label_schema(
    name="expected_response",
    type="expectation",
    title="What would be the ideal response?",
    input=InputText(),
    overwrite=True
)

# Create a labeling session
labeling_session = create_labeling_session(
    name="quickstart_review",
    label_schemas=[accuracy_schema.name, ideal_response_schema.name],
)

# Add your trace to the session
# Get the most recent trace from the current experiment
traces = mlflow.search_traces(
    max_results=5  # Gets the most recent trace
)
labeling_session.add_traces(traces)

# Share with reviewers
print(f"✅ Trace sent for review!")
print(f"Share this link with reviewers: {labeling_session.url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create eval dataset from traces

# COMMAND ----------

import mlflow
import mlflow.genai.datasets
import time


# 1. Create an evaluation dataset



eval_dataset = mlflow.genai.datasets.create_dataset(
    uc_table_name=f"benmackenzie_catalog.mlflow3.gisa_manual_evalaution_ds_2",
)


# 2. Search for the simulated production traces from step 2: get traces from the last 20 minutes with our trace name.
thirty_minutes_ago = int((time.time() - 30 * 60) * 1000)

traces = mlflow.search_traces(
    filter_string=f"attributes.timestamp_ms > {thirty_minutes_ago} AND attributes.status = 'OK' ",
    order_by=["attributes.timestamp_ms DESC"]
)

print(f"Found {len(traces)} successful traces from beta test")

# 3. Add the traces to the evaluation dataset
eval_dataset = eval_dataset.merge_records(traces)
print(f"Added {len(traces)} records to evaluation dataset")

# Preview the dataset
df = eval_dataset.to_df()
print(f"\nDataset preview:")
print(f"Total records: {len(df)}")
print("\nSample record:")
sample = df.iloc[0]
print(f"Inputs: {sample['inputs']}")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from benmackenzie_catalog.mlflow3.gisa_manual_evalaution_ds_2
# MAGIC

# COMMAND ----------

df = eval_dataset.to_df()

# COMMAND ----------

display(df)
