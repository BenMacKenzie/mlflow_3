# Databricks notebook source
# MAGIC %pip install -U databricks-agents mlflow 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

tables_config = config['data']['tables_config']

parsed_docs_table_name = tables_config['parsed_docs_table_name']

EVALUATION_SET_FQN =  config['eval']['synthetic_evaluation_set_fqn']

# COMMAND ----------

EVALUATION_SET_FQN

# COMMAND ----------

# MAGIC %sql
# MAGIC select path as doc_uri, CAST(pages AS STRING) as content from benmackenzie_catalog.mlflow3.parsed_gisa_document

# COMMAND ----------

docs = spark.sql("select path as doc_uri, CAST(pages AS STRING) as content from benmackenzie_catalog.mlflow3.parsed_gisa_document")

# COMMAND ----------


from databricks.agents.evals import generate_evals_df
import mlflow

agent_description = """A chatbot that answers questions about Automobile Statistical Plan Manual Including Facility Association Underwriting Information Plan."""

question_guidelines = """
# User personas
- An underwriting agent or analyst for insurance company
# Example questions
- What is Earned Exposure?
"""

# TODO: Spark/Pandas DataFrame with "content" and "doc_uri" columns.

evals = generate_evals_df(
    docs=docs,
    num_evals=5,
    agent_description=agent_description,
    question_guidelines=question_guidelines,
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Save data set

# COMMAND ----------

evals_spark = spark.createDataFrame(evals)
evals_spark.write.format("delta").mode("overwrite").saveAsTable(EVALUATION_SET_FQN)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from benmackenzie_catalog.mlflow3.gisa_manual_evalaution_s
