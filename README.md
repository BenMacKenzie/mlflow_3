# References

https://docs.databricks.com/aws/en/mlflow3/genai/




## 01 Agent Eval.

Examples illustrating agent evaluation and human feedback.


## 02 RAG Agent Example

Example from playground export with eval and vector index creation code.

#### To Do

Update vector index with Scott's version (in sql)

## 03 Other Agent Examples

Difference between ChatAgent and ResponseAgent


## Traditional ML with Deployment Job

Illustrates the asscociating a deployment job with a model




## Notes



## To Do:

1. are traces written to experiment with/without inference talbes enabled?  This is in flight..everything is moving to dedicated servers.
2. illustrate searching traces
3. illustrate versioning of eval table
4. Use better scorers in examples
5. illustrate difference between running eval with a logged model vs model that is only in memory.  illustrate running evals inside a mlflow run
6. prompt registry
7. what is a data set object?
8. autolog vs annotation
9. what does this page mean? https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing.  And following? https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing#log-traces-to-tables
I think monitoring tab is disabled to archive to delta table follow: https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring#archive-traces


assessments vs scorers
create labeleling session through UI
create eval data set using UI



