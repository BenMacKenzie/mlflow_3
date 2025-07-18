
## Quickstart.

Examples illustrating agent evaluation and human feedback.


## Agent_Examples

Examples illustrating agent evaluation on 'full' agents that can be registered in Unity Catalog (the simple functions in Quickstart cannot be registered..or at least I can't figure out how to get the signature right.)


## Traditional ML with Deployment Job

Illustrates the asscociating a deployment job with a model




Notes

1. use of traces, evaluate and human feedback works well.
2. new model-centric data model / UI works well.
3. it's pretty complex to author an agent by hand.  deterministic agents seem to be deprecated.    See https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent
4. bug in datasets requires to to re-read prior to using in mlflow.genai.evaluate
5. need to set experiment explicitly for feedback session to work if notebook is in a git folder.


To Do:

1. are traces written to experiment with/without inference talbes enabled?  This is in flight..everything is moving to dedicated servers.
2. illustrate searching traces
3. illustrate versioning of eval table
4. Use better scorers in examples