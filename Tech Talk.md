##Experiment UI
 - Type: Genai or traditional ml
 - Runs.  Anything that happens within a run: Logging a model, evaluating a model.  Running prediction outside of an evaluation does not generate a run.
 - Traces.  For genai applications
 - Evaluations.   Note dataset.
 - Labelling
 - Versions.
 - Prompts
 - Scorers
 - Delta Sync button

## 01 Agent Eval






##Gotchas

 - Experiment does not refresh consistently if you change it.
 - Trace UI does not display input properly
 - Dataset.merge returns a new dataset, does not update in place
 - Production scorers not working?
 - latency in populating inference tables
 - resource limits and time outs.
 - schemas don't show up in UI
 
