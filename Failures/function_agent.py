import mlflow
from openai import OpenAI

# Enable automatic tracing for all OpenAI API calls
#mlflow.openai.autolog()

# Connect to a Databricks LLM via OpenAI using the same credentials as MLflow
# Alternatively, you can use your own OpenAI credentials here
mlflow_creds = mlflow.utils.databricks_utils.get_databricks_host_creds()
client = OpenAI(
    api_key=mlflow_creds.token,
    base_url=f"{mlflow_creds.host}/serving-endpoints"
)

@mlflow.trace
def predict(model_input: list[str]) -> list[str]:
    return [my_chatbot(question) for question in model_input]

# Create a RAG app with tracing
@mlflow.trace
def my_chatbot(user_question: str) -> str:
    # Retrieve relevant contexr
        
    context = retrieve_context(user_question)

    # Generate response using LLM with retrieved context
    response = client.chat.completions.create(
        model="databricks-llama-4-maverick",  # If using OpenAI directly, use "gpt-4o" or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_question}"}
        ],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content

@mlflow.trace(span_type="RETRIEVER")
def retrieve_context(query: str) -> str:
    # Simulated retrieval - in production, this would search a vector database
    if "mlflow" in query.lower():
        return "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It provides tools for experiment tracking, model packaging, and deployment."
    return "General information about machine learning and data science."

mlflow.models.set_model(predict)
