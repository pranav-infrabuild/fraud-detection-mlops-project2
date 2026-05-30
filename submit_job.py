from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

subscription_id = "bdddc0ca-5595-41da-ba62-1e19e79fef55"  # Apni ID daalo
resource_group = "mlops-rg"
workspace_name = "fraud-mlops-ws"

ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)

job = command(
    code="./",
    command="python src/train_azure.py",  # Naya script
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
    compute="cpu-cluster",
    display_name="fraud-training-simple",
)

submitted_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted! Name: {submitted_job.name}")
print(f"View: {submitted_job.studio_url}")
