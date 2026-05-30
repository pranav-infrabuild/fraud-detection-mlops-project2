from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Replace with your details
subscription_id = "bdddc0ca-5595-41da-ba62-1e19e79fef55"
resource_group = "mlops-rg"
workspace_name = "fraud-mlops-ws"

# Connect
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name
)

print(f"Connected to workspace: {ml_client.workspace_name}")
