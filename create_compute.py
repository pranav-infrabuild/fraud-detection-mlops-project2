from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

# Your workspace details
subscription_id = "bdddc0ca-5595-41da-ba62-1e19e79fef55"
resource_group = "mlops-rg"
workspace_name = "fraud-mlops-ws"

ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)

# Create compute cluster
compute_name = "cpu-cluster"

try:
    compute = ml_client.compute.get(compute_name)
    print(f"Compute '{compute_name}' already exists")
except:
    print(f"Creating compute '{compute_name}'...")
    compute = AmlCompute(
        name=compute_name,
        size="STANDARD_DS3_V2",  # 4 cores, 14GB RAM
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=120
    )
    ml_client.compute.begin_create_or_update(compute).result()
    print("Compute created!")
