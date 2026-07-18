import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from src.utils.logger import azure_logger as logger


class AzureService:
    def __init__(self, config: dict):
        self.config = config
        self.resource_group = self.config["azure"]["resource_group"]
        self.workspace_name = self.config["azure"]["workspace_name"]
        load_dotenv()
        self.subscription_id = os.environ["SUBSCRIPTION_ID"]
        self._initialize_service()

    def _initialize_service(self) -> None:
        # Initialize Azure service based on the provided configuration
        logger.info(
            f"Initializing Azure service with the provided configuration: {self.config}"
        )

        if not self.subscription_id:
            logger.error(
                "Azure subscription ID is not set in the environment variables."
            )
            raise ValueError("Azure subscription ID is required.")

        if not self.resource_group or not self.workspace_name:
            logger.error(
                "Azure resource group or workspace name is missing in the configuration."
            )
            raise ValueError("Azure resource group and workspace name are required.")

        self.client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name,
        )

        # Verify the connection to the Azure ML workspace
        try:
            workspace = self.client.workspaces.get(self.workspace_name)

            logger.info(
                f"Successfully connected to Azure ML workspace " f"'{workspace.name}'."
            )

        except ResourceNotFoundError as e:
            logger.error(
                f"Workspace '{self.workspace_name}' not found "
                f"in resource group '{self.resource_group}'."
            )
            raise e

        except ClientAuthenticationError as e:
            logger.error(
                "Authentication with Azure failed. "
                "Run 'az login' or check your credentials."
            )
            raise e

        except Exception as e:
            logger.error(f"Unable to connect to Azure ML: {e}")
            raise

    def register_model(self, model_name: str, model_path: str):
        """Register a model in Azure ML workspace."""

        try:
            logger.info(
                f"Registering model '{model_name}' from path '{model_path}' in Azure ML workspace."
            )
            model = Model(
                name=model_name,
                path=model_path,
                description=f"Model registered from {model_path}",
            )
            self.client.models.create_or_update(model)
            logger.info(f"[SUCCESS] Model '{model_name}' registered successfully.")

        except Exception as e:
            logger.error(f"[ERROR] Failed to register model '{model_name}': {e}")
            raise
