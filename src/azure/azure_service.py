import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from src.utils.logger import create_logger

logger = create_logger("azure_service", "logs/azure_service.log", propagate=False)


class AzureService:
    def __init__(self, config: dict):
        self.config = config
        self.resource_group = self.config["azure"]["resource_group"]
        self.workspace_name = self.config["azure"]["workspace_name"]
        load_dotenv()
        self.subscription_id = os.environ["SUBSCRIPTION_ID"]
        self._initialize_service()
        self.registered_models = self.get_registered_models()

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
            logger.info(
                f"[REGISTER_MODEL] Model '{model_name}' registered successfully."
            )

        except Exception as e:
            logger.error(
                f"[REGISTER_MODEL] Failed to register model '{model_name}': {e}"
            )
            raise

    def get_registered_models(self) -> list[Model]:
        """Retrieve a list of registered models in Azure ML workspace."""

        try:
            models = list(self.client.models.list())
            logger.info(
                f"[GET_REGISTERED_MODELS] Found {len(models)} model(s) in Azure ML workspace '{self.workspace_name}'."
            )

            for model in models:
                logger.info(
                    f"[GET_REGISTERED_MODELS] Model Name: {model.name}, Latest Version: {model.latest_version}, ID: {model.id}, Created On: {model.creation_context.created_at}, Description: {model.description}"
                )
            return models

        except Exception as e:
            logger.error(
                f"[GET_REGISTERED_MODELS] Failed to retrieve registered models: {e}"
            )
            raise

    def download_model(
        self, model_name: str, model_version: int, download_path: str = "models"
    ) -> str:
        """Download a registered model from Azure ML workspace."""

        try:
            logger.info(
                f"Downloading model '{model_name}' version '{model_version}' to '{download_path}'."
            )

            # Create model directory
            os.makedirs(download_path, exist_ok=True)

            # Retrieve the model from Azure ML workspace
            self.client.models.download(
                name=model_name,
                version=model_version,
                download_path=download_path,
            )
            logger.info(
                f"[DOWNLOAD_MODEL] Model '{model_name}' version '{model_version}' downloaded successfully to '{download_path}'."
            )

            return f"{download_path}/{model_name}/best.pt"

        except Exception as e:
            logger.error(
                f"[DOWNLOAD_MODEL] Failed to download model '{model_name}' version '{model_version}': {e}"
            )
            raise
