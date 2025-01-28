import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class ModelMetadata:
    """Metadata for model versioning"""
    model_type: str  # e.g., 'forecasting', 'causal', 'rl'
    experiment_id: str
    performance_metrics: Dict[str, float]
    training_dataset: str
    model_parameters: Dict[str, Any]
    created_by: str
    
class MLflowRegistryManager:
    """Manager class for MLflow Model Registry operations"""
    
    def __init__(
        self,
        tracking_uri: str,
        registry_uri: str,
        base_model_name: str
    ):
        self.client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
        mlflow.set_tracking_uri(tracking_uri)
        self.base_model_name = base_model_name
        self._setup_logging()
        
    def _setup_logging(self):
        """
        Configures logging for the registry manager.

        Sets up the logging system to output logs with a specific format and logging level.

        Args:
            None

        Returns:
            None
        """
        logging.basicConfig(
            level=logging.INFO,  # Sets the logging level to INFO 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Defines the log message format.
            # Example format: "2025-01-28 10:00:00 - registry_manager - INFO - Log message"
        )

    def register_model(
        self,
        run_id: str,
        metadata: ModelMetadata,
        artifact_path: str = "model"
    ) -> str:
        """
        Register a new model version with metadata
        Returns: version number of the newly registered model
        """
        try:
            # Register the model
            model_version = self.client.create_model_version(
                name=self.base_model_name,
                source=f"runs:/{run_id}/{artifact_path}",
                run_id=run_id
            )
            
            # Add metadata as tags
            self._add_version_metadata(model_version.version, metadata)
            
            self.logger.info(
                f"Successfully registered model {self.base_model_name} "
                f"version {model_version.version}"
            )
            
            return model_version.version
            
        except Exception as e:
            self.logger.error(f"Error registering model: {str(e)}")
            raise

    def _add_version_metadata(self, version: str, metadata: ModelMetadata):
        """Add metadata to a model version as tags"""
        tags = {
            "model_type": metadata.model_type,
            "experiment_id": metadata.experiment_id,
            "training_dataset": metadata.training_dataset,
            "created_by": metadata.created_by,
            "creation_timestamp": datetime.now().isoformat(),
            "model_parameters": str(metadata.model_parameters),
        }
        
        # Add performance metrics
        for metric_name, metric_value in metadata.performance_metrics.items():
            tags[f"metric_{metric_name}"] = str(metric_value)
            
        for key, value in tags.items():
            self.client.set_model_version_tag(
                name=self.base_model_name,
                version=version,
                key=key,
                value=value
            )

    def transition_stage(
        self,
        version: str,
        stage: str,
        archive_existing: bool = True
    ):
        """
        Transition a model version to a new stage
        Stages: 'None', 'Staging', 'Production', 'Archived'
        """
        if stage not in ['None', 'Staging', 'Production', 'Archived']:
            raise ValueError(f"Invalid stage: {stage}")
            
        try:
            # If moving to production/staging and archive_existing is True,
            # archive existing models in that stage
            if archive_existing and stage in ['Production', 'Staging']:
                self._archive_existing_versions(stage)
                
            self.client.transition_model_version_stage(
                name=self.base_model_name,
                version=version,
                stage=stage
            )
            
            self.logger.info(
                f"Transitioned model {self.base_model_name} version {version} "
                f"to stage: {stage}"
            )
            
        except Exception as e:
            self.logger.error(f"Error transitioning model stage: {str(e)}")
            raise

    def _archive_existing_versions(self, stage: str):
        """Archive existing models in the specified stage"""
        versions = self.client.search_model_versions(f"name='{self.base_model_name}'")
        for version in versions:
            if version.current_stage == stage:
                self.client.transition_model_version_stage(
                    name=self.base_model_name,
                    version=version.version,
                    stage="Archived"
                )

    def get_latest_versions(self, stages: Optional[List[str]] = None) -> List[Dict]:
        """Get latest model versions, optionally filtered by stages"""
        versions = self.client.search_model_versions(f"name='{self.base_model_name}'")
        
        if stages:
            versions = [v for v in versions if v.current_stage in stages]
            
        return [{
            'version': v.version,
            'stage': v.current_stage,
            'run_id': v.run_id,
            'tags': self.client.get_model_version_tags(self.base_model_name, v.version)
        } for v in versions]

# Example usage
if __name__ == "__main__":
    # Initialize the registry manager
    registry_manager = MLflowRegistryManager(
        tracking_uri="sqlite:///mlflow.db",
        registry_uri="sqlite:///mlflow.db",
        base_model_name="forecasting_model"
    )
    
    # Example metadata for a new model version
    metadata = ModelMetadata(
        model_type="forecasting",
        experiment_id="exp_123",
        performance_metrics={
            "mape": 0.15,
            "rmse": 0.08,
            "mae": 0.12
        },
        training_dataset="sales_data_2024_q1",
        model_parameters={
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.01
        },
        created_by="data_scientist_1"
    )
    
    # Register a new model version
    with mlflow.start_run() as run:
        # training code here
        # ...
        
        # Log the model
        mlflow.sklearn.log_model(
            sk_model='your_model',
            artifact_path="model"
        )
        
        # Register the model with metadata
        version = registry_manager.register_model(
            run_id=run.info.run_id,
            metadata=metadata
        )
        
        # Transition to staging
        registry_manager.transition_stage(version, "Staging")
        
        # After validation, transition to production
        registry_manager.transition_stage(version, "Production")
        
    # Get latest versions
    latest_versions = registry_manager.get_latest_versions(
        stages=["Production", "Staging"]
    )
