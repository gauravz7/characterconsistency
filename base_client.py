"""
Base client for Google Cloud Vertex AI Vision services.
Provides common functionality for all vision-related operations.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

import vertexai
from google.cloud import storage
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError


class BaseVisionClient:
    """Base class for all Google Cloud Vision AI clients."""
    
    def __init__(self, config_path: Optional[str] = None, project_id: Optional[str] = None, 
                 location: Optional[str] = None, use_veo_config: bool = False):
        """
        Initialize the base vision client.
        
        Args:
            config_path: Path to configuration YAML file
            project_id: Google Cloud project ID (overrides config)
            location: Google Cloud location (overrides config)
            use_veo_config: Use Veo-specific configuration instead of default
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Determine which config section to use
        config_section = "veo" if use_veo_config else "project"
        
        # Set project and location with precedence: parameter > env > config
        self.project_id = (
            project_id or 
            os.getenv("GOOGLE_CLOUD_PROJECT") or 
            self.config[config_section]["default_project_id"]
        )
        
        self.location = (
            location or 
            os.getenv("GOOGLE_CLOUD_LOCATION") or 
            self.config[config_section]["default_location"]
        )
        
        self.gcs_bucket = self.config[config_section]["default_gcs_bucket"]
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.logger.info(f"Initialized Vertex AI for project: {self.project_id}, location: {self.location}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
        
        # Initialize Google Cloud Storage client
        try:
            self.storage_client = storage.Client(project=self.project_id)
        except DefaultCredentialsError as e:
            self.logger.error(f"Authentication failed: {e}")
            raise
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(self.__class__.__name__)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        log_level = self.config.get("logging", {}).get("level", "INFO")
        logger.setLevel(getattr(logging, log_level))
        
        return logger
    
    def upload_to_gcs(self, local_path: str, gcs_path: str, 
                      bucket_name: Optional[str] = None) -> str:
        """
        Upload a file to Google Cloud Storage.
        
        Args:
            local_path: Local file path
            gcs_path: Destination path in GCS (without gs:// prefix)
            bucket_name: GCS bucket name (uses default if not provided)
            
        Returns:
            Full GCS URI of uploaded file
        """
        if bucket_name is None:
            bucket_name = self.gcs_bucket.replace("gs://", "")
        
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(gcs_path)
            
            blob.upload_from_filename(local_path)
            self.logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")
            
            return f"gs://{bucket_name}/{gcs_path}"
        except Exception as e:
            self.logger.error(f"Failed to upload to GCS: {e}")
            raise
    
    def download_from_gcs(self, gcs_uri: str, local_path: str) -> str:
        """
        Download a file from Google Cloud Storage.
        
        Args:
            gcs_uri: Full GCS URI (gs://bucket/path)
            local_path: Local destination path
            
        Returns:
            Local file path
        """
        try:
            # Parse GCS URI
            gcs_uri = gcs_uri.replace("gs://", "")
            bucket_name, blob_path = gcs_uri.split("/", 1)
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Create directories if they don't exist
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            blob.download_to_filename(local_path)
            self.logger.info(f"Downloaded {gcs_uri} to {local_path}")
            
            return local_path
        except Exception as e:
            self.logger.error(f"Failed to download from GCS: {e}")
            raise
    
    def validate_image_format(self, file_path: str) -> bool:
        """Validate if file is a supported image format."""
        supported_formats = self.config["storage"]["output_formats"]["image"]
        file_extension = Path(file_path).suffix.lower().lstrip('.')
        return file_extension in supported_formats
    
    def validate_video_format(self, file_path: str) -> bool:
        """Validate if file is a supported video format."""
        supported_formats = self.config["storage"]["output_formats"]["video"]
        file_extension = Path(file_path).suffix.lower().lstrip('.')
        return file_extension in supported_formats
    
    def get_model_config(self, service_type: str) -> Dict[str, Any]:
        """Get model configuration for a specific service type."""
        return self.config["models"].get(service_type, {})
    
    def get_default_params(self, service_type: str) -> Dict[str, Any]:
        """Get default parameters for a specific service type."""
        return self.config["defaults"].get(service_type, {})
    
    def validate_safety_settings(self, safety_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize safety settings."""
        valid_safety_levels = ["block_most", "block_some", "block_few"]
        valid_person_settings = ["allow_adult", "disallow"]
        
        if "safety_filter_level" in safety_settings:
            if safety_settings["safety_filter_level"] not in valid_safety_levels:
                raise ValueError(f"Invalid safety_filter_level. Must be one of: {valid_safety_levels}")
        
        if "person_generation" in safety_settings:
            if safety_settings["person_generation"] not in valid_person_settings:
                raise ValueError(f"Invalid person_generation. Must be one of: {valid_person_settings}")
        
        return safety_settings