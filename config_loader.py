import yaml
import os
from typing import Any, Optional, Dict
import logging

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

logger = logging.getLogger(__name__)

_config_cache: Optional[Dict[str, Any]] = None

def load_config() -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            config_data = yaml.safe_load(f)
            if not isinstance(config_data, dict):
                logger.error(f"Configuration file {CONFIG_FILE_PATH} is not a valid YAML dictionary.")
                _config_cache = {}
                return {}
            _config_cache = config_data
            logger.info(f"Configuration loaded successfully from {CONFIG_FILE_PATH}")
            return _config_cache
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {CONFIG_FILE_PATH}")
        _config_cache = {}
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {CONFIG_FILE_PATH}: {e}")
        _config_cache = {}
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading configuration from {CONFIG_FILE_PATH}: {e}")
        _config_cache = {}
        return {}

def get_project_id() -> Optional[str]:
    """Returns the default project ID from the configuration."""
    config = load_config()
    return config.get('project', {}).get('default_project_id')

def get_location() -> Optional[str]:
    """Returns the default location from the configuration."""
    config = load_config()
    return config.get('project', {}).get('default_location')

def get_gcs_bucket() -> Optional[str]:
    """Returns the default GCS bucket from the configuration."""
    config = load_config()
    return config.get('project', {}).get('default_gcs_bucket')

def get_model_config(model_type: str) -> Dict[str, Any]:
    """Returns model configuration for a specific type (e.g., 'image_generation')."""
    config = load_config()
    return config.get('models', {}).get(model_type, {})

def get_default_params(service_type: str) -> Dict[str, Any]:
    """Returns default parameters for a specific service (e.g., 'image_generation')."""
    config = load_config()
    return config.get('defaults', {}).get(service_type, {})

if __name__ == '__main__':
    # Basic test for the loader
    logging.basicConfig(level=logging.INFO)
    
    project_id = get_project_id()
    location = get_location()
    gcs_bucket = get_gcs_bucket()
    
    logger.info(f"Default Project ID from config: {project_id}")
    logger.info(f"Default Location from config: {location}")
    logger.info(f"Default GCS Bucket from config: {gcs_bucket}")

    img_gen_models = get_model_config('image_generation')
    logger.info(f"Image Generation Models: {img_gen_models}")
    
    img_gen_defaults = get_default_params('image_generation')
    logger.info(f"Image Generation Defaults: {img_gen_defaults}")

    # Test with a non-existent config file temporarily to check error handling
    # original_path = CONFIG_FILE_PATH
    # CONFIG_FILE_PATH = "non_existent_config.yaml"
    # _config_cache = None # Clear cache
    # load_config()
    # CONFIG_FILE_PATH = original_path # Reset path
