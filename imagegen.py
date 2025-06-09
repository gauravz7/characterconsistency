"""
Image Generation client using Google Cloud Vertex AI Imagen models.
Supports text-to-image generation with various configuration options.
Enhanced with better seed handling and batch processing.
"""

import os
import base64
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime

from vertexai.preview.vision_models import ImageGenerationModel, GeneratedImage
from PIL import Image

from base_client import BaseVisionClient


class ImageGenerationClient(BaseVisionClient):
    """Client for generating images using Vertex AI Imagen models."""
    
    def __init__(self, config_path: Optional[str] = None, project_id: Optional[str] = None, 
                 location: Optional[str] = None, model_version: Optional[str] = None):
        """
        Initialize the Image Generation client.
        
        Args:
            config_path: Path to configuration YAML file
            project_id: Google Cloud project ID
            location: Google Cloud location
            model_version: Specific model version to use
        """
        super().__init__(config_path, project_id, location)
        
        # Get model configuration
        model_config = self.get_model_config("image_generation")
        self.model_version = model_version or model_config.get("default_model", "imagen-4.0-generate-preview-05-20")
        
        # Initialize the model
        try:
            self.model = ImageGenerationModel.from_pretrained(self.model_version)
            self.logger.info(f"Initialized Image Generation model: {self.model_version}")
        except Exception as e:
            self.logger.error(f"Failed to initialize model {self.model_version}: {e}")
            raise
    
    def generate_images(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        number_of_images: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        safety_filter_level: Optional[str] = None,
        person_generation: Optional[str] = None,
        seed: Optional[int] = None,
        add_watermark: Optional[bool] = None,
        language: str = "en",
        output_dir: Optional[str] = None,
        upload_to_gcs: bool = False,
        save_locally: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate images from text prompt.
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: Text describing what to avoid in the image
            number_of_images: Number of images to generate (1-4 for most models)
            aspect_ratio: Image aspect ratio (1:1, 9:16, 16:9, 4:3, 3:4)
            safety_filter_level: Safety filtering level (block_most, block_some, block_few)
            person_generation: Person generation setting (allow_adult, disallow)
            seed: Random seed for reproducibility
            add_watermark: Whether to add SynthID watermark
            language: Language code for prompt
            output_dir: Directory to save generated images
            upload_to_gcs: Whether to upload results to GCS
            save_locally: Whether to save images locally
            **kwargs: Additional model parameters
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        # Get default parameters and merge with provided ones
        defaults = self.get_default_params("image_generation")
        
        # Build generation parameters with only supported ones
        generation_params = {
            "prompt": prompt,
            "number_of_images": number_of_images or defaults.get("number_of_images", 1),
            "language": language,
        }
        
        # Add optional parameters only if provided
        if negative_prompt:
            generation_params["negative_prompt"] = negative_prompt
            
        if aspect_ratio:
            generation_params["aspect_ratio"] = aspect_ratio
        elif defaults.get("aspect_ratio"):
            generation_params["aspect_ratio"] = defaults.get("aspect_ratio", "1:1")
            
        if safety_filter_level:
            generation_params["safety_filter_level"] = safety_filter_level
        elif defaults.get("safety_filter_level"):
            generation_params["safety_filter_level"] = defaults.get("safety_filter_level", "block_some")
            
        if person_generation:
            generation_params["person_generation"] = person_generation
        elif defaults.get("person_generation"):
            generation_params["person_generation"] = defaults.get("person_generation", "allow_adult")
        
        # Handle seed and watermark (mutually exclusive in Imagen)
        if seed is not None:
            generation_params["seed"] = seed
            # When seed is provided, watermark is automatically disabled
            generation_params["add_watermark"] = False
            self.logger.info(f"Using seed: {seed} (watermark disabled)")
        else:
            # Only add watermark if no seed is provided
            if add_watermark is not None:
                generation_params["add_watermark"] = add_watermark
            elif defaults.get("add_watermark"):
                generation_params["add_watermark"] = defaults.get("add_watermark", True)
        
        # Remove None values and unsupported parameters
        generation_params = {k: v for k, v in generation_params.items() if v is not None}
        
        # Remove any unsupported parameters that might cause issues
        unsupported_params = ['include_rai_reason', 'include_safety_attributes']
        for param in unsupported_params:
            generation_params.pop(param, None)
        
        # Validate safety settings
        safety_settings = {
            k: v for k, v in generation_params.items() 
            if k in ["safety_filter_level", "person_generation"]
        }
        if safety_settings:
            self.validate_safety_settings(safety_settings)
        
        try:
            self.logger.info(f"Generating {generation_params['number_of_images']} images with prompt: '{prompt[:100]}...'")
            self.logger.debug(f"Generation parameters: {generation_params}")
            
            # Generate images using the model
            response = self.model.generate_images(**generation_params)
            
            # Process the generated images
            results = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for i, generated_image in enumerate(response.images):
                result = {
                    "index": i,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "model_version": self.model_version,
                    "generation_params": generation_params,
                    "timestamp": timestamp,
                    "seed_used": seed if seed is not None else None
                }
                
                # Store the PIL Image object for further processing
                result["image"] = generated_image._pil_image
                
                # Save image locally if output directory is specified and save_locally is True
                if output_dir and save_locally:
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create filename with seed info if used
                    if seed is not None:
                        local_filename = f"generated_image_{timestamp}_seed{seed}_{i:02d}.png"
                    else:
                        local_filename = f"generated_image_{timestamp}_{i:02d}.png"
                    
                    local_path = os.path.join(output_dir, local_filename)
                    
                    generated_image.save(local_path)
                    result["local_path"] = local_path
                    self.logger.info(f"Saved image to: {local_path}")
                    
                    # Upload to GCS if requested
                    if upload_to_gcs:
                        gcs_path = f"generated_images/{timestamp}/{local_filename}"
                        gcs_uri = self.upload_to_gcs(local_path, gcs_path)
                        result["gcs_uri"] = gcs_uri
                
                results.append(result)
            
            self.logger.info(f"Successfully generated {len(results)} images")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to generate images: {e}")
            raise
    
    def generate_single_image(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a single image (convenience method).
        
        Args:
            prompt: Text description of the desired image
            output_path: Path to save the generated image
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing image data and metadata
        """
        kwargs["number_of_images"] = 1
        
        # Set output directory if output_path is provided
        if output_path:
            output_dir = str(Path(output_path).parent)
            kwargs["output_dir"] = output_dir
        
        results = self.generate_images(prompt, **kwargs)
        
        # Rename the file if specific output path was requested
        if output_path and results:
            old_path = results[0].get("local_path")
            if old_path and os.path.exists(old_path):
                os.rename(old_path, output_path)
                results[0]["local_path"] = output_path
                self.logger.info(f"Renamed output file to: {output_path}")
        
        return results[0] if results else {}
    
    def batch_generate_images(
        self,
        prompts: List[str],
        output_dir: str,
        base_seed: Optional[int] = None,
        increment_seed: bool = True,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate images for multiple prompts with enhanced seed handling.
        
        Args:
            prompts: List of text prompts
            output_dir: Directory to save all generated images
            base_seed: Base seed value (if provided, will increment for each prompt)
            increment_seed: Whether to increment seed for each prompt
            **kwargs: Additional generation parameters
            
        Returns:
            List of results for each prompt
        """
        all_results = []
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            # Create subdirectory for this prompt
            prompt_dir = os.path.join(output_dir, f"prompt_{i+1:03d}")
            kwargs["output_dir"] = prompt_dir
            
            # Handle seed progression
            if base_seed is not None:
                if increment_seed:
                    kwargs["seed"] = base_seed + i
                else:
                    kwargs["seed"] = base_seed
                kwargs["add_watermark"] = False
            
            try:
                results = self.generate_images(prompt, **kwargs)
                all_results.append(results)
            except Exception as e:
                self.logger.error(f"Failed to generate images for prompt {i+1}: {e}")
                all_results.append([])
        
        return all_results
    
    def generate_with_seed_variations(
        self,
        prompt: str,
        base_seed: int,
        num_variations: int = 4,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple variations of the same prompt using different seeds.
        
        Args:
            prompt: Text description of the desired image
            base_seed: Base seed value
            num_variations: Number of seed variations to generate
            output_dir: Directory to save generated images
            **kwargs: Additional generation parameters
            
        Returns:
            List of results for all seed variations
        """
        all_results = []
        
        for i in range(num_variations):
            current_seed = base_seed + i
            self.logger.info(f"Generating variation {i+1}/{num_variations} with seed {current_seed}")
            
            try:
                kwargs_copy = kwargs.copy()
                kwargs_copy["seed"] = current_seed
                kwargs_copy["number_of_images"] = 1
                kwargs_copy["add_watermark"] = False
                
                if output_dir:
                    kwargs_copy["output_dir"] = output_dir
                
                results = self.generate_images(prompt, **kwargs_copy)
                all_results.extend(results)
                
            except Exception as e:
                self.logger.error(f"Failed to generate variation {i+1} with seed {current_seed}: {e}")
                continue
        
        return all_results
    
    def generate_style_variations(
        self,
        base_prompt: str,
        style_suffixes: List[str],
        output_dir: Optional[str] = None,
        use_consistent_seed: bool = True,
        base_seed: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate the same subject in different styles.
        
        Args:
            base_prompt: Base prompt describing the subject
            style_suffixes: List of style descriptions to append
            output_dir: Directory to save generated images
            use_consistent_seed: Whether to use the same seed for all styles
            base_seed: Base seed to use (if use_consistent_seed is True)
            **kwargs: Additional generation parameters
            
        Returns:
            List of results for all style variations
        """
        all_results = []
        
        # Determine seed strategy
        if use_consistent_seed and base_seed is None:
            base_seed = 42  # Default seed
        
        for i, style_suffix in enumerate(style_suffixes):
            full_prompt = f"{base_prompt}, {style_suffix}"
            self.logger.info(f"Generating style variation {i+1}/{len(style_suffixes)}: {style_suffix}")
            
            try:
                kwargs_copy = kwargs.copy()
                kwargs_copy["number_of_images"] = 1
                
                if use_consistent_seed:
                    kwargs_copy["seed"] = base_seed
                    kwargs_copy["add_watermark"] = False
                
                if output_dir:
                    style_dir = os.path.join(output_dir, f"style_{i+1:02d}_{style_suffix.replace(' ', '_')[:20]}")
                    kwargs_copy["output_dir"] = style_dir
                
                results = self.generate_images(full_prompt, **kwargs_copy)
                
                # Add style info to results
                for result in results:
                    result["style_suffix"] = style_suffix
                    result["style_index"] = i
                
                all_results.extend(results)
                
            except Exception as e:
                self.logger.error(f"Failed to generate style variation {i+1}: {e}")
                continue
        
        return all_results