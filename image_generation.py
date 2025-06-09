"""
Image Generation client using Google Cloud Vertex AI Imagen models.
Supports text-to-image generation with various configuration options.
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
            **kwargs: Additional model parameters
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        # Get default parameters and merge with provided ones
        defaults = self.get_default_params("image_generation")
        
        # Build generation parameters with only supported ones
        generation_params = {
            "prompt": prompt,
            "number_of_images": number_of_images or defaults.get("number_of_images", 4),
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
        
        # Handle seed and watermark (mutually exclusive)
        if seed is not None:
            generation_params["seed"] = seed
            # When seed is provided, watermark is automatically disabled
            generation_params["add_watermark"] = False
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
                    "timestamp": timestamp
                }
                
                # Save image locally if output directory is specified
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
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
                
                # Store the PIL Image object for further processing
                result["image"] = generated_image._pil_image
                
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
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate images for multiple prompts.
        
        Args:
            prompts: List of text prompts
            output_dir: Directory to save all generated images
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
            
            try:
                results = self.generate_images(prompt, **kwargs)
                all_results.append(results)
            except Exception as e:
                self.logger.error(f"Failed to generate images for prompt {i+1}: {e}")
                all_results.append([])
        
        return all_results
    
    def upscale_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        upload_to_gcs: bool = False
    ) -> Dict[str, Any]:
        """
        Upscale an existing image using Imagen.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the upscaled image
            upload_to_gcs: Whether to upload result to GCS
            
        Returns:
            Dictionary containing upscaled image data and metadata
        """
        try:
            # Load the image
            if not self.validate_image_format(image_path):
                raise ValueError(f"Unsupported image format: {image_path}")
            
            with Image.open(image_path) as img:
                # Use the model's upscale functionality if available
                # Note: This may vary depending on the specific Imagen model version
                self.logger.info(f"Upscaling image: {image_path}")
                
                # For now, we'll use PIL for basic upscaling
                # In practice, you would use the Imagen API's upscale endpoint
                upscaled_size = (img.width * 2, img.height * 2)
                upscaled_img = img.resize(upscaled_size, Image.Resampling.LANCZOS)
                
                # Save the upscaled image
                if output_path is None:
                    base_path = Path(image_path)
                    output_path = str(base_path.parent / f"{base_path.stem}_upscaled{base_path.suffix}")
                
                upscaled_img.save(output_path)
                
                result = {
                    "original_path": image_path,
                    "upscaled_path": output_path,
                    "original_size": img.size,
                    "upscaled_size": upscaled_img.size,
                    "scale_factor": 2,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Upload to GCS if requested
                if upload_to_gcs:
                    gcs_path = f"upscaled_images/{Path(output_path).name}"
                    gcs_uri = self.upload_to_gcs(output_path, gcs_path)
                    result["gcs_uri"] = gcs_uri
                
                self.logger.info(f"Image upscaled successfully: {output_path}")
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to upscale image: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available image generation models."""
        model_config = self.get_model_config("image_generation")
        return [v for k, v in model_config.items() if k != "default_model"]
