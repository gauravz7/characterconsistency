"""
Character Creator Module using Gemini 2.5 Pro for consistent character generation.
Image generation features are currently ignored/disabled.
Prompt generation is focused on combining character description with scene context.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from google import genai # Assuming this is the user's intended import for their environment
from google.genai.types import GenerateContentConfig, ThinkingConfig # ThinkingConfig currently unused

# from image_generation import ImageGenerationClient # Image generation IGNORED

class CharacterCreator:
    """Character creation and management using Gemini 2.5 Pro."""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        try:
            self.gemini_client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
            self.model_name = "gemini-2.0-flash-001" #"gemini-2.5-pro-preview-05-06" # User's specified model
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini client (genai.Client with vertexai=True): {e}")
        
        self.image_client = None # Image generation ignored
        self.base_characters_dir = Path("./characters") # Base directory
        self.base_characters_dir.mkdir(exist_ok=True)
        self.characters: Dict[str, Dict[str, Any]] = {} # In-memory cache

    def _save_character(self, character_data: Dict[str, Any]):
        char_id = character_data.get("id")
        if not char_id:
            print("Error: Character data is missing an ID. Cannot save.")
            return
        
        character_specific_dir = self.base_characters_dir / char_id
        character_specific_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = character_specific_dir / "character_data.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(character_data, f, indent=4)
            print(f"Character {char_id} saved to {file_path}")
        except Exception as e:
            print(f"Error saving character {char_id} to {file_path}: {e}")

    def _load_character(self, character_id: str) -> Optional[Dict[str, Any]]:
        if character_id in self.characters: # Check cache first
            return self.characters[character_id]
        
        character_specific_dir = self.base_characters_dir / character_id
        file_path = character_specific_dir / "character_data.json"
        
        if not file_path.exists():
            print(f"Character data file not found for ID {character_id} at {file_path}")
            return None
        try:
            with open(file_path, 'r') as f:
                character_data = json.load(f)
            self.characters[character_id] = character_data # Cache it
            return character_data
        except Exception as e:
            print(f"Error loading character {character_id} from {file_path}: {e}")
            return None

    def _get_text_from_response(self, response: Any) -> Optional[str]:
        """Extracts text from Gemini response, trying common structures."""
        text_content = None
        if hasattr(response, 'text') and response.text is not None:
            text_content = response.text
        elif hasattr(response, 'candidates') and response.candidates and \
             len(response.candidates) > 0 and \
             hasattr(response.candidates[0], 'content') and response.candidates[0].content and \
             hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts and \
             len(response.candidates[0].content.parts) > 0 and \
             hasattr(response.candidates[0].content.parts[0], 'text'):
            text_content = response.candidates[0].content.parts[0].text
        
        return text_content.strip() if text_content else None

    def _get_detailed_error_feedback(self, response: Any) -> str:
        """Constructs a detailed error feedback string from the response object."""
        feedback_parts = []
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            feedback_parts.append(f"Prompt Feedback: {response.prompt_feedback}")
        
        if hasattr(response, 'candidates') and response.candidates:
            for i, candidate in enumerate(response.candidates):
                cand_details = [f"Candidate {i+1}:"]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    cand_details.append(f"FinishReason: {candidate.finish_reason}")
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    cand_details.append(f"SafetyRatings: {candidate.safety_ratings}")
                
                has_text = False
                if (hasattr(candidate, 'content') and candidate.content and 
                    hasattr(candidate.content, 'parts') and candidate.content.parts and 
                    len(candidate.content.parts) > 0 and 
                    hasattr(candidate.content.parts[0], 'text') and candidate.content.parts[0].text):
                    has_text = True
                
                if not has_text:
                    cand_details.append("No text content in candidate parts.")
                feedback_parts.append(" ".join(cand_details))
        
        return " | ".join(feedback_parts) if feedback_parts else "No detailed feedback available in response object."

    def _generate_single_context_prompt(
        self, background_input: str, clothing_input: str, pose_input: str, view_input: str
    ) -> str:
        system_instruction_text = f"""Objective: Create a short scene description.
Input: Background, Clothing, Pose, View.
Output: 1-3 sentences ONLY. Under 80 words. Describe ONLY scene elements. NO character appearance. NO extra text.
"""
        user_prompt_text = f"""
        Scene Elements:
        - Background: {background_input}
        - Clothing: {clothing_input}
        - Pose: {pose_input}
        - View/Shot: {view_input}

        Task: Combine the above elements into a single, concise paragraph (1-3 sentences, under 80 words) describing ONLY the scene.
        Example: "A dimly lit, ancient library with towering bookshelves. The person wears a dark, hooded cloak, seated at a large wooden table, poring over an open tome. The view is a close-up focusing on the hands and the book."
        """
        try:
            generation_config_obj = GenerateContentConfig(
                system_instruction=system_instruction_text,
                temperature=0.4,
                max_output_tokens=250, # Corrected and reasonable token limit for short output
                tools=None, 
                tool_config=None 
            )
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=user_prompt_text,
                config=generation_config_obj
            )
            
            context_prompt = self._get_text_from_response(response)

            if context_prompt is None:
                error_feedback = self._get_detailed_error_feedback(response)
                print(f"Gemini API returned None for context. Inputs: BG='{background_input}'. Details: {error_feedback}")
                raise ValueError(f"Gemini API returned None for context. Details: {error_feedback}")

            if not context_prompt: # Check for empty string after strip
                 error_feedback = self._get_detailed_error_feedback(response)
                 print(f"Gemini API returned an empty string for context. Inputs: BG='{background_input}'. Details: {error_feedback}")
                 raise ValueError(f"Gemini API returned an empty string for context after stripping. Details: {error_feedback}")

            if context_prompt.startswith("```") and context_prompt.endswith("```"):
                context_prompt = context_prompt[3:-3].strip()
            if context_prompt.startswith('"') and context_prompt.endswith('"') and len(context_prompt) > 1:
                context_prompt = context_prompt[1:-1].strip()
            return context_prompt
        except Exception as e:
            print(f"Error in _generate_single_context_prompt (Inputs: BG='{background_input}'): {e}")
            raise Exception(f"Failed to generate single context prompt: {e}") from e

    def generate_prompts_from_specifications(
        self, character_data: Dict[str, Any], scene_specifications: List[Dict[str, str]]
    ) -> List[str]:
        if not scene_specifications: return []
        character_description = character_data.get('full_description', "")
        
        if not character_description or not character_description.strip():
            print(f"Error: Character {character_data.get('id', 'Unknown')} has an empty description.")

        final_prompts_for_return = []
        prompt_set_variations = []

        for i, spec in enumerate(scene_specifications):
            context_prompt_str = ""
            final_prompt_str = ""
            error_detail = None

            if not character_description or not character_description.strip():
                error_detail = "Character description is missing or empty."
                context_prompt_str = f"[ERROR: {error_detail}]"
                final_prompt_str = f"[ERROR: {error_detail} - Cannot generate full prompt for scene {i+1}]"
            else:
                try:
                    context_prompt_str = self._generate_single_context_prompt(
                        background_input=spec.get('background', 'N/A'),
                        clothing_input=spec.get('clothing', 'N/A'),
                        pose_input=spec.get('pose', 'N/A'),
                        view_input=spec.get('view', 'N/A')
                    )
                    # Combine and replace newlines in the final prompt string
                    combined_prompt = f"{character_description.strip()} {context_prompt_str.strip()}"
                    final_prompt_str = combined_prompt.replace('\n', ' ').replace('\r', ' ')
                except Exception as e:
                    print(f"Error generating context for spec {i}: {e}")
                    error_detail = str(e)
                    context_prompt_str = f"[CONTEXT_GENERATION_ERROR: {error_detail}]"
                    # Combine and replace newlines even in error case for consistency
                    combined_prompt = f"{character_description.strip()} {context_prompt_str}"
                    final_prompt_str = combined_prompt.replace('\n', ' ').replace('\r', ' ')
            
            final_prompts_for_return.append(final_prompt_str)
            variation_data = {
                "variation_index": i,
                "user_inputs": {
                    "background": spec.get('background', 'N/A'),
                    "clothing": spec.get('clothing', 'N/A'),
                    "pose": spec.get('pose', 'N/A'),
                    "view": spec.get('view', 'N/A'),
                },
                "context_prompt": context_prompt_str,
                "final_prompt_original": final_prompt_str,
                "final_prompt_current": final_prompt_str,
            }
            if error_detail:
                variation_data["error_details"] = error_detail
            prompt_set_variations.append(variation_data)

        character_data.setdefault('generated_prompt_sets', [])
        current_datetime_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_set_id = f"prompts_{current_datetime_stamp}_{str(uuid.uuid4())[:4]}"

        new_prompt_set = {
            "id": prompt_set_id,
            "type": "individually_specified_scenes",
            "timestamp": datetime.now().isoformat(), # Redundant with stamp in ID, but okay
            "variations": prompt_set_variations,
            # "prompts_file_path": "" # This will be set after saving
        }

        # Save prompts to a file within the character's generation folder
        char_id = character_data.get("id")
        if char_id:
            generations_dir = self.base_characters_dir / char_id / "generations"
            current_generation_session_dir = generations_dir / current_datetime_stamp
            current_generation_session_dir.mkdir(parents=True, exist_ok=True)
            
            prompts_file_path = current_generation_session_dir / "prompts.txt"
            try:
                with open(prompts_file_path, 'w') as pf:
                    for p_idx, p_text in enumerate(final_prompts_for_return):
                        pf.write(f"--- Prompt {p_idx + 1} ---\n")
                        pf.write(p_text + "\n\n")
                new_prompt_set["prompts_file_path"] = str(prompts_file_path.relative_to(self.base_characters_dir))
                print(f"Prompts for character {char_id} saved to {prompts_file_path}")
            except Exception as e:
                print(f"Error saving prompts to file {prompts_file_path}: {e}")
                new_prompt_set["prompts_file_error"] = str(e)
        else:
            print("Warning: Character ID not found, cannot save prompts to character-specific folder.")

        character_data['generated_prompt_sets'].append(new_prompt_set)
        self._save_character(character_data) # Save character data which now includes prompt set metadata
        return final_prompts_for_return

    def create_character_description(
        self, name: str, age: int, gender: str, ethnicity: str,
        facial_characteristics: str, body_type: str, other_details: str,
        user_edits: Optional[str] = None,
    ) -> Dict[str, Any]:
        system_instruction_text = """
        
        You are a professional character designer and creative writer.
Your primary task is to generate a rich, detailed, and engaging multi-paragraph character description based on the attributes provided.
This description should be suitable for use in AI photo generation prompt creation.
Focus on weaving the provided attributes into a cohesive narrative. Avoid simple lists of features; instead, paint a vivid picture with words.
The final description should be comprehensive, typically between 150 and 300 words, unless the input details are very sparse.
Ensure the tone is descriptive about visual characteristics.
Do not include any conversational elements, greetings, or self-references in your output. Only provide the character description itself.
Write as 1 paragraph in less than 200 words

Example Output1: Tiffany, a 30-year-old female of Japanese-European descent, possesses a slender and athletic physique with distinctly refined features. Her face is characterized by a sharp jawline and prominent high cheekbones, lending an elegant, subtly angular contour. Her eyes are a striking emerald green, almond-shaped, and framed by long, dark, naturally curved lashes. She has rich, dark brown hair with a subtle natural wave, typically styled to shoulder-length, and it possesses a healthy, smooth sheen. Her skin tone is a light olive with warm undertones, consistently clear and even. Standing approximately 5'7" (170 cm), her body is lean and toned, indicative of her athletic build, and she maintains a poised posture. A small, faint, well-healed scar is visible just above her left eyebrow. Her nose is straight and well-proportioned, harmonizing with her other facial features. Her lips are moderately full with a clearly defined cupid's bow, and her eyebrows are dark, neatly shaped, and naturally arched. Her hands are slender with well-maintained, unadorned nails
Example Output2: Generate a, photorealistic portrait of Eva, a captivating 20-year-old woman of Desi heritage. Her most striking feature is the dramatic contrast between her luminous, fair skin and her sharp, intelligent green eyes, the vivid color of polished jade, which hold a perceptive and assessing gaze. She has a confident, curvy figure, and her pose should convey an effortless, sinuous grace that radiates self-possession and a palpable, natural magnetism. The image must capture her unique duality: a composed and alluring exterior that hints at a fiery, untamed spirit simmering within. Use soft yet dramatic lighting to accentuate the contours of her face and form, creating a compelling atmosphere that is both deeply intelligent and undeniably magnetic.

"""
        prompt_parts = [
            "Please generate a detailed and vivid character description based on the following attributes:",
            f"- Character Name: {name}", f"- Age: {age}", f"- Gender: {gender}", f"- Ethnicity: {ethnicity}",
            f"- Facial Characteristics: {facial_characteristics}", f"- Body Type/Shape: {body_type}",
            f"- Other Notable Details/Quirks: {other_details}",
        ]
        if user_edits and user_edits.strip():
             prompt_parts.append(f"\nIncorporate the following user edits or refinements if a previous description existed: {user_edits}")
        prompt_parts.append("\nSynthesize these elements into a compelling narrative. Describe their presence, demeanor, and any subtle hints of their personality, history, or unique style that might be inferred from their appearance. Make the character come alive.")
        user_prompt_text = "\n".join(prompt_parts)

        try:
            generation_config_obj = GenerateContentConfig(
                system_instruction=system_instruction_text,
                temperature=0.7, max_output_tokens=2000
            )
            response = self.gemini_client.models.generate_content(
                model=self.model_name, contents=user_prompt_text, config=generation_config_obj
            )
            character_description = self._get_text_from_response(response)

            if character_description is None:
                error_feedback = self._get_detailed_error_feedback(response)
                print(f"Gemini API returned None for character description for {name}. Details: {error_feedback}")
                raise ValueError(f"Gemini API failed to generate character description (returned None). Details: {error_feedback}")

            if not character_description: # Check for empty string after strip
                 error_feedback = self._get_detailed_error_feedback(response)
                 print(f"Gemini API returned an empty string for character description for {name}. Details: {error_feedback}")
                 raise ValueError(f"Gemini API returned an empty string for character description for {name} after stripping. Details: {error_feedback}")
            
            if character_description.startswith("```") and character_description.endswith("```"):
                lines = character_description.splitlines()
                if len(lines) > 1 and lines[0].startswith("```") and lines[-1] == "```":
                    character_description = "\n".join(lines[1:-1]).strip()
                else:
                    character_description = character_description[3:-3].strip()
            if character_description.startswith('"') and character_description.endswith('"') and len(character_description) > 1:
                character_description = character_description[1:-1].strip()
            
            if not character_description: # Check again after potential stripping of markdown
                 raise ValueError(f"Gemini API returned an effectively empty character description for {name} after cleanup.")

            character_id = str(uuid.uuid4())[:8]
            character_data = {
                "id": character_id, "name": name,
                "input_details": {"age": age, "gender": gender, "ethnicity": ethnicity, 
                                  "facial_characteristics": facial_characteristics, 
                                  "body_type": body_type, "other_details": other_details,
                                  "user_edits_provided": user_edits if user_edits else None},
                "full_description": character_description, "created_at": datetime.now().isoformat(),
                "images": [], "scenes": [], "generated_prompt_sets": []
            }
            self.characters[character_id] = character_data
            self._save_character(character_data)
            return character_data
        except Exception as e:
            print(f"Detailed error in create_character_description for '{name}': {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to create character description for '{name}': {e}") from e

    def generate_character_images_batch(self, character_data: Dict[str, Any], scene_prompts: List[str], seed: int = 42) -> List[Dict[str, Any]]:
        print(f"[INFO] Image generation for {len(scene_prompts)} prompts IGNORED.")
        return [{"image_id": "ignored", "local_path": "ignored.png", "prompt_used": p, "status": "ignored"} for p in scene_prompts]

    def generate_character_image(self, character_data: Dict[str, Any], scene_data: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
        print(f"[INFO] Single image generation IGNORED.")
        return {"image_id": "ignored", "local_path": "ignored.png", "prompt_used": scene_data.get("enhanced_prompt", ""), "status": "ignored"}

    def get_character_images(self, character_id: str) -> List[Dict[str, Any]]:
        char_data = self.characters.get(character_id) or self._load_character(character_id)
        return char_data.get("images", []) if char_data else []
    
    def get_all_characters(self) -> Dict[str, Dict[str, Any]]:
        # Ensure the base directory exists
        self.base_characters_dir.mkdir(exist_ok=True)
        
        loaded_characters = {}
        for char_id_dir in self.base_characters_dir.iterdir():
            if char_id_dir.is_dir(): # Each character is in its own directory named by char_id
                char_id = char_id_dir.name
                if char_id in self.characters: # Prefer cached version if available
                    loaded_characters[char_id] = self.characters[char_id]
                    continue

                char_data_file = char_id_dir / "character_data.json"
                if char_data_file.exists():
                    try:
                        with open(char_data_file, 'r') as f:
                            char_data = json.load(f)
                        if char_data.get("id") == char_id:
                            self.characters[char_id] = char_data # Cache it
                            loaded_characters[char_id] = char_data
                        else:
                            print(f"Warning: Mismatch ID in directory name {char_id} and content of {char_data_file}.")
                    except Exception as e:
                        print(f"Error loading character file {char_data_file}: {e}")
        # Update self.characters to reflect only what's currently on disk + newly created in session
        # This simple approach might clear characters from cache if their folder was deleted.
        # A more robust cache would handle this differently.
        self.characters = loaded_characters 
        return self.characters
    
    def update_character_description(self, character_id: str, new_description: str ) -> Dict[str, Any]:
        char_data = self.characters.get(character_id) or self._load_character(character_id)
        if not char_data: 
            raise ValueError(f"Character {character_id} not found")
        if not new_description or not new_description.strip():
            raise ValueError("Updated character description cannot be empty.")
        
        char_data["full_description"] = new_description.strip()
        char_data["updated_at"] = datetime.now().isoformat()
        char_data.setdefault("description_history", []).append({
            "timestamp": char_data["updated_at"],
            "description": new_description.strip(),
            "source": "manual_update"
        })
        self._save_character(char_data)
        return char_data

if __name__ == '__main__':
    print("Character Creator Module Example")
    try:
        creator = CharacterCreator(project_id="vital-octagon-19612", location="us-central1") 

        print("\n1. Creating a new character...")
        new_char_data = creator.create_character_description(
            name="Lyra Silvertongue", age=28, gender="Female", ethnicity="Nordic-inspired",
            facial_characteristics="Sharp, intelligent sapphire-blue eyes, a light dusting of freckles across a straight nose, high cheekbones, and a determined chin. Her silver hair is often braided intricately.",
            body_type="Slender yet athletic, agile build",
            other_details="Always wears a moonstone pendant. Has a faint, silvery scar above her left eyebrow. Moves with a quiet grace, observant and thoughtful."
        )
        print(f"Character '{new_char_data['name']}' created with ID: {new_char_data['id']}")
        print("Full Description:")
        print(new_char_data['full_description'])
        char_id = new_char_data['id']

        print(f"\n2. Generating scene prompts for {new_char_data['name']}...")
        scene_specs = [
            {"background": "a mystical forest at twilight", "clothing": "dark leather armor with silver trim", "pose": "kneeling by a glowing rune stone", "view": "medium shot"},
            {"background": "a bustling fantasy marketplace", "clothing": "simple traveler's clothes, hooded cloak", "pose": "observing a merchant's stall", "view": "over the shoulder shot"}
        ]
        generated_prompts = creator.generate_prompts_from_specifications(new_char_data, scene_specs)
        
        if generated_prompts:
            print("Generated Scene Prompts:")
            for i, p_str in enumerate(generated_prompts): print(f"  Prompt {i+1}: {p_str}")
        else: print("No prompts generated.")

        print(f"\n3. Retrieving all characters...")
        all_chars = creator.get_all_characters()
        print(f"Found {len(all_chars)} character(s) in total.")
        for c_id, c_data in all_chars.items(): print(f"  - ID: {c_id}, Name: {c_data['name']}")

        print(f"\n4. Updating description for character ID {char_id}...")
        updated_char = creator.update_character_description(char_id, new_char_data['full_description'] + " She now also carries an ornate, ancient dagger at her hip.")
        print("Updated description snippet:", updated_char['full_description'][-100:])

    except Exception as e:
        print(f"\nAn error occurred during the example usage: {e}")
        import traceback
        traceback.print_exc()
    print("\nExample finished.")
