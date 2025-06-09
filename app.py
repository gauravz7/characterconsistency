import streamlit as st
from character_creator import CharacterCreator # Assuming character_creator.py is in the same directory
import imagegen # Import the imagegen module
from image_generation import ImageGenerationClient # Import the new image generation client
from vertexai.preview.vision_models import ImageGenerationModel # Import for model switching
from pathlib import Path
import io
import zipfile
import base64 # For creating unique filenames if needed, or just use indices
from PIL import Image
import os

# --- Configuration ---
PROJECT_ID = "vital-octagon-19612" 
LOCATION = "us-central1"

# Ensure the imagegen client is initialized with the correct project ID
# This happens within imagegen.py now, but we ensure it's called.
imagegen.PROJECT_ID = PROJECT_ID
imagegen.LOCATION = LOCATION


# --- Helper Functions ---
@st.cache_resource # Cache the CharacterCreator instance
def get_character_creator():
    """Initializes and returns the CharacterCreator instance."""
    try:
        creator_instance = CharacterCreator(project_id=PROJECT_ID, location=LOCATION)
        Path("./characters").mkdir(exist_ok=True)
        return creator_instance
    except Exception as e:
        st.error(f"Failed to initialize CharacterCreator: {e}")
        # st.stop() # Removed to allow other tabs to function if this one fails
        return None

@st.cache_resource
def get_image_generator_client():
    """Initializes and returns the ImageGenerationClient for image generation."""
    try:
        client = ImageGenerationClient(project_id=PROJECT_ID, location=LOCATION)
        if not client:
            st.error("Failed to initialize Image Generator Client. Check console for details.")
            return None
        Path("./app_generated_images").mkdir(parents=True, exist_ok=True) # For temporary storage
        return client
    except Exception as e:
        st.error(f"Error initializing Image Generator Client: {e}")
        return None

def load_characters(creator):
    """Loads all characters and returns a dictionary {name (id): id}."""
    if not creator:
        return {}
    try:
        all_chars_data = creator.get_all_characters()
        return {f"{data.get('name', 'Unnamed')} (ID: {char_id})": char_id 
                for char_id, data in all_chars_data.items()}
    except Exception as e:
        st.error(f"Error loading characters: {e}")
        return {}

def pil_to_bytes(pil_image, format="PNG"):
    """Converts a PIL Image to bytes."""
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

def create_zip_download(images_data, filename_prefix="generated_images"):
    """Creates a ZIP file containing all generated images."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img_data in enumerate(images_data):
            if 'image' in img_data and img_data['image']:
                img_bytes = pil_to_bytes(img_data['image'])
                filename = f"{filename_prefix}_{i+1:02d}.png"
                zip_file.writestr(filename, img_bytes)
    return zip_buffer.getvalue()

# --- Streamlit App UI ---
st.set_page_config(layout="wide") 

# --- Initialize Services ---
creator = get_character_creator()
image_client = get_image_generator_client()

st.title("Creative Suite: Character & Image Generation")

tab_titles = ["Create Character Description", "Generate Image Prompt"]
if image_client: # Only add image gen tab if client is available
    tab_titles.append("Generate Images") # This will be the third tab
else:
    st.warning("Image Generation tab is unavailable because the image client could not be initialized.")

tabs = st.tabs(tab_titles)

# --- Tab 1: Create Character Description ---
with tabs[0]:
    st.header("1. Create New Character Description")
    st.markdown("Provide details for your character. The AI will generate a rich description.")
    if not creator:
        st.error("Character Creator service is unavailable.")
    else:
        with st.form("character_form"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Character Name", placeholder="e.g., Elara Meadowlight")
                age = st.number_input("Age", min_value=0, max_value=2000, value=30)
                gender = st.text_input("Gender", placeholder="e.g., Female, Male, Non-binary")
                ethnicity = st.text_input("Ethnicity / Ancestry", placeholder="e.g., Forest Elf, Human (Highlander)")
            
            with c2:
                facial_characteristics = st.text_area("Facial Characteristics", height=100, placeholder="e.g., Sharp, intelligent eyes; a sprinkle of freckles; strong jawline")
                body_type = st.text_area("Body Type / Shape", height=100, placeholder="e.g., Slender and agile; powerfully built; average height with a slight stoop")
            
            other_details = st.text_area("Other Notable Details / Quirks / Attire Hints", height=120, placeholder="e.g., Always wears a silver locket; walks with a slight limp; prefers dark, practical clothing")
            user_edits = st.text_area("User Edits (Optional - if refining a previous attempt, describe changes here)", height=80, placeholder="e.g., Make the hair color dark brown instead of blonde.")
            submit_button_char = st.form_submit_button("Generate Character Description")

        if submit_button_char:
            if not name:
                st.warning("Please enter a character name.")
            else:
                with st.spinner("Generating character description..."):
                    try:
                        char_data = creator.create_character_description(
                            name=name, age=int(age), gender=gender, ethnicity=ethnicity,
                            facial_characteristics=facial_characteristics, body_type=body_type,
                            other_details=other_details, user_edits=user_edits if user_edits.strip() else None
                        )
                        st.success(f"Character '{char_data['name']}' created successfully!")
                        st.subheader("Generated Description:")
                        st.markdown(f"**ID:** `{char_data['id']}`")
                        st.markdown(char_data['full_description'])
                        st.subheader("Input Details Provided:")
                        st.json(char_data['input_details'])
                        if 'character_options' in st.session_state:
                            st.session_state.character_options = load_characters(creator)
                    except Exception as e:
                        st.error(f"Error generating character description: {e}")
                        st.exception(e)

# --- Tab 2: Generate Image Prompt ---
with tabs[1]:
    st.header("2. Generate Image Prompts for Multiple Scenes")
    st.markdown("Select a character, specify scenes, and generate combined prompts.")
    if not creator:
        st.error("Character Creator service is unavailable.")
    else:
        if 'character_options' not in st.session_state:
            st.session_state.character_options = load_characters(creator)
        
        character_options = st.session_state.character_options

        if not character_options:
            st.info("No characters found. Create a character in Tab 1 first.")
        else:
            selected_char_display_name = st.selectbox(
                "Select Character", options=list(character_options.keys()), index=0, key="char_select_tab2"
            )
            num_scenes = st.number_input("Number of Scenes", min_value=1, max_value=10, value=1, step=1, key="num_scenes_input")
            scene_specifications_input = []

            with st.form("multi_prompt_form"):
                for i in range(num_scenes):
                    st.subheader(f"Scene {i+1} Details")
                    cols = st.columns(2)
                    bg = cols[0].text_input(f"Background##scene{i}", placeholder="e.g., Mystical forest", key=f"bg_scene_{i}")
                    cloth = cols[0].text_input(f"Clothing##scene{i}", placeholder="e.g., Dark leather armor", key=f"cloth_scene_{i}")
                    pose = cols[1].text_input(f"Pose##scene{i}", placeholder="e.g., Kneeling by a rune stone", key=f"pose_scene_{i}")
                    view = cols[1].text_input(f"View / Shot Type##scene{i}", placeholder="e.g., Medium shot", key=f"view_scene_{i}")
                    scene_specifications_input.append({"background": bg, "clothing": cloth, "pose": pose, "view": view})
                
                submit_button_prompts = st.form_submit_button("Generate All Image Prompts")

            if submit_button_prompts and selected_char_display_name:
                selected_char_id = character_options[selected_char_display_name]
                with st.spinner(f"Generating {num_scenes} image prompt(s)..."):
                    try:
                        char_data = creator._load_character(selected_char_id)
                        if not char_data or not char_data.get('full_description'):
                            st.error(f"Character data or description missing for ID: {selected_char_id}")
                        else:
                            valid_scene_specs = [s for s in scene_specifications_input[:num_scenes] if any(s.values())]
                            if not valid_scene_specs:
                                st.warning("Please provide details for at least one scene.")
                            else:
                                prompts_list = creator.generate_prompts_from_specifications(char_data, valid_scene_specs)
                                st.subheader("Generated Image Prompts:")
                                all_prompts_text = ""
                                for idx, p_text in enumerate(prompts_list):
                                    st.text_area(f"Prompt {idx+1}", p_text, height=100, key=f"prompt_display_{idx}")
                                    all_prompts_text += p_text + "\n\n" # Separate prompts by double newline
                                if all_prompts_text:
                                    st.download_button("Download Prompts", all_prompts_text.strip(), 
                                                       f"{char_data.get('name', 'char').replace(' ', '_')}_prompts.txt", "text/plain")
                    except Exception as e:
                        st.error(f"Error generating image prompts: {e}")
                        st.exception(e)

# --- Tab 3: Generate Images ---
if image_client and len(tabs) > 2:
    with tabs[2]:
        st.header("3. Generate Images")
        st.markdown("Generate images from text prompts using AI image generation.")
        
        # Mode selection
        generation_mode = st.radio(
            "Generation Mode",
            ["Single Image", "Batch Generation"],
            horizontal=True
        )
        
        if generation_mode == "Single Image":
            st.subheader("Single Image Generation")
            
            with st.form("single_image_form"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    single_prompt = st.text_area(
                        "Image Prompt", 
                        height=100,
                        placeholder="e.g., A mystical forest elf with silver hair, standing in an enchanted grove, fantasy art style"
                    )
                    negative_prompt = st.text_area(
                        "Negative Prompt (Optional)",
                        height=80,
                        placeholder="e.g., blurry, low quality, distorted"
                    )
                
                with col2:
                    # Model selection
                    model_options = {
                        "Imagen 3.0": "imagen-3.0-generate-002",
                        "Imagen 4.0": "imagen-4.0-generate-preview-05-20"
                    }
                    selected_model_name = st.selectbox(
                        "Model Version",
                        options=list(model_options.keys()),
                        index=0,
                        help="Choose between different Imagen model versions"
                    )
                    selected_model = model_options[selected_model_name]
                    
                    aspect_ratio = st.selectbox(
                        "Aspect Ratio",
                        options=["1:1", "9:16", "16:9", "4:3", "3:4"],
                        index=0
                    )
                    
                    safety_filter = st.selectbox(
                        "Safety Filter Level",
                        options=["block_some", "block_most", "block_few"],
                        index=0
                    )
                    
                    person_generation = st.selectbox(
                        "Person Generation",
                        options=["allow_adult", "disallow"],
                        index=0
                    )
                    
                    # Seed parameter - always visible for editing
                    seed_value = st.number_input(
                        "Seed Value (0 = random)",
                        min_value=0,
                        max_value=2147483647,
                        value=0,
                        help="Use 0 for random generation, or set a specific number for reproducible results"
                    )
                
                submit_single = st.form_submit_button("Generate Single Image")
            
            if submit_single and single_prompt.strip():
                with st.spinner("Generating image..."):
                    try:
                        generation_params = {
                            "prompt": single_prompt,
                            "aspect_ratio": aspect_ratio,
                            "safety_filter_level": safety_filter,
                            "person_generation": person_generation,
                            "output_dir": "./app_generated_images"
                        }
                        
                        if negative_prompt.strip():
                            generation_params["negative_prompt"] = negative_prompt
                        
                        # Handle seed parameter
                        if seed_value > 0:
                            generation_params["seed"] = seed_value
                            generation_params["add_watermark"] = False
                        else:
                            generation_params["add_watermark"] = True
                        
                        # Set model version
                        image_client.model_version = selected_model
                        image_client.model = ImageGenerationModel.from_pretrained(selected_model)
                        
                        result = image_client.generate_single_image(**generation_params)
                        
                        if result and 'image' in result:
                            st.success("Image generated successfully!")
                            
                            # Display the image
                            st.image(result['image'], caption=f"Generated Image", use_column_width=True)
                            
                            # Show metadata
                            with st.expander("Generation Details"):
                                st.json({
                                    "model_version": result.get('model_version'),
                                    "timestamp": result.get('timestamp'),
                                    "prompt": result.get('prompt'),
                                    "negative_prompt": result.get('negative_prompt'),
                                    "generation_params": result.get('generation_params')
                                })
                            
                            # Download button
                            img_bytes = pil_to_bytes(result['image'])
                            st.download_button(
                                "Download Image",
                                data=img_bytes,
                                file_name=f"generated_image_{result.get('timestamp', 'unknown')}.png",
                                mime="image/png"
                            )
                        else:
                            st.error("Failed to generate image. Please try again.")
                            
                    except Exception as e:
                        st.error(f"Error generating image: {e}")
                        st.exception(e)
        
        else:  # Batch Generation
            st.subheader("Batch Image Generation")
            
            with st.form("batch_image_form"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    batch_prompts = st.text_area(
                        "Image Prompts (one per line)",
                        height=150,
                        placeholder="Enter multiple prompts, one per line:\nA mystical forest elf with silver hair\nA dragon soaring over mountains\nA medieval castle at sunset"
                    )
                    batch_negative_prompt = st.text_area(
                        "Negative Prompt (applies to all images)",
                        height=80,
                        placeholder="e.g., blurry, low quality, distorted"
                    )
                
                with col2:
                    # Model selection for batch
                    batch_model_options = {
                        "Imagen 3.0": "imagen-3.0-generate-002",
                        "Imagen 4.0": "imagen-4.0-generate-preview-05-20"
                    }
                    batch_selected_model_name = st.selectbox(
                        "Model Version",
                        options=list(batch_model_options.keys()),
                        index=0,
                        key="batch_model_select",
                        help="Choose between different Imagen model versions"
                    )
                    batch_selected_model = batch_model_options[batch_selected_model_name]
                    
                    batch_aspect_ratio = st.selectbox(
                        "Aspect Ratio",
                        options=["1:1", "9:16", "16:9", "4:3", "3:4"],
                        index=0,
                        key="batch_aspect"
                    )
                    
                    batch_safety_filter = st.selectbox(
                        "Safety Filter Level",
                        options=["block_some", "block_most", "block_few"],
                        index=0,
                        key="batch_safety"
                    )
                    
                    batch_person_generation = st.selectbox(
                        "Person Generation",
                        options=["allow_adult", "disallow"],
                        index=0,
                        key="batch_person"
                    )
                    
                    images_per_prompt = st.number_input(
                        "Images per Prompt",
                        min_value=1,
                        max_value=4,
                        value=1,
                        help="Number of images to generate for each prompt"
                    )
                    
                    # Seed parameter - always visible for editing
                    batch_seed_base = st.number_input(
                        "Base Seed (0 = random)",
                        min_value=0,
                        max_value=2147483647,
                        value=0,
                        key="batch_seed_value",
                        help="Base seed for batch generation. Use 0 for random. Will increment for each prompt."
                    )
                
                submit_batch = st.form_submit_button("Generate Batch Images")
            
            if submit_batch and batch_prompts.strip():
                # Parse prompts
                prompts_list = [p.strip() for p in batch_prompts.split('\n') if p.strip()]
                
                if not prompts_list:
                    st.warning("Please enter at least one prompt.")
                else:
                    st.info(f"Generating images for {len(prompts_list)} prompts...")
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_results = []
                    
                    for i, prompt in enumerate(prompts_list):
                        status_text.text(f"Processing prompt {i+1}/{len(prompts_list)}: {prompt[:50]}...")
                        
                        try:
                            generation_params = {
                                "prompt": prompt,
                                "number_of_images": images_per_prompt,
                                "aspect_ratio": batch_aspect_ratio,
                                "safety_filter_level": batch_safety_filter,
                                "person_generation": batch_person_generation,
                                "output_dir": "./app_generated_images"
                            }
                            
                            if batch_negative_prompt.strip():
                                generation_params["negative_prompt"] = batch_negative_prompt
                            
                            # Handle seed parameter
                            if batch_seed_base > 0:
                                generation_params["seed"] = batch_seed_base + i
                                generation_params["add_watermark"] = False
                            else:
                                generation_params["add_watermark"] = True
                            
                            # Set model version for this batch
                            image_client.model_version = batch_selected_model
                            image_client.model = ImageGenerationModel.from_pretrained(batch_selected_model)
                            
                            results = image_client.generate_images(**generation_params)
                            all_results.extend(results)
                            
                        except Exception as e:
                            st.error(f"Error generating images for prompt {i+1}: {e}")
                            continue
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(prompts_list))
                    
                    status_text.text("Generation complete!")
                    
                    if all_results:
                        st.success(f"Generated {len(all_results)} images successfully!")
                        
                        # Display results in a grid
                        st.subheader("Generated Images")
                        
                        # Create columns for grid display
                        cols_per_row = 3
                        for i in range(0, len(all_results), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j, result in enumerate(all_results[i:i+cols_per_row]):
                                with cols[j]:
                                    if 'image' in result:
                                        st.image(
                                            result['image'],
                                            caption=f"Image {i+j+1}: {result['prompt'][:30]}...",
                                            use_column_width=True
                                        )
                        
                        # Batch download option
                        if len(all_results) > 1:
                            zip_data = create_zip_download(all_results, "batch_generated_images")
                            st.download_button(
                                "Download All Images (ZIP)",
                                data=zip_data,
                                file_name=f"batch_generated_images_{len(all_results)}_images.zip",
                                mime="application/zip"
                            )
                        
                        # Individual download buttons
                        with st.expander("Individual Downloads & Details"):
                            for i, result in enumerate(all_results):
                                if 'image' in result:
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.text(f"Image {i+1}: {result['prompt']}")
                                    with col2:
                                        img_bytes = pil_to_bytes(result['image'])
                                        st.download_button(
                                            "Download",
                                            data=img_bytes,
                                            file_name=f"batch_image_{i+1:02d}_{result.get('timestamp', 'unknown')}.png",
                                            mime="image/png",
                                            key=f"download_batch_{i}"
                                        )


# --- Sidebar ---
st.sidebar.header("App Info")
st.sidebar.info(
    """
    **How to Use:**
    1.  **Create Character Description:** (Tab 1) Fill details, generate.
    2.  **Generate Image Prompt:** (Tab 2) Select character, define scenes, generate.
    3.  **Generate Images:** (Tab 3) Input prompts & params, generate images.
    
    Character data is saved in './characters'.
    Generated images are saved in './app_generated_images'.
    """
)
st.sidebar.markdown(f"Using Project ID: `{PROJECT_ID}`")
if not creator:
    st.sidebar.warning("Character Creator features might be limited.")
if not image_client:
    st.sidebar.warning("Image Generation features are unavailable.")
else:
    st.sidebar.success("Image Generation client initialized successfully.")