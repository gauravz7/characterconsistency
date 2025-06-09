import streamlit as st
from character_creator import CharacterCreator # Assuming character_creator.py is in the same directory
from pathlib import Path

# --- Configuration ---
PROJECT_ID = "vital-octagon-19612" # As used in your character_creator.py example
LOCATION = "us-central1"

# --- Helper Functions ---
@st.cache_resource # Cache the CharacterCreator instance
def get_character_creator():
    """Initializes and returns the CharacterCreator instance."""
    try:
        creator_instance = CharacterCreator(project_id=PROJECT_ID, location=LOCATION)
        # Ensure characters directory exists, though CharacterCreator does this too
        Path("./characters").mkdir(exist_ok=True)
        return creator_instance
    except Exception as e:
        st.error(f"Failed to initialize CharacterCreator: {e}")
        st.stop() # Stop the app if creator can't be initialized
        return None

def load_characters(creator):
    """Loads all characters and returns a dictionary {name (id): id}."""
    if not creator:
        return {}
    try:
        all_chars_data = creator.get_all_characters() # Returns Dict[str, Dict[str, Any]]
        # Create a user-friendly mapping for the dropdown
        # Format: "Character Name (ID: XXXXXXXX)" -> character_id
        return {f"{data.get('name', 'Unnamed')} (ID: {char_id})": char_id 
                for char_id, data in all_chars_data.items()}
    except Exception as e:
        st.error(f"Error loading characters: {e}")
        return {}

# --- Streamlit App UI ---
st.set_page_config(layout="wide") # Must be the first Streamlit command

# --- Initialize CharacterCreator ---
creator = get_character_creator()

st.title("Character Creator & Prompt Generator")

if creator:
    tab1, tab2 = st.tabs(["Create Character Description", "Generate Image Prompt"])

    # --- Tab 1: Create Character Description ---
    with tab1:
        st.header("1. Create New Character Description")
        st.markdown("Provide details for your character. The AI will generate a rich description.")

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
                with st.spinner("Generating character description... This may take a moment."):
                    try:
                        char_data = creator.create_character_description(
                            name=name,
                            age=int(age), # Ensure age is int
                            gender=gender,
                            ethnicity=ethnicity,
                            facial_characteristics=facial_characteristics,
                            body_type=body_type,
                            other_details=other_details,
                            user_edits=user_edits if user_edits.strip() else None
                        )
                        st.success(f"Character '{char_data['name']}' created successfully!")
                        
                        st.subheader("Generated Description:")
                        st.markdown(f"**ID:** `{char_data['id']}`")
                        st.markdown(char_data['full_description'])

                        st.subheader("Input Details Provided:")
                        st.json(char_data['input_details'])
                        
                        # Refresh character list for Tab 2 if it's already initialized
                        if 'character_options' in st.session_state:
                            st.session_state.character_options = load_characters(creator)

                    except Exception as e:
                        st.error(f"Error generating character description: {e}")
                        st.exception(e) # Show full traceback for debugging

    # --- Tab 2: Generate Image Prompt ---
    with tab2:
        st.header("2. Generate Image Prompts for Multiple Scenes")
        st.markdown("Select a character, specify the number of scenes, and provide details for each scene to generate combined prompts.")

        if 'character_options' not in st.session_state:
            st.session_state.character_options = load_characters(creator)
        
        character_options = st.session_state.character_options

        if not character_options:
            st.info("No characters found. Please create a character in Tab 1 first.")
        else:
            selected_char_display_name = st.selectbox(
                "Select Character", 
                options=list(character_options.keys()),
                index=0,
                key="char_select_tab2",
                help="Choose a character whose description will be used."
            )

            num_scenes = st.number_input("Number of Scenes", min_value=1, max_value=10, value=1, step=1, key="num_scenes_input")

            scene_specifications_input = []

            with st.form("multi_prompt_form"):
                for i in range(num_scenes):
                    st.subheader(f"Scene {i+1} Details")
                    cols = st.columns(2)
                    bg = cols[0].text_input(f"Background##scene{i}", placeholder="e.g., A mystical forest at twilight", key=f"bg_scene_{i}")
                    cloth = cols[0].text_input(f"Clothing##scene{i}", placeholder="e.g., Dark leather armor", key=f"cloth_scene_{i}")
                    pose = cols[1].text_input(f"Pose##scene{i}", placeholder="e.g., Kneeling by a rune stone", key=f"pose_scene_{i}")
                    view = cols[1].text_input(f"View / Shot Type##scene{i}", placeholder="e.g., Medium shot", key=f"view_scene_{i}")
                    scene_specifications_input.append({
                        "background": bg, "clothing": cloth, "pose": pose, "view": view
                    })
                
                submit_button_prompts = st.form_submit_button("Generate All Image Prompts")

            if submit_button_prompts and selected_char_display_name:
                selected_char_id = character_options[selected_char_display_name]
                
                with st.spinner(f"Generating {num_scenes} image prompt(s)..."):
                    try:
                        char_data = creator._load_character(selected_char_id)
                        if not char_data:
                            st.error(f"Could not load data for character ID: {selected_char_id}")
                        elif not char_data.get('full_description'):
                            st.error(f"Character '{char_data.get('name', 'Unknown')}' (ID: {selected_char_id}) has no description. Cannot generate prompts.")
                        else:
                            # Filter out empty scene specifications if any (e.g., if user reduces num_scenes after filling some)
                            valid_scene_specs = [
                                spec for spec in scene_specifications_input[:num_scenes] 
                                if any(spec.values()) # Consider a spec valid if at least one field is filled
                            ]

                            if not valid_scene_specs:
                                st.warning("Please provide details for at least one scene.")
                            else:
                                generated_prompts_list = creator.generate_prompts_from_specifications(
                                    character_data=char_data,
                                    scene_specifications=valid_scene_specs
                                )
                                
                                st.subheader(f"Generated Image Prompts (Single Line Each):")
                                all_prompts_text = ""
                                for idx, prompt_text in enumerate(generated_prompts_list):
                                    st.text_area(f"Prompt {idx+1}", prompt_text, height=100, key=f"prompt_display_{idx}")
                                    all_prompts_text += prompt_text + "\n"
                                
                                if all_prompts_text:
                                    st.download_button(
                                        label="Download Prompts as .txt",
                                        data=all_prompts_text.strip(), # Remove trailing newline
                                        file_name=f"{char_data.get('name', 'character').replace(' ', '_')}_prompts.txt",
                                        mime="text/plain"
                                    )
                    except Exception as e:
                        st.error(f"Error generating image prompts: {e}")
                        st.exception(e)
else:
    st.error("CharacterCreator could not be initialized. Please check the console for errors.")

st.sidebar.info(
    """
    **How to Use:**
    1.  **Create Character Description:** Go to the first tab, fill in character details, and click "Generate".
    2.  **Generate Image Prompt:** Go to the second tab, select a created character, provide scene details, and click "Generate".
    
    Character data is saved in the './characters' directory.
    """
)
st.sidebar.markdown(f"Using Project ID: `{PROJECT_ID}`")
