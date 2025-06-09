# Creative Suite: Character & Image Generation

This application provides a suite of tools for creative content generation, including character description creation and AI-powered image generation using Google Cloud's Vertex AI models.

## Features

*   **Character Description Generation:** Create rich, detailed descriptions for fictional characters based on user-provided attributes.
*   **Image Prompt Generation:** Generate effective prompts for AI image models by combining character descriptions with scene specifications.
*   **AI Image Generation:** Generate images directly using text prompts, with options for single image or batch generation, leveraging Vertex AI Imagen models.

## Setup and Installation

### Prerequisites

*   Python 3.8 or higher
*   Google Cloud SDK installed and authenticated (`gcloud auth application-default login`)
*   A Google Cloud Project with Vertex AI API enabled.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Configure the Application

The application uses a `config.yaml` file for base configuration, but these can be overridden via the Streamlit application's sidebar.

*   **Primary Configuration (via Streamlit Sidebar):**
    *   When you run `app.py`, the sidebar will have input fields for:
        *   Google Cloud Project ID
        *   Google Cloud Location
        *   GCS Bucket Path (e.g., `gs://your-bucket`)
    *   Values entered here will take precedence and will be used for the current session.

*   **Fallback Configuration (`config.yaml`):**
    *   Ensure `config.yaml` is present in the root directory. This file provides default values if the sidebar fields are left empty.
    *   You can set your default Google Cloud Project ID, location, and GCS bucket in `config.yaml`:
        ```yaml
        project:
          default_project_id: "your-gcp-project-id"  # Replace with your actual Project ID
          default_location: "us-central1"          # Replace if needed
          default_gcs_bucket: "gs://your-default-bucket" # Replace if needed
          # ... other configurations
        ```

### 5. Google Cloud Authentication

Ensure your environment is authenticated to Google Cloud. The application uses Application Default Credentials (ADC). Typically, running the following command is sufficient:

```bash
gcloud auth application-default login
```
Alternatively, you can set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your service account key JSON file.

## Running the Application

Once the setup is complete, you can run the Streamlit application:

```bash
streamlit run app.py
```

This will typically open the application in your default web browser.

## Project Structure

*   `app.py`: Main Streamlit application file.
*   `imagegen.py`: Contains the `ImageGenerationClient` for interacting with Vertex AI image models (this seems to be the active one, previously named `image_generation.py`).
*   `character_creator.py`: Handles character description generation using Gemini.
*   `base_client.py`: Base client for Vertex AI services, handles configuration loading.
*   `config_loader.py`: Utility to load configurations from `config.yaml` (used by `app.py`).
*   `config.yaml`: Configuration file for project ID, location, model defaults, etc.
*   `requirements.txt`: Python package dependencies.
*   `characters/`: Directory where generated character data (JSON files) is stored.
*   `app_generated_images/`: Directory where images generated via the app UI are saved (this directory is in `.gitignore`).
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
