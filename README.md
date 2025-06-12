# Ready Tensor Publications Conversational Assistant

This project provides a Streamlit chatbot that helps users explore Ready Tensor publications by asking natural language questions. The assistant uses a sample of 20-50 publications from `project_1_publications.json` and answers questions using Google Gemini.

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your Gemini API key:**
   - Get your API key from https://aistudio.google.com/app/apikey
   - Set it as an environment variable:
     ```bash
     export GOOGLE_API_KEY=your-key-here
     ```
   - Or add it to Streamlit secrets (`.streamlit/secrets.toml`):
     ```toml
     GOOGLE_API_KEY = "your-key-here"
     ```

3. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage
- Ask questions like:
  - What's this publication about?
  - What models or tools were used?
  - Any limitations or assumptions?

The assistant will answer using only the information from the sample publications.