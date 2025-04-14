import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_gemini_api():
    """Test connection to the Gemini API."""
    try:
        # Configure the API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("API key not found. Please set GOOGLE_API_KEY in your .env file.")
            return False

        genai.configure(api_key=api_key)

        # List available models
        models = genai.list_models()
        text_models = [m for m in models if 'generateContent' in m.supported_generation_methods]
        embedding_models = [m for m in models if 'embedContent' in m.supported_generation_methods]

        logger.info(f"Found {len(text_models)} text generation models:")
        for model in text_models:
            logger.info(f"- {model.name}")

        logger.info(f"Found {len(embedding_models)} embedding models:")
        for model in embedding_models:
            logger.info(f"- {model.name}")

        # Test a simple generation
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Write a one-sentence description of what RAG means in AI.")

        logger.info("API Test Response:")
        logger.info(response.text)

        logger.info("API connection successful!")
        return True

    except Exception as e:
        logger.error(f"API connection failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_gemini_api()