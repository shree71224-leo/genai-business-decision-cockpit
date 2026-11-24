import pandas as pd
from modules.gemini_config import client, MODEL_NAME


def ask_business_question(df, question, mapping=None):
    """Answer business questions using Gemini AI."""

    if client is None:
        return "Gemini API is not configured. Please check your .env file."

    try:
        summary = df.head(30).to_string(index=False)

        prompt = f"""
        You are an expert Business Data Analyst.

        Analyze the following business dataset.

        Dataset columns:
        {list(df.columns)}

        Sample data:
        {summary}

        Column mapping:
        {mapping}

        User question:
        {question}

        Instructions:
        1. Answer based on the provided data.
        2. Explain insights in simple language.
        3. Provide practical business recommendations.
        4. Do not invent statistics.
        """

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"AI Analysis Error: {str(e)}"