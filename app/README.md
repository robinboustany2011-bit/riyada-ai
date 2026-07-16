# AI Entrepreneurship Advisor

A complete AI-powered entrepreneurship advisor web application built with Streamlit and OpenAI. Supports entrepreneurs at every stage of their business journey with personalized guidance, Khalifa Fund program recommendations, progress tracking, and multilingual support (English & Arabic).

## Features

- **Entrepreneur Profile** — Collect and store entrepreneur details (name, language, business stage, industry, location, Khalifa Fund history, challenges, goals)
- **AI Agents** — Three specialized AI agents powered by OpenAI:
  - **Research Agent** — Searches and retrieves relevant information with sources
  - **Validation Agent** — Verifies information, resolves conflicts, indicates confidence
  - **Arabic Agent** — Fluent bilingual communication in English and Arabic
- **Personalized Guidance** — Khalifa Fund program recommendations, funding options, training, coaching, and a personalized learning plan with milestones
- **Progress Tracking** — Track milestones, upcoming opportunities, and reminders
- **Case Escalation** — Intelligent escalation to a Khalifa Fund advisor for complex cases
- **Multilingual** — Full support for English and Arabic
- **Source Citations** — Every response includes sources

## Prerequisites

- Python 3.9 or higher
- An OpenAI API key

## Installation

1. **Clone or download** the project files:
   ```
   project/
   └── app/
       ├── app.py
       ├── requirements.txt
       └── README.md
   ```

2. **Navigate to the project directory:**
   ```bash
   cd project
   ```

3. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate     # On Linux/Mac
   # or
   venv\Scripts\activate        # On Windows
   ```

4. **Install dependencies:**
   ```bash
   pip install -r app/requirements.txt
   ```

5. **Set up your OpenAI API key.**

   Create a `.env` file in the project root:
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```
   
   Or set the environment variable directly:
   ```bash
   export OPENAI_API_KEY=your-api-key-here   # Linux/Mac
   # or
   set OPENAI_API_KEY=your-api-key-here      # Windows
   ```

## Usage

1. **Run the application:**
   ```bash
   streamlit run app/app.py
   ```

2. **Open your browser** to the URL shown in the terminal (typically http://localhost:8501).

3. **Complete your profile** in the sidebar:
   - Name
   - Preferred language (English or Arabic)
   - Business stage (Idea, Validation, Launch, Growth, Scaling)
   - Industry/Sector
   - Business location
   - History with Khalifa Fund
   - Current challenges
   - Goals

4. **Start chatting** with the AI advisor in the main chat window.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r app/requirements.txt` |
| `AuthenticationError` | Check your `OPENAI_API_KEY` is set correctly in the `.env` file |
| App won't start | Ensure you're running `streamlit run app/app.py` from the project root directory |
| Arabic text not displaying | Ensure your browser supports Arabic (all modern browsers do) |
| No sources shown | The Research Agent requires internet access to fetch information |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |

## Project Structure

```
project/
└── app/
    ├── app.py            # Main application (single file)
    ├── requirements.txt  # Python dependencies
    └── README.md         # This file
```

## Tech Stack

- **Python** — Core programming language
- **Streamlit** — Web framework (frontend + backend)
- **OpenAI API** — AI language model for agents
- **python-dotenv** — Environment variable management
- **Requests** — HTTP requests for web search

## License

This project is provided for educational and demonstration purposes.