# AI Entrepreneurship Advisor

Deploy-ready AI-powered entrepreneurship advisor. Supports entrepreneurs at every stage with personalized guidance, Khalifa Fund programs, progress tracking, and multilingual support (English/Arabic).

## Quick Deploy (Streamlit Cloud — 3 clicks, free)

1. **Push to GitHub:**
   ```bash
   cd deploy_project
   git init
   git add .
   git commit -m "Initial commit"
   # Create repo on github.com, then:
   git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
   git push -u origin main
   ```

2. **Go to** https://share.streamlit.io
3. **Click "New app"** → connect your GitHub repo → pick `app/app.py`
4. **Add secrets** (Settings → Secrets):
   ```
   OPENAI_API_KEY=sk-or-v1-your-full-key-here
   OPENAI_BASE_URL=https://openrouter.ai/api/v1
   OPENAI_MODEL=deepseek/deepseek-v4-flash
   ```

## Alternative: Hugging Face Spaces

1. Go to https://huggingface.co/new-space
2. Choose **Streamlit** SDK
3. Push the same repo
4. Add the same secrets in Settings

## Files

```
deploy_project/
├── app/
│   ├── app.py           # Main app (single file)
│   ├── requirements.txt # Dependencies
│   ├── README.md        # Full docs
│   └── .env.example     # Environment template
├── .streamlit/
│   └── config.toml      # Streamlit theme config
└── .gitignore
```