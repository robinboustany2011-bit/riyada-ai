"""
AI Entrepreneurship Advisor — A complete AI-powered entrepreneurship advisor
built with Streamlit and OpenAI. Supports entrepreneurs at every stage of their
business journey with personalized guidance, Khalifa Fund program recommendations,
progress tracking, and multilingual support (English & Arabic).

Architecture:
  - Single-file Streamlit app (app.py)
  - Three AI agents orchestrated via OpenAI Chat Completions:
      1. Research Agent  — web search + synthesis
      2. Validation Agent — fact-checking and confidence scoring
      3. Arabic Agent    — bilingual (EN/AR) communication
  - Entrepreneur profile stored in Streamlit session state
  - Lightweight keyword-based search (no vector DBs, no embeddings)

Author: AI Entrepreneurship Advisor
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import os
import json
import time
import re
from datetime import datetime, date
from typing import Optional

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import requests

# ── CONFIGURATION ────────────────────────────────────────────────────────────

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))  # Load .env from app/ directory
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek/deepseek-v4-flash")

# Khalifa Fund official resources (used as sources by the agents)
KHALIFA_FUND_SOURCES = {
    "website": "https://www.khalifafund.ae",
    "programs": "https://www.khalifafund.ae/programs",
    "services": "https://www.khalifafund.ae/services",
    "contact": "https://www.khalifafund.ae/contact",
    "faq": "https://www.khalifafund.ae/faq",
}

# Business stages
BUSINESS_STAGES = [
    "Idea / الفكرة",
    "Validation / التحقق",
    "Launch / الإطلاق",
    "Growth / النمو",
    "Scaling / التوسع",
]

# Industries
INDUSTRIES = [
    "Technology / التكنولوجيا",
    "Retail / التجزئة",
    "Manufacturing / التصنيع",
    "Food & Beverage / الطعام والشراب",
    "Healthcare / الرعاية الصحية",
    "Education / التعليم",
    "Construction / البناء",
    "Tourism / السياحة",
    "Agriculture / الزراعة",
    "Finance / المالية",
    "Media / الإعلام",
    "Other / أخرى",
]

# Default common challenges
COMMON_CHALLENGES = [
    "Funding / التمويل",
    "Market Research / أبحاث السوق",
    "Business Planning / تخطيط الأعمال",
    "Legal & Licensing / القانوني والتراخيص",
    "Marketing / التسويق",
    "Team Building / بناء الفريق",
    "Operations / العمليات",
    "Technology / التكنولوجيا",
]

# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────


def get_openai_client() -> Optional[OpenAI]:
    """Return an OpenAI client or None if the API key is missing."""
    key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key, base_url=OPENAI_BASE_URL)


def translate_label(label_en: str, lang: str) -> str:
    """Return a bilingual label combining English and Arabic."""
    if lang == "Arabic":
        translations = {
            "Name": "الاسم",
            "Preferred Language": "اللغة المفضلة",
            "Business Stage": "مرحلة العمل",
            "Industry / Sector": "القطاع / الصناعة",
            "Business Location": "موقع العمل",
            "History with Khalifa Fund": "التاريخ مع صندوق خليفة",
            "Current Challenges": "التحديات الحالية",
            "Goals": "الأهداف",
            "Save Profile": "حفظ الملف الشخصي",
            "Profile Saved": "تم حفظ الملف الشخصي",
            "Chat with your AI Advisor": "Riyada AI",
            "Type your message here...": "اكتب رسالتك هنا...",
            "Send": "إرسال",
            "Clear Chat": "مسح المحادثة",
            "Progress Tracking": "تتبع التقدم",
            "Milestones": "المعالم",
            "Completed": "مكتمل",
            "In Progress": "قيد التقدم",
            "Pending": "معلق",
            "Personalized Guidance": "الإرشادات المخصصة",
            "Your Learning Plan": "خطة التعلم الخاصة بك",
            "Escalate to Advisor": "الرفع إلى مستشار",
            "Sources": "المصادر",
        }
        return translations.get(label_en, label_en)
    return label_en


def format_sources(sources: list) -> str:
    """Format a list of source URLs/names into a markdown string."""
    if not sources:
        return ""
    lines = ["**" + translate_label("Sources", st.session_state.get("language", "English")) + ":**"]
    for s in sources:
        lines.append(f"- {s}")
    return "\n".join(lines)


# ── WEB SEARCH (Lightweight Keyword Search) ──────────────────────────────────


def lightweight_web_search(query: str, max_results: int = 5) -> list:
    """
    Perform a lightweight keyword-based web search using DuckDuckGo's
    HTML scraping (no API key required). Falls back gracefully on failure.

    Returns a list of dicts: [{"title": ..., "url": ..., "snippet": ...}, ...]
    """
    results = []
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code == 200:
            # Simple regex-based extraction of search results from HTML
            # This avoids heavy parsing libraries while still being effective
            # Pattern: <a rel="nofollow" class="result__a" href="...">title</a>
            # and <a class="result__snippet" ...>snippet</a>
            html = resp.text

            # Extract result blocks using a rough heuristic
            result_blocks = re.findall(
                r'<a rel="nofollow" class="result__a" href="(.*?)".*?>(.*?)</a>'
                r'.*?<a class="result__snippet".*?>(.*?)</a>',
                html,
                re.DOTALL,
            )

            for url_str, title_html, snippet_html in result_blocks[:max_results]:
                # Clean up HTML tags in title and snippet
                title = re.sub(r"<.*?>", "", title_html).strip()
                snippet = re.sub(r"<.*?>", "", snippet_html).strip()
                results.append({"title": title, "url": url_str, "snippet": snippet})

    except Exception as e:
        # Silently return partial results on error
        pass

    # Always include Khalifa Fund sources for relevant queries,
    # even if the web search failed
    if any(
        kw in query.lower()
        for kw in ["khalifa", "khalifa fund", "صندوق خليفة"]
    ):
        existing_urls = {r["url"] for r in results}
        for key, url in KHALIFA_FUND_SOURCES.items():
            if url not in existing_urls:
                results.append(
                    {
                        "title": f"Khalifa Fund - {key.title()}",
                        "url": url,
                        "snippet": f"Official Khalifa Fund {key} page.",
                    }
                )

    return results


# ── OPENAI AGENT CLASSES ─────────────────────────────────────────────────────


class AIAgent:
    """Base class for AI agents with shared OpenAI interaction logic."""

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.client = get_openai_client()

    def chat(self, messages: list, temperature: float = 0.7) -> Optional[str]:
        """
        Send a message list to the OpenAI API and return the response text.
        Returns None on failure.
        """
        if not self.client:
            return None
        try:
            system_msg = {"role": "system", "content": self.system_prompt}
            full_messages = [system_msg] + messages
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=full_messages,
                temperature=temperature,
                max_tokens=2000,
                extra_body={"reasoning": {"enabled": False}},
            )
            msg = response.choices[0].message
            return msg.content or ""
        except Exception as e:
            return f"⚠️ Error: {str(e)}"


class ResearchAgent(AIAgent):
    """
    Research Agent: Searches the web, always includes Khalifa Fund for
    Enterprise Development, prioritizes official sources, returns concise
    answers with sources.
    """

    def __init__(self):
        system_prompt = """You are the Research Agent for an AI Entrepreneurship Advisor.
Your responsibilities:
1. Search the web for relevant, up-to-date information on entrepreneurship topics.
2. ALWAYS include Khalifa Fund for Enterprise Development when relevant to the UAE.
3. Prioritize official sources (.gov, .ae, official organizational websites).
4. Return concise, actionable answers with sources cited.
5. If you cannot find a specific answer, say so honestly.

When given search results, synthesize them into a clear answer with sources.
When the user asks about Khalifa Fund programs, provide specific details about
funding, training, coaching, and resources they offer.

Format your response with:
- A clear answer to the question
- Sources listed at the bottom
"""
        super().__init__("Research Agent", system_prompt)

    def research(self, query: str, context: dict = None) -> dict:
        """
        Perform research on a query. Returns a dict with 'answer' and 'sources'.
        Uses the AI model directly with Khalifa Fund context (no web search needed).
        """
        # Build context from Khalifa Fund sources
        search_context = "Khalifa Fund for Enterprise Development official resources:\n"
        for key, url in KHALIFA_FUND_SOURCES.items():
            search_context += f"- {key.title()}: {url}\n"

        # Build the user message
        user_message = f"Research Query: {query}\n\n{search_context}"
        if context:
            user_message += f"\n\nEntrepreneur Context: {json.dumps(context, indent=2)}"

        # Get response from OpenAI
        response = self.chat([{"role": "user", "content": user_message}])

        # Include Khalifa Fund sources
        sources = list(KHALIFA_FUND_SOURCES.values())

        return {
            "answer": response or "I could not find information on that topic.",
            "sources": sources,
            "search_results": [],
        }


class ValidationAgent(AIAgent):
    """
    Validation Agent: Verifies information, resolves conflicts, prioritizes
    official sources, and indicates confidence level.
    """

    def __init__(self):
        system_prompt = """You are the Validation Agent for an AI Entrepreneurship Advisor.
Your responsibilities:
1. Verify information accuracy using available sources.
2. Resolve conflicts between sources by prioritizing official sources.
3. Indicate your confidence level (HIGH, MEDIUM, LOW) for each piece of information.
4. Flag any information that appears outdated or unreliable.
5. Be honest about uncertainty.

When given a claim and supporting sources, evaluate:
- Is the source official and authoritative?
- Is the information current?
- Are there conflicting sources?
- What is the confidence level?

Format your response with:
- Confidence level: [HIGH/MEDIUM/LOW]
- Verification details
- Any caveats or concerns
"""
        super().__init__("Validation Agent", system_prompt)

    def validate(self, claim: str, sources: list = None) -> dict:
        """
        Validate a claim against available sources. Returns a dict with
        'confidence', 'verification', and 'notes'.
        """
        sources_text = ""
        if sources:
            sources_text = "Sources:\n" + "\n".join(f"- {s}" for s in sources)

        user_message = (
            f"Claim to verify: {claim}\n\n{sources_text}\n\n"
            "Please validate this claim and provide confidence level."
        )

        response = self.chat([{"role": "user", "content": user_message}])

        # Parse confidence from response
        confidence = "MEDIUM"
        if response:
            if "HIGH" in response.upper():
                confidence = "HIGH"
            elif "LOW" in response.upper():
                confidence = "LOW"

        return {
            "confidence": confidence,
            "verification": response or "Could not verify.",
            "claim": claim,
        }


class ArabicAgent(AIAgent):
    """
    Arabic Agent: Communicates fluently in English and Arabic and automatically
    uses the user's preferred language.
    """

    def __init__(self):
        system_prompt = """You are the Arabic Agent for an AI Entrepreneurship Advisor.
You are bilingual in English and Arabic.
Your responsibilities:
1. Communicate fluently in both English and Arabic.
2. Automatically detect and use the user's preferred language.
3. When the user writes in Arabic, respond in Arabic.
4. When the user writes in English, respond in English.
5. For bilingual users, you can mix languages naturally.
6. Maintain cultural sensitivity and use appropriate formal/informal registers.
7. Entrepreneurship terms should be clearly translated when needed.

You specialize in entrepreneurship advice for the UAE market, with deep
knowledge of Khalifa Fund programs and local business regulations.
"""
        super().__init__("Arabic Agent", system_prompt)

    def respond(self, user_message: str, context: dict = None) -> str:
        """
        Generate a response in the appropriate language based on user input.
        """
        context_str = ""
        if context:
            context_str = (
                f"\n\nUser Profile Context:\n{json.dumps(context, indent=2)}"
            )

        messages = [
            {
                "role": "user",
                "content": f"{user_message}\n\n{context_str}"
                if context_str
                else user_message,
            }
        ]
        return self.chat(messages, temperature=0.8)


# ── ORCHESTRATOR ─────────────────────────────────────────────────────────────


class AdvisorOrchestrator:
    """
    Orchestrates the three AI agents (Research, Validation, Arabic) to provide
    comprehensive entrepreneurship advice. This replaces n8n's orchestration
    role within the single Streamlit app.
    """

    def __init__(self):
        self.research_agent = ResearchAgent()
        self.validation_agent = ValidationAgent()
        self.arabic_agent = ArabicAgent()
        self.client = get_openai_client()

    def _get_orchestrator_response(
        self, user_message: str, profile: dict, conversation_history: list
    ) -> str:
        """
        Get the main orchestrator response that decides which agents to invoke.
        """
        if not self.client:
            return (
                "⚠️ OpenAI API key is not configured. "
                "Please set your OPENAI_API_KEY in the .env file."
            )

        # Build the system prompt for the orchestrator
        # (the main AI that decides what to do)
        lang = profile.get("language", "English")
        lang_instruction = (
            "Respond in Arabic. Use Arabic for all responses."
            if lang == "Arabic"
            else "Respond in English."
        )

        profile_str = json.dumps(profile, indent=2)
        system_prompt = f"""You are a friendly AI Entrepreneurship Advisor. Be warm, natural, and conversational.

RULES:
- If the user says a casual greeting (hi, hello, hey), greet them back in 1-2 sentences and ask how you can help.
- If they ask about Khalifa Fund, funding, or business topics, give concise helpful advice answering their question directly.
- Do NOT introduce yourself or repeat the user's profile info unless they ask. Just answer their question.
- Keep responses short and natural. Never output JSON, logs, or structured data.
- Do NOT prefix your response with "Current conversation:", "User:", "Assistant:", timestamps, or any metadata. Respond directly as yourself.
- {lang_instruction}

You have access to: Research Agent (web search), Validation Agent (fact-checking), Arabic Agent (bilingual).

Entrepreneur Profile:
{profile_str}
"""

        messages = [{"role": "system", "content": system_prompt}]
        # Add conversation history (last 10 messages for context)
        for msg in conversation_history[-10:]:
            messages.append(msg)
        # Add the user's current question
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                extra_body={"reasoning": {"enabled": False}},
            )
            msg = response.choices[0].message
            content = msg.content or ""
            # Strip any metadata prefixes the model might add
            content = re.sub(
                r"^(Current conversation:|User:|Assistant:|\[?\d{1,2}:\d{2}\s*(AM|PM)?\]?\s*)",
                "", content, flags=re.IGNORECASE | re.MULTILINE
            ).strip()
            return content
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

    def get_response(
        self, user_message: str, profile: dict, conversation_history: list
    ) -> dict:
        """
        Process a user message through the orchestrator and return a response
        with sources and any agent-specific metadata.
        """
        # Get the orchestrator's main response (single API call)
        main_response = self._get_orchestrator_response(
            user_message, profile, conversation_history
        )

        # Provide Khalifa Fund sources when relevant
        kf_keywords = ["khalifa", "fund", "funding", "finance", "program", "grant",
                        "loan", "uae", "abu dhabi", "entrepreneur", "startup",
                        "تمويل", "منحة", "قرض", "برنامج"]
        sources = []
        if any(kw in user_message.lower() for kw in kf_keywords):
            sources = list(KHALIFA_FUND_SOURCES.values())

        # Arabic support
        arabic_response = None
        if self._is_arabic_text(user_message):
            arabic_response = self.arabic_agent.respond(
                user_message, profile
            )

        return {
            "main_response": main_response,
            "sources": sources,
            "research_result": None,
            "validation_result": None,
            "arabic_response": arabic_response,
            "needs_escalation": self._check_escalation_needed(
                user_message, main_response
            ),
        }

    @staticmethod
    def _is_arabic_text(text: str) -> bool:
        """Check if text contains Arabic characters."""
        arabic_pattern = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
        return bool(arabic_pattern.search(text))

    @staticmethod
    def _check_escalation_needed(user_message: str, response: str) -> dict:
        """
        Check if the case should be escalated to a Khalifa Fund advisor.
        Returns a dict with 'needed' and 'reason'.
        """
        escalation_keywords = [
            "complex", "urgent", "legal dispute", "lawsuit", "crisis",
            "bankruptcy", "closure", "investigation", "compliance",
            "معقد", "عاجل", "نزاع", "دعوى", "أزمة", "إفلاس",
        ]
        reason = None
        needed = False

        for kw in escalation_keywords:
            if kw in user_message.lower() or (
                response and kw in response.lower()
            ):
                needed = True
                reason = (
                    "This case involves complex issues that may benefit from "
                    "direct consultation with a Khalifa Fund advisor. "
                    "Recommended actions: schedule a one-on-one session, "
                    "contact the Khalifa Fund support team for personalized guidance."
                )
                break

        return {"needed": needed, "reason": reason}


# ── PERSONALIZED GUIDANCE ────────────────────────────────────────────────────


def generate_learning_plan(profile: dict) -> dict:
    """
    Generate a personalized learning plan with milestones, timeline, coaching
    sessions, and next steps. Uses the OpenAI orchestrator.

    Returns a dict with:
      - milestones: list of {"name", "description", "timeline", "status"}
      - coaching_sessions: list of {"topic", "format", "provider"}
      - next_steps: list of strings
    """
    client = get_openai_client()
    if not client:
        return _default_learning_plan(profile)

    stage = profile.get("business_stage", "Idea / الفكرة")
    industry = profile.get("industry", "Technology / التكنولوجيا")
    location = profile.get("location", "Abu Dhabi")
    challenges = profile.get("challenges", [])
    goals = profile.get("goals", "")

    prompt = f"""Generate a personalized entrepreneurship learning plan in JSON format.

Entrepreneur Profile:
- Business Stage: {stage}
- Industry: {industry}
- Location: {location}
- Challenges: {', '.join(challenges) if challenges else 'Not specified'}
- Goals: {goals if goals else 'Not specified'}

The plan must include:
1. milestones: 3-5 milestones with name, description, timeline (in weeks), and status (all "pending")
2. coaching_sessions: 2-4 coaching sessions with topic, format (e.g., "One-on-One", "Workshop", "Online Course"), and provider (prefer "Khalifa Fund" where relevant)
3. next_steps: 3-5 actionable next steps as strings

Return ONLY valid JSON with this structure:
{{
  "milestones": [
    {{"name": "...", "description": "...", "timeline": "2 weeks", "status": "pending"}}
  ],
  "coaching_sessions": [
    {{"topic": "...", "format": "...", "provider": "Khalifa Fund"}}
  ],
  "next_steps": ["...", "..."]
}}

Include Khalifa Fund programs, training, and resources where relevant.
The user is in {location}, UAE, so include UAE-specific resources.
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a business planning expert. Generate learning plans in JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"},
            extra_body={"reasoning": {"enabled": False}},
        )
        msg = response.choices[0].message
        content = msg.content or "{}"
        plan = json.loads(content)
        return plan
    except Exception:
        return _default_learning_plan(profile)


def _default_learning_plan(profile: dict) -> dict:
    """Return a sensible default learning plan when the API fails."""
    stage = profile.get("business_stage", "Idea / الفكرة")

    milestones = [
        {
            "name": "Business Model Canvas",
            "description": "Complete a Business Model Canvas for your idea",
            "timeline": "2 weeks",
            "status": "pending",
        },
        {
            "name": "Market Research",
            "description": "Conduct market research and competitor analysis",
            "timeline": "4 weeks",
            "status": "pending",
        },
        {
            "name": "Financial Planning",
            "description": "Develop financial projections and funding strategy",
            "timeline": "6 weeks",
            "status": "pending",
        },
        {
            "name": "Khalifa Fund Application",
            "description": "Prepare and submit Khalifa Fund program application",
            "timeline": "8 weeks",
            "status": "pending",
        },
        {
            "name": "Launch Plan",
            "description": "Finalize launch strategy and execution plan",
            "timeline": "12 weeks",
            "status": "pending",
        },
    ]

    coaching_sessions = [
        {
            "topic": "Business Planning Fundamentals",
            "format": "Workshop",
            "provider": "Khalifa Fund",
        },
        {
            "topic": "Funding & Financial Strategy",
            "format": "One-on-One Coaching",
            "provider": "Khalifa Fund",
        },
        {
            "topic": "Marketing & Market Entry",
            "format": "Online Course",
            "provider": "Khalifa Fund",
        },
    ]

    next_steps = [
        "Register for a Khalifa Fund introductory workshop",
        "Complete a Business Model Canvas",
        "Research your target market and competitors",
        "Prepare a preliminary financial plan",
        "Schedule a consultation with a Khalifa Fund advisor",
    ]

    return {
        "milestones": milestones,
        "coaching_sessions": coaching_sessions,
        "next_steps": next_steps,
    }


# ── PROGRESS TRACKING ────────────────────────────────────────────────────────


def update_milestone(milestone_name: str, new_status: str):
    """Update the status of a milestone in session state."""
    if "learning_plan" not in st.session_state:
        return
    plan = st.session_state.learning_plan
    for m in plan.get("milestones", []):
        if m["name"] == milestone_name:
            m["status"] = new_status
            break


def render_progress_tracker():
    """Render the progress tracking section in the sidebar or main area."""
    plan = st.session_state.get("learning_plan")
    if plan is None:
        plan = {}
    milestones = plan.get("milestones", [])
    coaching = plan.get("coaching_sessions", [])
    next_steps = plan.get("next_steps", [])

    if not milestones:
        st.info("No learning plan yet. Start chatting to generate one!")
        return

    lang = st.session_state.get("language", "English")

    # Milestones with progress
    completed = sum(1 for m in milestones if m["status"] == "completed")
    total = len(milestones)
    progress_pct = completed / total if total > 0 else 0

    st.subheader(f"📊 {translate_label('Progress Tracking', lang)}")
    st.progress(progress_pct)
    st.caption(f"{completed}/{total} {translate_label('Completed', lang).lower()}")

    # Milestone list with toggle-able status
    st.subheader(f"🎯 {translate_label('Milestones', lang)}")
    for m in milestones:
        col1, col2 = st.columns([3, 1])
        with col1:
            status_icon = {
                "completed": "✅",
                "in_progress": "🔄",
                "pending": "⏳",
            }.get(m["status"], "⏳")
            st.markdown(f"{status_icon} **{m['name']}** — {m['description']}")
            st.caption(f"⏱ {m['timeline']}")
        with col2:
            new_status = st.selectbox(
                "",
                ["pending", "in_progress", "completed"],
                index=["pending", "in_progress", "completed"].index(
                    m["status"]
                ),
                key=f"milestone_{m['name']}",
                label_visibility="collapsed",
            )
            if new_status != m["status"]:
                update_milestone(m["name"], new_status)
                st.rerun()

    # Coaching sessions
    if coaching:
        st.subheader(f"📚 {translate_label('Personalized Guidance', lang)}")
        st.markdown("**Coaching Sessions:**")
        for s in coaching:
            st.markdown(
                f"- **{s['topic']}** ({s['format']}) — {s['provider']}"
            )

    # Next steps
    if next_steps:
        st.markdown("**Next Steps:**")
        for i, step in enumerate(next_steps, 1):
            st.markdown(f"{i}. {step}")


# ── SIDEBAR: ENTREPRENEUR PROFILE ────────────────────────────────────────────


def render_sidebar():
    """Render the sidebar with entrepreneur profile form and progress tracking."""
    with st.sidebar:
        st.title("🧑‍💼 Riyada AI")
        st.markdown("---")

        # Language selector at the top
        lang = st.session_state.get("language", "English")
        st.session_state.language = st.selectbox(
            translate_label("Preferred Language", lang),
            ["English", "Arabic"],
            index=0 if lang == "English" else 1,
            key="language_selector",
        )

        # Re-read language after possible change
        lang = st.session_state.language

        # Profile section
        st.header(translate_label("Name", lang))
        st.session_state.profile["name"] = st.text_input(
            translate_label("Name", lang),
            value=st.session_state.profile.get("name", ""),
            placeholder="e.g., Ahmed Al Mansouri",
            label_visibility="collapsed",
        )

        st.header(translate_label("Business Stage", lang))
        st.session_state.profile["business_stage"] = st.selectbox(
            translate_label("Business Stage", lang),
            BUSINESS_STAGES,
            index=BUSINESS_STAGES.index(
                st.session_state.profile.get("business_stage", BUSINESS_STAGES[0])
            )
            if st.session_state.profile.get("business_stage") in BUSINESS_STAGES
            else 0,
            label_visibility="collapsed",
        )

        st.header(translate_label("Industry / Sector", lang))
        st.session_state.profile["industry"] = st.selectbox(
            translate_label("Industry / Sector", lang),
            INDUSTRIES,
            index=INDUSTRIES.index(
                st.session_state.profile.get("industry", INDUSTRIES[0])
            )
            if st.session_state.profile.get("industry") in INDUSTRIES
            else 0,
            label_visibility="collapsed",
        )

        st.header(translate_label("Business Location", lang))
        st.session_state.profile["location"] = st.text_input(
            translate_label("Business Location", lang),
            value=st.session_state.profile.get("location", "Abu Dhabi"),
            placeholder="e.g., Abu Dhabi, Dubai, Sharjah",
            label_visibility="collapsed",
        )

        st.header(translate_label("History with Khalifa Fund", lang))
        st.session_state.profile["khalifa_fund_history"] = st.text_area(
            translate_label("History with Khalifa Fund", lang),
            value=st.session_state.profile.get("khalifa_fund_history", ""),
            placeholder="Have you interacted with Khalifa Fund before?",
            label_visibility="collapsed",
            height=80,
        )

        st.header(translate_label("Current Challenges", lang))
        selected_challenges = st.multiselect(
            translate_label("Current Challenges", lang),
            COMMON_CHALLENGES,
            default=st.session_state.profile.get("challenges", []),
            label_visibility="collapsed",
        )
        st.session_state.profile["challenges"] = selected_challenges

        # Custom challenges
        custom_challenge = st.text_input(
            "Other challenge (optional)",
            placeholder="Add a challenge not listed...",
            key="custom_challenge_input",
        )
        if custom_challenge and st.button("➕ Add Challenge"):
            if custom_challenge not in st.session_state.profile.get("challenges", []):
                st.session_state.profile.setdefault("challenges", []).append(
                    custom_challenge
                )
                st.rerun()

        st.header(translate_label("Goals", lang))
        st.session_state.profile["goals"] = st.text_area(
            translate_label("Goals", lang),
            value=st.session_state.profile.get("goals", ""),
            placeholder="What are your main business goals?",
            label_visibility="collapsed",
            height=100,
        )

        # Save profile button
        if st.button(
            translate_label("Save Profile", lang),
            type="primary",
            use_container_width=True,
        ):
            st.session_state.profile_saved = True
            st.success(translate_label("Profile Saved", lang))

        st.markdown("---")

        # Progress tracking section in sidebar
        render_progress_tracker()

        st.markdown("---")
        st.caption("Built with Streamlit + OpenRouter")


# ── CHAT INTERFACE ───────────────────────────────────────────────────────────


def render_chat():
    """Render the main chat interface with message history and input."""
    lang = st.session_state.get("language", "English")

    # Title
    title = translate_label("Chat with your AI Advisor", lang)
    st.title(f"💬 {title}")

    # Check if profile is saved
    if not st.session_state.profile_saved:
        st.info(
            "👈 Please fill in your profile in the sidebar and click **Save Profile** to start."
            if lang == "English"
            else "👈 يرجى ملء ملفك الشخصي في الشريط الجانبي والنقر على **حفظ الملف الشخصي** للبدء."
        )
        return

    # Check API key
    if not get_openai_client():
        st.error(
            "⚠️ OpenAI API key is not configured. "
            "Please create a `.env` file in the project root with:\n\n"
            "```\nOPENAI_API_KEY=your-api-key-here\n```\n\n"
            "Then restart the app."
        )
        return

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show sources if present
            if msg.get("sources"):
                st.markdown(format_sources(msg["sources"]))
            # Show escalation notice if present
            if msg.get("escalation"):
                st.warning(msg["escalation"])

    # Chat input
    placeholder = translate_label("Type your message here...", lang)
    send_label = translate_label("Send", lang)
    clear_label = translate_label("Clear Chat", lang)

    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.chat_input(placeholder)
    with col2:
        if st.button(clear_label, use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Process user input
    if user_input:
        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(
                "Thinking..." if lang == "English" else "جاري التفكير..."
            ):
                orchestrator = AdvisorOrchestrator()
                result = orchestrator.get_response(
                    user_message=user_input,
                    profile=st.session_state.profile,
                    conversation_history=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ],
                )

                response_text = result["main_response"]
                sources = result["sources"]
                escalation = result.get("needs_escalation", {})

                # Display the response
                st.markdown(response_text)

                # Show sources
                if sources:
                    st.markdown(format_sources(sources))

                # Show validation if available
                validation = result.get("validation_result")
                if validation and validation.get("confidence"):
                    confidence_icon = {
                        "HIGH": "🟢",
                        "MEDIUM": "🟡",
                        "LOW": "🔴",
                    }.get(validation["confidence"], "🟡")
                    st.caption(
                        f"{confidence_icon} Confidence: {validation['confidence']}"
                    )

                # Handle escalation
                escalation_text = None
                if escalation.get("needed"):
                    escalation_text = escalation["reason"]
                    st.warning(escalation_text)

                # Store assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources,
                }
                if escalation_text:
                    assistant_msg["escalation"] = escalation_text
                st.session_state.messages.append(assistant_msg)

                # Generate learning plan if not yet generated
                if "learning_plan" not in st.session_state:
                    with st.spinner(
                        "Generating your personalized learning plan..."
                        if lang == "English"
                        else "جاري إنشاء خطة التعلم المخصصة..."
                    ):
                        plan = generate_learning_plan(st.session_state.profile)
                        st.session_state.learning_plan = plan
                        st.success(
                            "✅ Personalized learning plan generated! Check the sidebar."
                            if lang == "English"
                            else "✅ تم إنشاء خطة التعلم المخصصة! تحقق من الشريط الجانبي."
                        )
                        time.sleep(1)
                        st.rerun()


# ── MAIN APP ─────────────────────────────────────────────────────────────────


def initialize_session_state():
    """Initialize all session state variables."""
    if "profile" not in st.session_state:
        st.session_state.profile = {
            "name": "",
            "language": "English",
            "business_stage": BUSINESS_STAGES[0],
            "industry": INDUSTRIES[0],
            "location": "Abu Dhabi",
            "khalifa_fund_history": "",
            "challenges": [],
            "goals": "",
        }
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "profile_saved" not in st.session_state:
        st.session_state.profile_saved = False
    if "language" not in st.session_state:
        st.session_state.language = "English"
    if "learning_plan" not in st.session_state:
        st.session_state.learning_plan = None
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = AdvisorOrchestrator()


def main():
    """Main entry point for the Streamlit app."""
    # Page config
    st.set_page_config(
        page_title="Riyada AI",
        page_icon="🧑‍💼",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for a modern look
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .stChatMessage {
            background-color: white;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stSidebar {
            background-color: #f0f2f6;
        }
        .stProgress > div > div > div > div {
            background-color: #00a67e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize state
    initialize_session_state()

    # Render sidebar
    render_sidebar()

    # Render main chat area
    render_chat()


if __name__ == "__main__":
    main()