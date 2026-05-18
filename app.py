import os
import cohere
import streamlit as st

# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="Decision Court AI",
    page_icon="⚖️",
    layout="wide"
)

st.title("Decision Court AI")
st.caption("Put your decision on trial. Two AI agents argue both sides, then a judge gives the verdict.")

# -----------------------------
# API setup
# -----------------------------

def get_api_key():
    """Reads the Cohere API key from Streamlit secrets or environment variables."""
    try:
        return st.secrets["COHERE_API_KEY"]
    except Exception:
        return os.getenv("COHERE_API_KEY")


api_key = get_api_key()

if not api_key:
    st.error("Missing COHERE_API_KEY. Add it in Streamlit secrets before running the app.")
    st.stop()

co = cohere.ClientV2(api_key=api_key)


# -----------------------------
# Helper function
# -----------------------------

def ask_agent(system_prompt, user_prompt, temperature=0.4):
    """
    Sends a prompt to Cohere and returns the model's text response.
    Uses Cohere ClientV2 chat format.
    """
    try:
        response = co.chat(
            model="command-r-plus-08-2024",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )

        return response.message.content[0].text

    except Exception as e:
        return f"Error: {e}"


# -----------------------------
# Sidebar controls
# -----------------------------

st.sidebar.header("Case Settings")

decision_type = st.sidebar.selectbox(
    "Decision type",
    [
        "School",
        "Money",
        "Social",
        "Productivity",
        "Career",
        "Personal",
        "Other"
    ]
)

main_goal = st.sidebar.selectbox(
    "What matters most?",
    [
        "Long-term growth",
        "Happiness",
        "Grades / achievement",
        "Money",
        "Health",
        "Relationships",
        "Low stress"
    ]
)

risk_tolerance = st.sidebar.slider(
    "Risk tolerance",
    min_value=1,
    max_value=10,
    value=5,
    help="1 means avoid risk. 10 means you are comfortable taking big risks."
)

urgency = st.sidebar.slider(
    "Urgency",
    min_value=1,
    max_value=10,
    value=5,
    help="1 means you have time. 10 means you need to decide immediately."
)

judge_style = st.sidebar.selectbox(
    "Judge style",
    [
        "Balanced",
        "Brutally honest",
        "Long-term focused",
        "Low-risk"
    ]
)


# -----------------------------
# Main input
# -----------------------------

decision = st.text_area(
    "What decision are you stuck on?",
    placeholder="Example: Should I quit soccer to focus more on school?",
    height=130
)

extra_context = st.text_area(
    "Optional context",
    placeholder="Add details like deadlines, people involved, what you already tried, or what you are worried about.",
    height=100
)

run_button = st.button("Run Decision Court", type="primary")


# -----------------------------
# Prompt templates
# -----------------------------

PRO_PROMPT = """
You are the Pro Advocate in Decision Court.

Your job is to make the strongest realistic argument FOR the user's decision.

Rules:
- Be specific to the user's situation.
- Do not give generic motivational advice.
- Mention the best upside, hidden benefits, and why the decision could be smart.
- Acknowledge risks only briefly.
- Do not make the final decision.
- Keep the response clear and concise.
"""

CON_PROMPT = """
You are the Con Advocate in Decision Court.

Your job is to make the strongest realistic argument AGAINST the user's decision.

Rules:
- Be specific to the user's situation.
- Focus on risks, weak assumptions, opportunity costs, and possible regret.
- Do not exaggerate.
- Do not make the final decision.
- Keep the response clear and concise.
"""

CROSS_EXAM_PROMPT = """
You are the Cross Examiner in Decision Court.

You will receive the user's situation, the Pro Advocate's argument, and the Con Advocate's argument.

Your job:
1. Identify the strongest point from the Pro side.
2. Identify the strongest point from the Con side.
3. Attack the weakest assumption in the Pro argument.
4. Attack the weakest assumption in the Con argument.
5. List the key missing information that would improve the verdict.

Rules:
- Do not make the final decision.
- Be direct.
- Keep the response organized.
"""

JUDGE_PROMPT = """
You are the Judge in Decision Court.

You must give a final verdict based on the user's situation, goal, risk tolerance, urgency, judge style, and the arguments from both sides.

Output exactly in this format:

## Final Verdict
Choose one:
- Do it
- Do not do it
- Delay and gather more information

## Confidence Score
Give a score out of 100.

## Why This Verdict Wins
Explain the main reason.

## Best Counterargument
Give the strongest reason your verdict could be wrong.

## Next 24 Hours
Give a practical next step the user should take.

## Mistake to Avoid
Give one mistake the user should avoid.

Rules:
- Be clear and decisive.
- Do not be vague.
- Do not say both sides are equally valid unless the evidence truly supports that.
- Match the judge style requested by the user.
"""


# -----------------------------
# App logic
# -----------------------------

if run_button:
    if not decision.strip():
        st.warning("Enter a decision first.")
        st.stop()

    user_context = f"""
Decision type: {decision_type}
User's decision: {decision}
Extra context: {extra_context if extra_context.strip() else "No extra context provided."}
Main goal: {main_goal}
Risk tolerance: {risk_tolerance}/10
Urgency: {urgency}/10
Judge style: {judge_style}
"""

    with st.spinner("The Pro Advocate is building the case..."):
        pro_case = ask_agent(PRO_PROMPT, user_context, temperature=0.45)

    with st.spinner("The Con Advocate is building the opposition..."):
        con_case = ask_agent(CON_PROMPT, user_context, temperature=0.45)

    cross_context = f"""
User context:
{user_context}

Pro Advocate argument:
{pro_case}

Con Advocate argument:
{con_case}
"""

    with st.spinner("Cross Examiner is testing both arguments..."):
        cross_exam = ask_agent(CROSS_EXAM_PROMPT, cross_context, temperature=0.35)

    judge_context = f"""
User context:
{user_context}

Pro Advocate argument:
{pro_case}

Con Advocate argument:
{con_case}

Cross Examination:
{cross_exam}
"""

    with st.spinner("Judge is deciding the final verdict..."):
        verdict = ask_agent(JUDGE_PROMPT, judge_context, temperature=0.25)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pro Advocate")
        st.markdown(pro_case)

    with col2:
        st.subheader("Con Advocate")
        st.markdown(con_case)

    st.subheader("Cross-Examination")
    st.markdown(cross_exam)

    st.subheader("Judge's Verdict")
    st.markdown(verdict)

    st.divider()

