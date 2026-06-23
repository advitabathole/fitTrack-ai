import os

import ollama
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FitTrack AI",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = "fitness_data.csv"
MODEL = "llama3.2"


def load_data():
    if os.path.exists(DATA_FILE):
        data = pd.read_csv(DATA_FILE)

        if data.empty:
            return None

        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date")
        data["date"] = data["date"].dt.strftime("%Y-%m-%d")

        return data

    return None


def ask_ai(prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def save_entry(entry_date, goal_weight, weight, calories, protein, workout):
    new_entry = pd.DataFrame(
        {
            "date": [entry_date.strftime("%Y-%m-%d")],
            "goal_weight": [goal_weight],
            "weight": [weight],
            "calories": [calories],
            "protein": [protein],
            "workout": [workout],
        }
    )

    data = load_data()

    if data is not None:
        data = pd.concat([data, new_entry], ignore_index=True)
    else:
        data = new_entry

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")
    data["date"] = data["date"].dt.strftime("%Y-%m-%d")

    data.to_csv(DATA_FILE, index=False)


def get_goal_progress(data):
    latest = data.iloc[-1]
    starting_weight = data.iloc[0]["weight"]
    current_weight = latest["weight"]
    goal_weight = latest["goal_weight"]

    if starting_weight == goal_weight:
        return 0

    progress = (starting_weight - current_weight) / (
        starting_weight - goal_weight
    )

    return max(0, min(progress, 1))


st.sidebar.title("🏋️ FitTrack AI")
st.sidebar.caption("Personal fitness accountability dashboard")
st.sidebar.divider()
st.sidebar.metric("Coach Mode", "Direct")
st.sidebar.metric("AI Model", MODEL)
st.sidebar.metric("Storage", "Local CSV")
st.sidebar.divider()
st.sidebar.caption("Built with Python, Streamlit, pandas, and Ollama.")

st.title("🏋️ FitTrack AI")
st.caption(
    "Track workouts, nutrition, progress trends, and AI coaching in one dashboard."
)

data = load_data()

if data is not None:
    latest = data.iloc[-1]

    top_col1, top_col2, top_col3, top_col4 = st.columns(4)

    top_col1.metric("Current Weight", f"{latest['weight']} lbs")
    top_col2.metric("Protein", f"{latest['protein']} g")
    top_col3.metric("Calories", f"{latest['calories']}")
    top_col4.metric("Entries", f"{len(data)}")
else:
    st.info("Start by saving your first daily check-in.")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Daily Check-In",
        "Progress",
        "AI Coach",
        "Settings",
    ]
)


with tab1:
    left, right = st.columns([1.1, 0.9])

    with left:
        with st.container(border=True):
            st.subheader("Check-In Entry")

            entry_date = st.date_input("Entry date")

            goal_weight = st.number_input(
                "Goal weight",
                min_value=70.0,
                max_value=250.0,
                value=110.0,
                step=0.1,
            )

            weight = st.number_input(
                "Weight",
                min_value=70.0,
                max_value=250.0,
                value=116.6,
                step=0.1,
            )

            workout = st.selectbox(
                "Workout",
                [
                    "Rest Day",
                    "Upper Body",
                    "Lower Body",
                    "Cardio",
                    "Full Body",
                ],
            )

            calories = st.number_input(
                "Calories eaten",
                min_value=0,
                max_value=5000,
                value=1200,
                step=50,
            )

            protein = st.number_input(
                "Protein eaten",
                min_value=0,
                max_value=300,
                value=120,
                step=5,
            )

            if st.button("Save Check-In", use_container_width=True):
                save_entry(
                    entry_date,
                    goal_weight,
                    weight,
                    calories,
                    protein,
                    workout,
                )
                st.success("Check-in saved. Refresh to update dashboard.")

    with right:
        with st.container(border=True):
            st.subheader("Daily Targets")
            st.metric("Protein Target", "100–130 g")
            st.metric("Calories", "Controlled deficit")
            st.metric("Workout Focus", "Lift + steps")
            st.caption(
                "These are general targets. The AI coach uses your logged data for more specific feedback."
            )

        with st.container(border=True):
            st.subheader("Project Value")
            st.write(
                "This app tracks user behavior, stores structured data, visualizes trends, and uses a local LLM for personalized feedback."
            )


with tab2:
    st.subheader("Progress Dashboard")

    data = load_data()

    if data is not None:
        latest = data.iloc[-1]

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric("Latest Weight", f"{latest['weight']} lbs")
        metric2.metric("Goal Weight", f"{latest['goal_weight']} lbs")
        metric3.metric("Protein", f"{latest['protein']} g")
        metric4.metric("Calories", f"{latest['calories']}")

        progress = get_goal_progress(data)

        with st.container(border=True):
            st.subheader("Goal Progress")
            st.progress(progress)
            st.caption(f"{progress * 100:.1f}% toward your logged goal")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            with st.container(border=True):
                st.subheader("Weight Trend")
                st.line_chart(data.set_index("date")["weight"])

        with chart_col2:
            with st.container(border=True):
                st.subheader("Protein Trend")
                st.line_chart(data.set_index("date")["protein"])

        with st.container(border=True):
            st.subheader("Calories Trend")
            st.line_chart(data.set_index("date")["calories"])

        with st.expander("View logged entries"):
            st.dataframe(data, use_container_width=True)

        with st.container(border=True):
            st.subheader("AI Trend Analysis")

            if st.button("Analyze My Trends", use_container_width=True):
                prompt = f"""
Analyze this fitness data:

{data.tail(30).to_string(index=False)}

Find:
1. Patterns
2. Good habits
3. Bad habits
4. One thing slowing progress
5. One recommendation

Be specific, direct, and concise.
Do not encourage unsafe dieting.
"""

                with st.spinner("Analyzing trends..."):
                    st.write(ask_ai(prompt))
    else:
        st.info("No entries saved yet. Add your first check-in.")


with tab3:
    st.subheader("AI Coach")

    coach_left, coach_right = st.columns(2)

    with coach_left:
        with st.container(border=True):
            st.subheader("Coach Input")

            coach_weight = st.number_input(
                "Current weight",
                min_value=70.0,
                max_value=250.0,
                value=116.6,
                step=0.1,
                key="coach_weight",
            )

            coach_calories = st.number_input(
                "Calories today",
                min_value=0,
                max_value=5000,
                value=1200,
                step=50,
                key="coach_calories",
            )

            coach_protein = st.number_input(
                "Protein today",
                min_value=0,
                max_value=300,
                value=120,
                step=5,
                key="coach_protein",
            )

            coach_workout = st.selectbox(
                "Workout",
                [
                    "Rest Day",
                    "Upper Body",
                    "Lower Body",
                    "Cardio",
                    "Full Body",
                ],
                key="coach_workout",
            )

            goal = st.text_area(
                "Goal",
                value="Get leaner, build glutes, and get more defined.",
                key="coach_goal",
            )

            if st.button("Get Coach Feedback", use_container_width=True):
                prompt = f"""
You are a realistic fitness coach.

Weight: {coach_weight} lbs
Calories: {coach_calories}
Protein: {coach_protein} g
Workout: {coach_workout}
Goal: {goal}

Give:
1. Today's grade out of 10.
2. What went well.
3. What to improve tomorrow.
4. One actionable tip.

Do not encourage extreme dieting.
Keep it concise.
"""

                with st.spinner("Analyzing..."):
                    st.write(ask_ai(prompt))

    with coach_right:
        with st.container(border=True):
            st.subheader("Tomorrow's Plan")

            data = load_data()

            if data is not None:
                recent_data = data.tail(7).to_string(index=False)

                if st.button(
                    "Generate Tomorrow's Plan",
                    use_container_width=True,
                ):
                    prompt = f"""
You are a realistic fitness coach.

User goal:
{goal}

Recent tracking data:
{recent_data}

Create tomorrow's plan with:
1. Calories target
2. Protein target
3. Workout plan
4. One thing to avoid
5. One non-negotiable action

Be direct but healthy.
Do not recommend extreme dieting.
"""

                    with st.spinner("Planning tomorrow..."):
                        st.write(ask_ai(prompt))
            else:
                st.info("Save at least one entry first.")

        with st.container(border=True):
            st.subheader("Progress Predictor")

            data = load_data()

            if data is not None:
                recent_data = data.tail(14).to_string(index=False)

                if st.button(
                    "Predict My Progress",
                    use_container_width=True,
                ):
                    prompt = f"""
You are a realistic, healthy fitness coach.

User goal:
{goal}

Recent tracking data:
{recent_data}

Based on consistency, estimate realistic progress in:
- 4 weeks
- 8 weeks
- 12 weeks

Include:
1. Expected weight trend
2. Expected physique changes
3. Biggest risk
4. What the user must do consistently

Do not promise extreme results.
Do not encourage unsafe dieting.
Be direct.
"""

                    with st.spinner("Predicting progress..."):
                        st.write(ask_ai(prompt))
            else:
                st.info("Save at least one entry first.")

    with st.container(border=True):
        st.subheader("Weekly Check-In")

        data = load_data()

        if data is not None and len(data) >= 3:
            weekly_data = data.tail(7).to_string(index=False)

            if st.button(
                "Generate Weekly Check-In",
                use_container_width=True,
            ):
                prompt = f"""
You are a direct but healthy fitness coach.

User goal:
{goal}

Last 7 logged entries:
{weekly_data}

Analyze:
1. Overall consistency grade out of 10
2. Protein consistency
3. Calorie consistency
4. Workout consistency
5. Biggest pattern
6. What to fix next week
7. One non-negotiable rule

Do not encourage unsafe dieting.
Be honest and concise.
"""

                with st.spinner("Reviewing your week..."):
                    st.write(ask_ai(prompt))
        else:
            st.info("Save at least 3 entries to generate a weekly check-in.")


with tab4:
    st.subheader("Settings")

    with st.container(border=True):
        st.warning("These actions affect your saved local data.")

        setting_col1, setting_col2 = st.columns(2)

        with setting_col1:
            if st.button("Delete Last Entry", use_container_width=True):
                data = load_data()

                if data is not None and len(data) > 0:
                    data = data.iloc[:-1]
                    data.to_csv(DATA_FILE, index=False)
                    st.success("Last entry deleted. Refresh the page.")
                else:
                    st.warning("No entries to delete.")

        with setting_col2:
            if st.button("Clear All Data", use_container_width=True):
                if os.path.exists(DATA_FILE):
                    os.remove(DATA_FILE)
                    st.success("All data deleted. Refresh the page.")
                else:
                    st.warning("No data file found.")