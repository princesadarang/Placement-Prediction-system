import streamlit as st
import pandas as pd
import pickle

from io import BytesIO

from company_recommendation import suggest_companies

from mcq_data import (
    APTITUDE_QUESTIONS,
    TECHNICAL_QUESTIONS,
    CODING_QUESTIONS,
    COMMUNICATION_QUESTIONS
)

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm



# PAGE CONFIG


st.set_page_config(
    page_title="Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)



# SESSION STATE


if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "placement_probability" not in st.session_state:
    st.session_state.placement_probability = 0.0

if "recommended_companies" not in st.session_state:
    st.session_state.recommended_companies = []



# LOAD MODEL

@st.cache_resource
def load_model():

    try:

        with open(
            "placement_model.pkl",
            "rb"
        ) as file:

            return pickle.load(file)

    except FileNotFoundError:

        return None


model = load_model()



# DARK GLASSMORPHISM CSS

st.markdown(
    """
    <style>

    .stApp {

        background:

        radial-gradient(
            circle at 10% 10%,
            rgba(0, 153, 255, 0.14),
            transparent 30%
        ),

        radial-gradient(
            circle at 90% 20%,
            rgba(132, 0, 255, 0.14),
            transparent 30%
        ),

        radial-gradient(
            circle at 50% 100%,
            rgba(0, 212, 255, 0.08),
            transparent 35%
        ),

        linear-gradient(
            135deg,
            #050816 0%,
            #080b1c 45%,
            #050713 100%
        );

        color: white;

    }


    .block-container {

        max-width: 1250px;

        padding-top: 2rem;

        padding-bottom: 4rem;

    }


    .main-title {

        text-align: center;

        font-size: 48px;

        font-weight: 900;

        background:
        linear-gradient(
            90deg,
            #00c6ff,
            #7c3aed,
            #c084fc
        );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;

        margin-bottom: 5px;

    }


    .subtitle {

        text-align: center;

        color: #a7b0c5;

        font-size: 17px;

        margin-bottom: 30px;

    }


    h1,
    h2,
    h3,
    h4 {

        color: white !important;

    }


    label {

        color: #dce4f7 !important;

        font-weight: 700 !important;

    }


    .stTextInput input,
    .stNumberInput input {

        background:
        rgba(
            255,
            255,
            255,
            0.055
        ) !important;

        color: white !important;

        border:
        1px solid
        rgba(
            120,
            160,
            255,
            0.25
        ) !important;

        border-radius: 14px !important;

        padding: 13px 15px !important;

    }


    div[data-baseweb="select"] > div {

        background:
        rgba(
            255,
            255,
            255,
            0.055
        ) !important;

        border-radius: 14px !important;

        border:
        1px solid
        rgba(
            120,
            160,
            255,
            0.25
        ) !important;

    }


    .glass-card {

        background:

        linear-gradient(
            135deg,
            rgba(
                255,
                255,
                255,
                0.075
            ),
            rgba(
                255,
                255,
                255,
                0.025
            )
        );

        border:
        1px solid
        rgba(
            145,
            165,
            255,
            0.18
        );

        border-radius: 22px;

        padding: 25px;

        margin-bottom: 20px;

        box-shadow:

        0 20px 50px
        rgba(
            0,
            0,
            0,
            0.28
        );

    }


    .stButton > button {

        border: none !important;

        border-radius: 14px !important;

        min-height: 50px;

        font-size: 16px !important;

        font-weight: 800 !important;

        color: white !important;

        background:

        linear-gradient(
            90deg,
            #008cff,
            #6d28d9
        ) !important;

    }


    [data-testid="stMetric"] {

        background:
        rgba(
            255,
            255,
            255,
            0.045
        );

        border:
        1px solid
        rgba(
            100,
            160,
            255,
            0.18
        );

        padding: 18px;

        border-radius: 18px;

    }


    [data-testid="stMetricValue"] {

        color: white !important;

        font-weight: 900 !important;

    }


    button[data-baseweb="tab"] {

        font-size: 16px !important;

        font-weight: 800 !important;

        color: #aeb8ce !important;

    }


    button[data-baseweb="tab"][aria-selected="true"] {

        color: #00c6ff !important;

    }


    div[role="radiogroup"] label {

        background:
        rgba(
            255,
            255,
            255,
            0.035
        );

        padding: 10px;

        border-radius: 10px;

        margin-bottom: 5px;

    }


    .footer {

        text-align: center;

        color: #667085;

        margin-top: 50px;

        font-size: 13px;

    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL CHECK
# =========================================================

if model is None:

    st.error(
        "⚠️ placement_model.pkl not found. "
        "First run: python train_model.py"
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        🎓 Placement Prediction
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="subtitle">
        AI-powered placement prediction •
        Skill assessment •
        Career dashboard •
        Company recommendation
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# STUDENT INFORMATION
# =========================================================

st.markdown(
    '<div class="glass-card">',
    unsafe_allow_html=True
)


st.subheader(
    "👨‍🎓 Student Information"
)


col1, col2 = st.columns(2)


with col1:

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter your full name"
    )


    email = st.text_input(
        "Email Address",
        placeholder="example@gmail.com"
    )


    phone = st.text_input(
        "Phone Number",
        placeholder="Enter phone number"
    )


    college_name = st.text_input(
        "College Name",
        placeholder="Enter your college name"
    )


with col2:

    branch = st.text_input(
        "Branch",
        placeholder="CSE / AIML / IT / ECE"
    )


    semester = st.number_input(
        "Year / Semester",
        min_value=1,
        max_value=8,
        value=1,
        step=1
    )


    cgpa = st.number_input(
        "CGPA (0–10)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.01
    )


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# =========================================================
# SKILLS
# =========================================================

st.markdown(
    '<div class="glass-card">',
    unsafe_allow_html=True
)


st.subheader(
    "📊 Skills & Performance"
)


col1, col2, col3 = st.columns(3)


with col1:

    aptitude_score = st.number_input(
        "Aptitude Score (0–100)",
        min_value=0,
        max_value=100,
        value=0
    )


    technical_skill_score = st.number_input(
        "Technical Skill Score (0–100)",
        min_value=0,
        max_value=100,
        value=0
    )


with col2:

    communication_skill_score = st.number_input(
        "Communication Skill Score (0–100)",
        min_value=0,
        max_value=100,
        value=0
    )


    coding_score = st.number_input(
        "Coding Score (0–100)",
        min_value=0,
        max_value=100,
        value=0
    )


with col3:

    internship_experience = st.number_input(
        "Internship Experience",
        min_value=0,
        max_value=20,
        value=0
    )


    projects = st.number_input(
        "Number of Projects",
        min_value=0,
        max_value=50,
        value=0
    )


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# =========================================================
# ACADEMIC DETAILS
# =========================================================

st.markdown(
    '<div class="glass-card">',
    unsafe_allow_html=True
)


st.subheader(
    "📚 Academic Details"
)


col1, col2 = st.columns(2)


with col1:

    backlogs = st.number_input(
        "Number of Backlogs",
        min_value=0,
        max_value=20,
        value=0
    )


with col2:

    certifications = st.number_input(
        "Number of Certifications",
        min_value=0,
        max_value=50,
        value=0
    )


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# =========================================================
# PREDICT BUTTON
# =========================================================

predict_button = st.button(
    "🚀 Predict My Placement",
    type="primary",
    use_container_width=True
)


# =========================================================
# MCQ TEST FUNCTION
# =========================================================

def run_mcq_test(
    title,
    questions,
    session_key
):

    st.subheader(title)


    score_key = (
        f"{session_key}_score"
    )

    submitted_key = (
        f"{session_key}_submitted"
    )

    answers_key = (
        f"{session_key}_answers"
    )


    # -----------------------------------------------------

    if score_key not in st.session_state:

        st.session_state[
            score_key
        ] = None


    if submitted_key not in st.session_state:

        st.session_state[
            submitted_key
        ] = False


    if answers_key not in st.session_state:

        st.session_state[
            answers_key
        ] = []


    # =====================================================
    # SHOW QUESTIONS
    # =====================================================

    answers = []


    if not st.session_state[
        submitted_key
    ]:


        for index, question in enumerate(
            questions
        ):


            st.markdown(
                f"### Q{index + 1}. "
                f"{question['question']}"
            )


            selected = st.radio(

                "Select your answer:",

                question["options"],

                key=(
                    f"{session_key}_q_{index}"
                ),

                index=None

            )


            answers.append(
                selected
            )


            st.divider()


        # =================================================
        # SUBMIT TEST
        # =================================================

        if st.button(

            "✅ Submit Test",

            key=(
                f"{session_key}_submit"
            )

        ):


            unanswered = [

                i + 1

                for i, answer in enumerate(
                    answers
                )

                if answer is None

            ]


            if unanswered:


                st.warning(

                    "⚠️ Please answer all questions. "

                    f"Unanswered: {unanswered}"

                )


            else:


                correct = 0


                for index, answer in enumerate(
                    answers
                ):


                    if answer == questions[
                        index
                    ]["answer"]:

                        correct += 1


                score = round(

                    (
                        correct /
                        len(questions)
                    ) * 100

                )


                st.session_state[
                    score_key
                ] = score


                st.session_state[
                    answers_key
                ] = answers


                st.session_state[
                    submitted_key
                ] = True


                st.rerun()


    # =====================================================
    # SHOW TEST RESULT
    # =====================================================

    else:


        score = st.session_state[
            score_key
        ]


        saved_answers = st.session_state[
            answers_key
        ]


        correct_answers = sum(

            1

            for i, answer in enumerate(
                saved_answers
            )

            if answer == questions[
                i
            ]["answer"]

        )


        wrong_answers = (

            len(questions)
            -
            correct_answers

        )


        st.markdown(
            "## 🏆 Test Result"
        )


        r1, r2, r3 = st.columns(3)


        with r1:

            st.metric(

                "📊 Score",

                f"{score}%"

            )


        with r2:

            st.metric(

                "✅ Correct",

                correct_answers

            )


        with r3:

            st.metric(

                "❌ Wrong",

                wrong_answers

            )


        # =================================================
        # PERFORMANCE MESSAGE
        # =================================================

        if score >= 80:

            st.success(
                "🌟 Excellent Performance!"
            )


        elif score >= 60:

            st.info(
                "👍 Good Performance!"
            )


        else:

            st.warning(
                "📚 Keep practicing and improve your skills."
            )


        st.divider()


        # =================================================
        # ANSWER REVIEW
        # =================================================

        st.markdown(
            "## 📝 Answer Review"
        )


        for index, question in enumerate(
            questions
        ):


            user_answer = saved_answers[
                index
            ]


            correct_answer = question[
                "answer"
            ]


            # CORRECT ANSWER

            if user_answer == correct_answer:


                with st.expander(

                    f"✅ Question "
                    f"{index + 1} — Correct"

                ):


                    st.markdown(
                        f"### "
                        f"{question['question']}"
                    )


                    st.success(

                        f"Your Answer: "
                        f"{user_answer}"

                    )


                    st.info(

                        f"Correct Answer: "
                        f"{correct_answer}"

                    )


            # WRONG ANSWER

            else:


                with st.expander(

                    f"❌ Question "
                    f"{index + 1} — Wrong"

                ):


                    st.markdown(
                        f"### "
                        f"{question['question']}"
                    )


                    st.error(

                        f"❌ Your Answer: "
                        f"{user_answer}"

                    )


                    st.success(

                        f"✅ Correct Answer: "
                        f"{correct_answer}"

                    )


        st.divider()


        # =================================================
        # RETAKE BUTTON
        # =================================================

        if st.button(

            "🔄 Retake Test",

            key=(
                f"{session_key}_retake"
            )

        ):


            st.session_state[
                score_key
            ] = None


            st.session_state[
                submitted_key
            ] = False


            st.session_state[
                answers_key
            ] = []


            for index in range(
                len(questions)
            ):


                question_key = (

                    f"{session_key}_q_{index}"

                )


                if question_key in st.session_state:

                    del st.session_state[
                        question_key
                    ]


            st.rerun()


    return st.session_state[
        score_key
    ]


# =========================================================
# PDF FUNCTION
# =========================================================

def create_resume_pdf(

    student_name,
    email,
    phone,
    college_name,
    branch,
    semester,
    cgpa,
    aptitude_score,
    technical_skill_score,
    communication_skill_score,
    coding_score,
    internship_experience,
    projects,
    backlogs,
    certifications,
    prediction,
    placement_probability,
    recommended_companies

):


    buffer = BytesIO()


    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=15 * mm,

        leftMargin=15 * mm,

        topMargin=15 * mm,

        bottomMargin=15 * mm

    )


    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "TitleCustom",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=20,

        leading=24

    )


    heading_style = ParagraphStyle(

        "HeadingCustom",

        parent=styles["Heading2"],

        fontSize=13,

        leading=16,

        spaceAfter=6

    )


    normal_style = styles[
        "Normal"
    ]


    story = []


    # =====================================================
    # TITLE
    # =====================================================

    story.append(

        Paragraph(

            "STUDENT PLACEMENT REPORT",

            title_style

        )

    )


    story.append(
        Spacer(1, 10)
    )


    # =====================================================
    # STUDENT DETAILS
    # =====================================================

    story.append(

        Paragraph(

            f"<b>{student_name}</b>",

            styles["Heading1"]

        )

    )


    story.append(

        Paragraph(

            f"Email: {email}<br/>"
            f"Phone: {phone}<br/>"
            f"College: {college_name}<br/>"
            f"Branch: {branch}<br/>"
            f"Semester: {semester}",

            normal_style

        )

    )


    story.append(
        Spacer(1, 10)
    )


    # =====================================================
    # EDUCATION
    # =====================================================

    story.append(

        Paragraph(

            "Education",

            heading_style

        )

    )


    education_data = [

        ["CGPA", f"{cgpa}/10"],

        ["Semester", str(semester)],

        ["Backlogs", str(backlogs)]

    ]


    table = Table(

        education_data,

        colWidths=[
            70 * mm,
            90 * mm
        ]

    )


    table.setStyle(

        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )

        ])

    )


    story.append(
        table
    )


    story.append(
        Spacer(1, 10)
    )


    # =====================================================
    # SKILLS
    # =====================================================

    story.append(

        Paragraph(

            "Skills & Performance",

            heading_style

        )

    )


    skills_data = [

        ["Skill", "Score"],

        ["Aptitude", f"{aptitude_score}/100"],

        [
            "Technical Skills",
            f"{technical_skill_score}/100"
        ],

        [
            "Communication",
            f"{communication_skill_score}/100"
        ],

        [
            "Coding",
            f"{coding_score}/100"
        ]

    ]


    table = Table(

        skills_data,

        colWidths=[
            100 * mm,
            60 * mm
        ]

    )


    table.setStyle(

        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )

        ])

    )


    story.append(
        table
    )


    story.append(
        Spacer(1, 10)
    )


    # =====================================================
    # EXPERIENCE
    # =====================================================

    story.append(

        Paragraph(

            "Experience",

            heading_style

        )

    )


    experience_data = [

        [
            "Internships",
            str(internship_experience)
        ],

        [
            "Projects",
            str(projects)
        ],

        [
            "Certifications",
            str(certifications)
        ]

    ]


    table = Table(

        experience_data,

        colWidths=[
            100 * mm,
            60 * mm
        ]

    )


    table.setStyle(

        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )

        ])

    )


    story.append(
        table
    )


    story.append(
        Spacer(1, 10)
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    story.append(

        Paragraph(

            "Placement Prediction",

            heading_style

        )

    )


    status = (

        "Placed / Good Chance"

        if prediction == 1

        else

        "Needs Improvement"

    )


    result_data = [

        [
            "Prediction",
            status
        ],

        [
            "Probability",
            f"{placement_probability:.2f}%"
        ]

    ]


    table = Table(

        result_data,

        colWidths=[
            100 * mm,
            60 * mm
        ]

    )


    table.setStyle(

        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )

        ])

    )


    story.append(
        table
    )


    story.append(
        Spacer(1, 10)
    )


    # =====================================================
    # COMPANIES
    # =====================================================

    story.append(

        Paragraph(

            "Recommended Companies",

            heading_style

        )

    )


    if recommended_companies:


        for company in recommended_companies:


            story.append(

                Paragraph(

                    f"• {company}",

                    normal_style

                )

            )


    else:


        story.append(

            Paragraph(

                "No recommendations available.",

                normal_style

            )

        )


    story.append(
        Spacer(1, 15)
    )


    story.append(

        Paragraph(

            "Generated by Placement Prediction System",

            normal_style

        )

    )


    document.build(
        story
    )


    buffer.seek(
        0
    )


    return buffer


# =========================================================
# PREDICTION LOGIC
# =========================================================

if predict_button:


    # VALIDATION

    if student_name.strip() == "":

        st.warning(
            "⚠️ Please enter student name."
        )

        st.stop()


    if email.strip() == "":

        st.warning(
            "⚠️ Please enter email."
        )

        st.stop()


    if phone.strip() == "":

        st.warning(
            "⚠️ Please enter phone number."
        )

        st.stop()


    if college_name.strip() == "":

        st.warning(
            "⚠️ Please enter college name."
        )

        st.stop()


    if branch.strip() == "":

        st.warning(
            "⚠️ Please enter branch."
        )

        st.stop()


    if cgpa <= 0:

        st.warning(
            "⚠️ Please enter a valid CGPA."
        )

        st.stop()


    # INPUT DATA

    input_data = pd.DataFrame({

        "college_name": [
            college_name.strip()
        ],

        "branch": [
            branch.strip()
        ],

        "semester": [
            semester
        ],

        "cgpa": [
            cgpa
        ],

        "aptitude_score": [
            aptitude_score
        ],

        "technical_skill_score": [
            technical_skill_score
        ],

        "communication_skill_score": [
            communication_skill_score
        ],

        "coding_score": [
            coding_score
        ],

        "internship_experience": [
            internship_experience
        ],

        "projects": [
            projects
        ],

        "backlogs": [
            backlogs
        ],

        "certifications": [
            certifications
        ]

    })


    # MODEL PREDICTION

    try:


        prediction = model.predict(
            input_data
        )[0]


        probabilities = model.predict_proba(
            input_data
        )[0]


        classes = list(
            model.classes_
        )


        if 1 in classes:


            placed_index = classes.index(
                1
            )


            placement_probability = (

                probabilities[
                    placed_index
                ]

                * 100

            )


        else:

            placement_probability = 0


    except Exception as error:


        st.error(
            "❌ Model prediction error."
        )


        st.code(
            str(error)
        )


        st.stop()


    # COMPANY RECOMMENDATION

    recommended_companies = suggest_companies(

        cgpa=cgpa,

        aptitude_score=aptitude_score,

        technical_skill_score=technical_skill_score,

        communication_skill_score=communication_skill_score,

        coding_score=coding_score,

        internship_experience=internship_experience,

        projects=projects,

        backlogs=backlogs,

        certifications=certifications

    )


    # SAVE PREDICTION

    st.session_state.prediction_done = True

    st.session_state.prediction = prediction

    st.session_state.placement_probability = (
        placement_probability
    )

    st.session_state.recommended_companies = (
        recommended_companies
    )


# =========================================================
# SHOW RESULT
# =========================================================

if st.session_state.prediction_done:


    prediction = st.session_state.prediction


    placement_probability = (

        st.session_state.placement_probability

    )


    recommended_companies = (

        st.session_state.recommended_companies

    )


    st.divider()


    if prediction == 1:


        st.success(

            "🎉 Good News! You have a good chance of getting placed."

        )


    else:


        st.error(

            "❌ Your profile currently needs improvement."

        )


    # =====================================================
    # TOP RESULT
    # =====================================================

    st.subheader(
        "🎯 Placement Probability"
    )


    result_col1, result_col2, result_col3 = st.columns(
        3
    )


    with result_col1:


        st.metric(

            "Placement Probability",

            f"{placement_probability:.2f}%"

        )


    with result_col2:


        status = (

            "Good Chance"

            if prediction == 1

            else

            "Needs Improvement"

        )


        st.metric(
            "Status",
            status
        )


    with result_col3:


        st.metric(

            "CGPA",

            f"{cgpa:.2f}"

        )


    # =====================================================
    # STUDENT PROFILE
    # =====================================================

    st.divider()


    st.markdown(
        f"""
        <div class="glass-card">

        <h2>👨‍🎓 {student_name}</h2>

        <p>

        📧 <b>Email:</b> {email}<br>

        📱 <b>Phone:</b> {phone}<br>

        🏫 <b>College:</b> {college_name}<br>

        💻 <b>Branch:</b> {branch}<br>

        📚 <b>Semester:</b> {semester}

        </p>

        <hr>

        <h3>🎯 Placement Preparation</h3>

        <p>

        Take skill tests, check your overview,
        monitor your dashboard and improve
        your weak areas.

        </p>

        </div>
        """,

        unsafe_allow_html=True

    )


    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3 = st.tabs([

        "📝 Take Test",

        "📋 Overview",

        "📊 Dashboard"

    ])


    # =====================================================
    # TAKE TEST
    # =====================================================

    with tab1:


        st.markdown(

            """
            <div class="glass-card">

            <h2>📝 Skill Assessment</h2>

            <p>

            Test your Aptitude, Technical,
            Coding and Communication skills.

            </p>

            </div>
            """,

            unsafe_allow_html=True

        )


        test_type = st.selectbox(

            "🎯 Select Test",

            [

                "Aptitude Test",

                "Technical Test",

                "Coding Test",

                "Communication Test"

            ]

        )


        if test_type == "Aptitude Test":


            run_mcq_test(

                "🧠 Aptitude Test",

                APTITUDE_QUESTIONS,

                "aptitude_test"

            )


        elif test_type == "Technical Test":


            run_mcq_test(

                "💻 Technical Test",

                TECHNICAL_QUESTIONS,

                "technical_test"

            )


        elif test_type == "Coding Test":


            run_mcq_test(

                "👨‍💻 Coding Test",

                CODING_QUESTIONS,

                "coding_test"

            )


        elif test_type == "Communication Test":


            run_mcq_test(

                "🗣️ Communication Test",

                COMMUNICATION_QUESTIONS,

                "communication_test"

            )


        st.divider()


        st.subheader(
            "🏆 Your Test Scores"
        )


        t1, t2, t3, t4 = st.columns(4)


        with t1:


            score = st.session_state.get(
                "aptitude_test_score"
            )


            st.metric(

                "🧠 Aptitude",

                f"{score}%"

                if score is not None

                else "Not Attempted"

            )


        with t2:


            score = st.session_state.get(
                "technical_test_score"
            )


            st.metric(

                "💻 Technical",

                f"{score}%"

                if score is not None

                else "Not Attempted"

            )


        with t3:


            score = st.session_state.get(
                "coding_test_score"
            )


            st.metric(

                "👨‍💻 Coding",

                f"{score}%"

                if score is not None

                else "Not Attempted"

            )


        with t4:


            score = st.session_state.get(
                "communication_test_score"
            )


            st.metric(

                "🗣️ Communication",

                f"{score}%"

                if score is not None

                else "Not Attempted"

            )


    # OVERVIEW

    with tab2:


        st.subheader(
            "📋 Student Overview"
        )


        p1, p2 = st.columns(2)


        with p1:


            st.write(
                f"**Name:** {student_name}"
            )


            st.write(
                f"**Email:** {email}"
            )


            st.write(
                f"**Phone:** {phone}"
            )


        with p2:


            st.write(
                f"**College:** {college_name}"
            )


            st.write(
                f"**Branch:** {branch}"
            )


            st.write(
                f"**Semester:** {semester}"
            )


        st.divider()


        a1, a2, a3 = st.columns(3)


        with a1:

            st.metric(
                "CGPA",
                f"{cgpa:.2f}/10"
            )


        with a2:

            st.metric(
                "Backlogs",
                backlogs
            )


        with a3:

            st.metric(
                "Certifications",
                certifications
            )


        st.divider()


        skill_data = pd.DataFrame({

            "Skill": [

                "🧠 Aptitude",

                "💻 Technical",

                "🗣️ Communication",

                "👨‍💻 Coding"

            ],

            "Score": [

                aptitude_score,

                technical_skill_score,

                communication_skill_score,

                coding_score

            ]

        })


        st.dataframe(

            skill_data,

            use_container_width=True,

            hide_index=True

        )


        st.divider()


        e1, e2, e3 = st.columns(3)


        with e1:

            st.metric(
                "🏢 Internships",
                internship_experience
            )


        with e2:

            st.metric(
                "🚀 Projects",
                projects
            )


        with e3:

            st.metric(
                "📜 Certifications",
                certifications
            )


    
    # DASHBOARD
    

    with tab3:


        st.subheader(
            "📊 Performance Dashboard"
        )


        d1, d2, d3, d4 = st.columns(4)


        with d1:

            st.metric(
                "🧠 Aptitude",
                f"{aptitude_score}%"
            )


        with d2:

            st.metric(
                "💻 Technical",
                f"{technical_skill_score}%"
            )


        with d3:

            st.metric(
                "🗣️ Communication",
                f"{communication_skill_score}%"
            )


        with d4:

            st.metric(
                "👨‍💻 Coding",
                f"{coding_score}%"
            )


        st.divider()


        st.markdown(
            "### 📈 Skill Progress"
        )


        st.write(
            f"🧠 Aptitude — {aptitude_score}%"
        )

        st.progress(
            aptitude_score / 100
        )


        st.write(
            f"💻 Technical — {technical_skill_score}%"
        )

        st.progress(
            technical_skill_score / 100
        )


        st.write(
            f"🗣️ Communication — {communication_skill_score}%"
        )

        st.progress(
            communication_skill_score / 100
        )


        st.write(
            f"👨‍💻 Coding — {coding_score}%"
        )

        st.progress(
            coding_score / 100
        )


        st.divider()


        st.markdown(
            "### 🎯 Placement Prediction"
        )


        st.metric(

            "Placement Probability",

            f"{placement_probability:.2f}%"

        )


        if placement_probability >= 75:

            st.success(

                "🟢 Excellent profile! "
                "Strong placement probability."

            )


        elif placement_probability >= 50:

            st.warning(

                "🟡 Good profile, "
                "but some skills can be improved."

            )


        else:

            st.error(

                "🔴 Your profile needs improvement."

            )


        st.divider()


        
        # COMPANY RECOMMENDATION
        

        st.markdown(
            "### 🏢 Recommended Companies"
        )


        if recommended_companies:


            company_cols = st.columns(
                3
            )


            for index, company in enumerate(
                recommended_companies
            ):


                with company_cols[
                    index % 3
                ]:


                    st.success(
                        f"🏢 {company}"
                    )


        else:


            st.info(
                "No company recommendation available."
            )


        st.divider()


        
        # IMPROVEMENT ROADMAP
        

        st.markdown(
            "### 💡 Skill Improvement Roadmap"
        )


        suggestions = []


        if cgpa < 7:

            suggestions.append(
                "📚 Improve your CGPA."
            )


        if aptitude_score < 60:

            suggestions.append(
                "🧠 Practice aptitude and reasoning."
            )


        if technical_skill_score < 60:

            suggestions.append(
                "💻 Improve technical skills."
            )


        if communication_skill_score < 60:

            suggestions.append(
                "🗣️ Improve communication skills."
            )


        if coding_score < 60:

            suggestions.append(
                "👨‍💻 Practice coding and DSA."
            )


        if projects < 3:

            suggestions.append(
                "🚀 Build at least 3 strong projects."
            )


        if internship_experience == 0:

            suggestions.append(
                "🏢 Try to get internship experience."
            )


        if certifications < 2:

            suggestions.append(
                "📜 Complete relevant certifications."
            )


        if backlogs > 0:

            suggestions.append(
                "⚠️ Clear your backlogs."
            )


        if suggestions:


            for suggestion in suggestions:


                st.write(
                    suggestion
                )


        else:


            st.success(
                "🌟 Excellent profile! Keep improving."
            )


    
    # PDF REPORT
    
    st.divider()


    st.subheader(
        "📥 Download Placement Report"
    )


    pdf_file = create_resume_pdf(

        student_name=student_name,

        email=email,

        phone=phone,

        college_name=college_name,

        branch=branch,

        semester=semester,

        cgpa=cgpa,

        aptitude_score=aptitude_score,

        technical_skill_score=technical_skill_score,

        communication_skill_score=communication_skill_score,

        coding_score=coding_score,

        internship_experience=internship_experience,

        projects=projects,

        backlogs=backlogs,

        certifications=certifications,

        prediction=prediction,

        placement_probability=placement_probability,

        recommended_companies=recommended_companies

    )


    safe_name = (

        student_name

        .strip()

        .replace(" ", "_")

        .replace("/", "_")

        .replace("\\", "_")

    )


    st.download_button(

        label="📄 Download Placement Report",

        data=pdf_file,

        file_name=(
            f"{safe_name}_Placement_Report.pdf"
        ),

        mime="application/pdf",

        use_container_width=True

    )



# streamlit run app.py