# import streamlit as st
# import requests
# import json

# # ----------------------------
# # CONFIG
# # ----------------------------
# API_BASE = "http://localhost:8000"   # change if your API is hosted elsewhere

# st.set_page_config(page_title="Adaptive RAG Quiz System", layout="wide")


# # ========================================================
# # 1. HEADER
# # ========================================================
# st.title("📘 Adaptive RAG Quiz System - Streamlit UI")


# # ========================================================
# # 2. HEALTH CHECK
# # ========================================================
# st.subheader("🔍 Server Health Check")
# if st.button("Check Server"):
#     try:
#         r = requests.get(f"{API_BASE}/health")
#         st.success(r.json())
#     except Exception as e:
#         st.error(f"Server Error: {e}")


# # ========================================================
# # 3. UPLOAD PDF
# # ========================================================
# st.subheader("📤 Upload Book PDF")

# pdf_file = st.file_uploader("Choose PDF", type=["pdf"])

# if pdf_file is not None:
#     if st.button("Upload PDF"):
#         files = {"file": pdf_file}
#         res = requests.post(f"{API_BASE}/upload-pdf", files=files)

#         if res.status_code == 200:
#             data = res.json()
#             st.success("PDF Uploaded Successfully!")
#             st.json(data)
#             st.session_state["pdf_path"] = data["server_pdf_path"]
#         else:
#             st.error(res.json())


# # ========================================================
# # 4. PROCESS BOOK → Create Vector DB
# # ========================================================
# st.subheader("📚 Process Book (Generate Vector DB)")

# col1, col2 = st.columns(2)

# with col1:
#     pdf_path = st.text_input("Saved PDF Path",
#                              value=st.session_state.get("pdf_path", ""))

# with col2:
#     collection_name = st.text_input("Collection Name (Unique)")

# if st.button("Process Book"):
#     payload = {
#         "pdf_path": pdf_path,
#         "collection_name": collection_name
#     }

#     res = requests.post(f"{API_BASE}/process-book", json=payload)
#     if res.status_code == 200:
#         st.success("Book processed successfully!")
#         st.json(res.json())
#     else:
#         st.error(res.json())


# # ========================================================
# # 5. LIST COLLECTIONS
# # ========================================================
# st.subheader("📂 Available Textbook Collections")

# if st.button("List Collections"):
#     res = requests.get(f"{API_BASE}/list-collections")
#     if res.status_code == 200:
#         st.json(res.json())
#     else:
#         st.error(res.json())


# # ========================================================
# # 6. EMAIL → STUDENT ID
# # ========================================================
# st.subheader("👤 Convert Email to Student ID")

# email = st.text_input("Enter Email")

# if st.button("Convert Email"):
#     res = requests.post(f"{API_BASE}/email-to-student-id", json={"email": email})
#     if res.status_code == 200:
#         res_data = res.json()
#         st.success("Converted Successfully!")
#         st.write("Generated Student ID:", res_data["student_id"])
#         st.session_state["student_id"] = res_data["student_id"]
#     else:
#         st.error(res.json())


# # ========================================================
# # 7. GENERATE QUIZ
# # ========================================================
# st.subheader("📝 Generate Quiz")

# col1, col2, col3 = st.columns(3)

# with col1:
#     gen_collection = st.text_input("Collection Name")

# with col2:
#     topic = st.text_input("Topic", placeholder="e.g. Periodic Table")

# with col3:
#     num_questions = st.number_input("Num Questions", min_value=1, max_value=20, value=5)

# student_id = st.text_input("Student ID (optional)", 
#                            value=st.session_state.get("student_id", ""))

# with st.expander("📊 Provide Student Performance (Optional - For Adaptive Quiz)"):
#     time_per_question = st.text_input("Time per question (comma-separated)", "10,12,8")
#     hints_used = st.number_input("Hints Used", min_value=0, value=0)
#     correct_answers = st.number_input("Correct Answers", min_value=0, value=0)
#     total_questions_attempted = st.number_input("Total Questions Attempted", min_value=1, value=5)
#     current_difficulty = st.selectbox("Current Difficulty", ["easy", "medium", "hard"])

#     use_performance = st.checkbox("Use student performance for adaptive quiz")


# if st.button("Generate Quiz"):
#     payload = {
#         "collection_name": gen_collection,
#         "topic": topic,
#         "num_questions": num_questions
#     }

#     if student_id:
#         payload["student_id"] = student_id

#     if use_performance:
#         payload["student_performance"] = {
#             "time_per_question": [float(x) for x in time_per_question.split(",")],
#             "hints_used": hints_used,
#             "correct_answers": correct_answers,
#             "total_questions": total_questions_attempted,
#             "current_difficulty": current_difficulty
#         }

#     res = requests.post(f"{API_BASE}/generate-quiz", json=payload)

#     if res.status_code == 200:
#         data = res.json()
#         st.success("Quiz Generated!")

#         st.write("### Topic Structure")
#         st.json(data["topic_structure"])

#         st.write("### Quiz Questions")
#         for q in data["quiz"]:
#             st.write("---")
#             st.write(f"**Q:** {q['question']}")
#             st.write("Options:", q["options"])
#             st.write(f"Correct: `{q['correct_answer']}`")
#     else:
#         st.error(res.json())


# # ========================================================
# # 8. GET STUDENT METRICS
# # ========================================================
# st.subheader("📈 View Student Performance Metrics")

# student_query = st.text_input("Enter Student ID to Check Metrics")

# if st.button("Get Metrics"):
#     res = requests.get(f"{API_BASE}/get-student-metrics/{student_query}")
#     if res.status_code == 200:
#         st.json(res.json())
#     else:
#         st.error(res.json())


import streamlit as st
import requests
import json

# ----------------------------
# CONFIG
# ----------------------------
API_BASE = "http://localhost:8000"   # change if your API is hosted elsewhere

st.set_page_config(page_title="Adaptive RAG Quiz System", layout="wide")


# ========================================================
# 1. HEADER
# ========================================================
st.title("📘 Adaptive RAG Quiz System - Streamlit UI")


# ========================================================
# 2. HEALTH CHECK
# ========================================================
st.subheader("🔍 Server Health Check")
if st.button("Check Server"):
    try:
        r = requests.get(f"{API_BASE}/health")
        st.success(r.json())
    except Exception as e:
        st.error(f"Server Error: {e}")


# ========================================================
# 3. UPLOAD PDF
# ========================================================
st.subheader("📤 Upload Book PDF")

pdf_file = st.file_uploader("Choose PDF", type=["pdf"])

if pdf_file is not None:
    if st.button("Upload PDF"):
        files = {"file": pdf_file}
        res = requests.post(f"{API_BASE}/upload-pdf", files=files)

        if res.status_code == 200:
            data = res.json()
            st.success("PDF Uploaded Successfully!")
            st.json(data)
            st.session_state["pdf_path"] = data["server_pdf_path"]
        else:
            st.error(res.json())


# ========================================================
# 4. PROCESS BOOK → Create Vector DB
# ========================================================
st.subheader("📚 Process Book (Generate Vector DB)")

col1, col2 = st.columns(2)

with col1:
    pdf_path = st.text_input("Saved PDF Path",
                             value=st.session_state.get("pdf_path", ""))

with col2:
    collection_name = st.text_input("Collection Name (Unique)")

if st.button("Process Book"):
    payload = {
        "pdf_path": pdf_path,
        "collection_name": collection_name
    }

    res = requests.post(f"{API_BASE}/process-book", json=payload)
    if res.status_code == 200:
        st.success("Book processed successfully!")
        st.json(res.json())
    else:
        st.error(res.json())


# ========================================================
# 5. LIST COLLECTIONS
# ========================================================
st.subheader("📂 Available Textbook Collections")

if st.button("List Collections"):
    res = requests.get(f"{API_BASE}/list-collections")
    if res.status_code == 200:
        st.json(res.json())
    else:
        st.error(res.json())


# ========================================================
# 6. EMAIL → STUDENT ID
# ========================================================
st.subheader("👤 Convert Email to Student ID")

email = st.text_input("Enter Email")

if st.button("Convert Email"):
    res = requests.post(f"{API_BASE}/email-to-student-id", json={"email": email})
    if res.status_code == 200:
        res_data = res.json()
        st.success("Converted Successfully!")
        st.write("Generated Student ID:", res_data["student_id"])
        st.session_state["student_id"] = res_data["student_id"]
    else:
        st.error(res.json())


# ========================================================
# 7. GENERATE QUIZ
# ========================================================
st.subheader("📝 Generate Quiz")

col1, col2, col3 = st.columns(3)

with col1:
    gen_collection = st.text_input("Collection Name")

with col2:
    topic = st.text_input("Topic", placeholder="e.g. Periodic Table")

with col3:
    num_questions = st.number_input("Num Questions", min_value=1, max_value=20, value=5)

student_id = st.text_input("Student ID (optional)", 
                           value=st.session_state.get("student_id", ""))

with st.expander("📊 Provide Student Performance (Optional - For Adaptive Quiz)"):
    time_per_question = st.text_input("Time per question (comma-separated)", "10,12,8")
    hints_used = st.number_input("Hints Used", min_value=0, value=0)
    correct_answers = st.number_input("Correct Answers", min_value=0, value=0)
    total_questions_attempted = st.number_input("Total Questions Attempted", min_value=1, value=5)
    current_difficulty = st.selectbox("Current Difficulty", ["easy", "medium", "hard"])

    use_performance = st.checkbox("Use student performance for adaptive quiz")


if st.button("Generate Quiz"):
    payload = {
        "collection_name": gen_collection,
        "topic": topic,
        "num_questions": num_questions
    }

    if student_id:
        payload["student_id"] = student_id

    if use_performance:
        payload["student_performance"] = {
            "time_per_question": [float(x) for x in time_per_question.split(",")],
            "hints_used": hints_used,
            "correct_answers": correct_answers,
            "total_questions": total_questions_attempted,
            "current_difficulty": current_difficulty
        }

    res = requests.post(f"{API_BASE}/generate-quiz", json=payload)

    if res.status_code == 200:
        data = res.json()
        st.success("Quiz Generated!")

        st.write("### Topic Structure")
        st.json(data["topic_structure"])

        st.write("### Quiz Questions")

        for q in data["quiz"]:
            st.write("---")
            st.write(f"**Q:** {q['question']}")
            st.write("Options:", q["options"])

            # ❌ REMOVED — No correct answers shown
            # st.write(f"Correct: `{q['correct_answer']}`")   # REMOVED

    else:
        st.error(res.json())


# ========================================================
# 8. GET STUDENT METRICS
# ========================================================
st.subheader("📈 View Student Performance Metrics")

student_query = st.text_input("Enter Student ID to Check Metrics")

if st.button("Get Metrics"):
    res = requests.get(f"{API_BASE}/get-student-metrics/{student_query}")
    if res.status_code == 200:
        st.json(res.json())
    else:
        st.error(res.json())
