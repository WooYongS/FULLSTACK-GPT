import streamlit as st
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import WikipediaRetriever
from PyPDF2 import PdfReader
import docx

st.set_page_config(page_title="QuizGPT", page_icon="📝")

# Sidebar: OpenAI API Key 입력 & 옵션 선택
with st.sidebar:
    st.title("Settings")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    choice = st.selectbox("Choose what you want to use.", ["File", "Wikipedia Article"])
    difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"])
    st.markdown("[GitHub Repository](https://github.com/your-repo-link)")

# Ensure API key is provided
if not openai_api_key:
    st.error("Please enter your OpenAI API Key in the sidebar.")
    st.stop()

# Initialize LLM
llm = ChatOpenAI(temperature=0.5, openai_api_key=openai_api_key)


# Function to extract text from uploaded file
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = "\n".join(
            [page.extract_text() for page in reader.pages if page.extract_text()]
        )
    elif uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = ""
    return text[:2000]  # 최대 2000자 미리보기


# Function to fetch Wikipedia content (using WikipediaRetriever)
def fetch_wikipedia_content(topic):
    retriever = WikipediaRetriever(top_k_results=1)  # 가장 관련 있는 문서 1개만 가져옴
    docs = retriever.get_relevant_documents(topic)

    if docs:
        return docs[0].page_content  # 첫 번째 문서 내용 반환
    else:
        return "No relevant Wikipedia page found."


# Function to generate quiz questions dynamically from OpenAI
def generate_quiz(content, difficulty):
    prompt = ChatPromptTemplate.from_template(
        """
        You are a quiz generator AI. Based on the provided content, generate a {difficulty} 5-question multiple-choice quiz.
        Each question should have exactly 4 answer options.
        The answer must be based on the given content.

        Return the quiz in JSON format as a list where each item contains:
        - 'question': The quiz question
        - 'options': A list of 4 options
        - 'answer': The correct answer

        Content:
        {content}
    """
    )

    formatted_prompt = prompt.format(content=content, difficulty=difficulty)

    response = llm.invoke(formatted_prompt)

    try:
        quiz_data = json.loads(response.content)

        # 데이터 형식 검증
        if not isinstance(quiz_data, list) or not all(
            "question" in q and "options" in q and "answer" in q for q in quiz_data
        ):
            raise ValueError("Invalid JSON format received.")

        return quiz_data
    except (json.JSONDecodeError, ValueError):
        st.error("⚠ AI가 퀴즈 데이터를 올바르게 생성하지 못했습니다. 다시 시도하세요.")
        return None


# Start quiz
st.title("Quiz GPT 📝")
st.write("Test your knowledge! Choose the correct answers below.")

# 파일 업로드 또는 Wikipedia 검색 선택
quiz_content = None
preview_text = None

if choice == "File":
    uploaded_file = st.file_uploader(
        "Upload a .docx, .txt or .pdf file", type=["pdf", "txt", "docx"]
    )
    if uploaded_file:
        quiz_content = extract_text_from_file(uploaded_file)
        preview_text = quiz_content

elif choice == "Wikipedia Article":
    topic = st.text_input("Search Wikipedia...")
    if topic:
        quiz_content = fetch_wikipedia_content(topic)
        preview_text = quiz_content

# 좌측 미리보기 화면 추가
if preview_text:
    with st.expander("🔍 Preview Content"):
        st.write(preview_text[:2000])

# 퀴즈 생성
if quiz_content:
    if "quiz_data" not in st.session_state or st.button("Generate New Quiz"):
        st.session_state.quiz_data = generate_quiz(quiz_content, difficulty)
        st.session_state.user_answers = {}

# 퀴즈 데이터가 없으면 중단
if not st.session_state.get("quiz_data"):
    st.stop()

# Display quiz questions
correct_answers = 0
for idx, question in enumerate(st.session_state.quiz_data):
    user_choice = st.radio(
        f"**{idx + 1}. {question['question']}**", question["options"], key=f"q{idx}"
    )
    st.session_state.user_answers[idx] = user_choice

# Submit button
if st.button("Submit Quiz"):
    for idx, question in enumerate(st.session_state.quiz_data):
        if st.session_state.user_answers[idx] == question["answer"]:
            correct_answers += 1

    total_questions = len(st.session_state.quiz_data)

    st.write(f"### You got {correct_answers}/{total_questions} correct!")

    if correct_answers == total_questions:
        st.balloons()
        st.success("🎉 Perfect Score! Well done!")
    else:
        st.warning("You can retake the quiz to improve your score.")
