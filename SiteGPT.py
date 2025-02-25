import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import SitemapLoader
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
import os

# ✅ Cloudflare 문서 사이트맵 URL
SITEMAP_URL = "https://developers.cloudflare.com/sitemap-0.xml"

# ✅ Chatbot UI를 위한 Streamlit 설정
st.set_page_config(page_title="SiteGPT - Cloudflare Docs", page_icon="☁️")

st.title("☁️ SiteGPT - Cloudflare Docs")
st.write("Ask me anything about Cloudflare's AI Gateway, Vectorize, or Workers AI.")

# ✅ 사용자 API Key 입력받기
with st.sidebar:
    st.subheader("Settings")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    st.markdown("[GitHub Repository](https://github.com/your-repo-link)")

if not openai_api_key:
    st.error("🚨 Please enter your OpenAI API Key to use SiteGPT.")
    st.stop()


# ✅ Sitemap에서 Cloudflare 문서 URL 추출
@st.cache_data(show_spinner="🔄 Loading Cloudflare Documentation...")
def extract_urls_from_sitemap(sitemap_url):
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "xml")
        urls = [loc.text for loc in soup.find_all("loc")]
        # AI Gateway, Vectorize, Workers AI 문서만 필터링
        filtered_urls = [
            url
            for url in urls
            if any(
                product in url for product in ["ai-gateway", "vectorize", "workers-ai"]
            )
        ]
        return filtered_urls
    except Exception as e:
        st.error(f"🚨 Failed to retrieve sitemap: {e}")
        return []


# ✅ 사이트맵에서 가져온 문서를 벡터 스토어에 저장
@st.cache_resource(show_spinner="🔄 Indexing Cloudflare Documentation...")
def load_and_index_docs(urls):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in ["script", "style", "header", "footer", "nav"]:
                for element in soup.find_all(tag):
                    element.decompose()
            text = " ".join(soup.stripped_strings)
            chunks = text_splitter.split_text(text)
            docs.extend(
                [
                    Document(page_content=chunk, metadata={"source": url})
                    for chunk in chunks
                ]
            )
        except Exception as e:
            st.warning(f"⚠ Failed to fetch page: {url} - {e}")

    # ✅ OpenAI Embeddings을 사용하여 FAISS 벡터 스토어 생성
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# ✅ Load Sitemap and Index Documents
doc_urls = extract_urls_from_sitemap(SITEMAP_URL)
if not doc_urls:
    st.error("🚨 No relevant Cloudflare documentation pages found in sitemap.")
    st.stop()

retriever = load_and_index_docs(doc_urls).as_retriever()

# ✅ 챗봇을 위한 LangChain RetrievalQA 설정
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.5, openai_api_key=openai_api_key)

prompt_template = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that answers questions based on Cloudflare's documentation.
    Use the following context to provide accurate answers. If the information is not available, say "I don't know."
    
    Context: {context}
    
    Question: {question}
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

# ✅ Streamlit Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about Cloudflare documentation...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("🔎 Searching documentation..."):
        response = qa_chain({"query": user_input})
        answer = response["result"]
        sources = response.get("source_documents", [])

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            st.markdown("**Sources:**")
            for source in sources[:3]:  # 최대 3개의 출처 표시
                st.markdown(
                    f"- [{source.metadata['source']}]({source.metadata['source']})"
                )

    st.session_state.messages.append({"role": "assistant", "content": answer})
