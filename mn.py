import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
import tempfile
import threading
import asyncio
import PyPDF2
from io import BytesIO
import re


def run_async_in_thread(func, *args, **kwargs):
    result = [None]
    exception = [None]

    def target():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result[0] = loop.run_until_complete(func(*args, **kwargs))
        except Exception as e:
            exception[0] = e
        finally:
            loop.close()

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

    if exception[0]:
        raise exception[0]
    return result[0]

@st.cache_resource
def get_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return embeddings, "huggingface"
    except Exception as e:
        st.error(f"Error loading HuggingFace embeddings: {e}")
        return None, None

def extract_document_info(text_content):
    """Extract basic information about the document"""
    lines = text_content.split('\n')
    total_lines = len([line for line in lines if line.strip()])
    
    # Try to estimate chapters/sections
    chapter_patterns = [
        r'chapter\s+\d+',
        r'section\s+\d+',
        r'unit\s+\d+',
        r'part\s+\d+',
        r'lesson\s+\d+'
    ]
    
    chapters = []
    for line in lines:
        for pattern in chapter_patterns:
            if re.search(pattern, line.lower()):
                chapters.append(line.strip())
                break
    
    word_count = len(text_content.split())
    
    return {
        'total_lines': total_lines,
        'chapters': chapters[:20], 
        'estimated_chapters': len(chapters),
        'word_count': word_count,
        'estimated_pages': word_count
    }

def load_documents(uploaded_file=None):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'pdf':
            try:
                pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
                text_content = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                if not text_content.strip():
                    st.error("Could not extract text from PDF.")
                    return None, None
                
                # Extract document information
                doc_info = extract_document_info(text_content)
                docs = [Document(page_content=text_content, metadata={
                    "source": uploaded_file.name,
                    "total_pages": len(pdf_reader.pages),
                    **doc_info
                })]
                return docs, doc_info
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                return None, None
        elif file_extension == 'txt':
            try:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
                    content = uploaded_file.getvalue().decode('utf-8')
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                loader = TextLoader(tmp_file_path, encoding='utf-8')
                docs = loader.load()
                os.unlink(tmp_file_path)
                
                # Extract document information
                doc_info = extract_document_info(content)
                for doc in docs:
                    doc.metadata.update(doc_info)
                return docs, doc_info
            except:
                st.error("Could not read the text file.")
                return None, None
        else:
            st.error("Unsupported file type.")
            return None, None
    else:
        default_content = """
        Study Tips for Effective Learning:
        
        Chapter 1: Time Management
        - Create a study schedule
        - Break tasks into smaller chunks
        - Use the Pomodoro technique
        
        Chapter 2: Active Learning
        - Take notes while reading
        - Summarize key concepts
        - Practice retrieval
        
        Chapter 3: Memory Techniques
        - Use mnemonics
        - Create visual associations
        - Practice spaced repetition
        """
        doc_info = extract_document_info(default_content)
        docs = [Document(page_content=default_content, metadata={"source": "default", **doc_info})]
        return docs, doc_info

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_documents(docs), doc_info

def build_vector_store(documents):
    try:
        embeddings, embedding_type = get_embeddings()
        if embeddings is None:
            st.error("Failed to load embeddings")
            return None
        st.info(f"Using {embedding_type} embeddings...")
        return FAISS.from_documents(documents, embeddings)
    except Exception as e:
        st.error(f"Error building vector store: {e}")
        return None

def create_enhanced_prompt():
    """Create an enhanced prompt template for better study planning"""
    template = """
You are an expert AI study planner and tutor. Your task is to provide comprehensive, structured, and practical study plans based on the uploaded document.

Document Context: {context}

User Question: {question}

IMPORTANT INSTRUCTIONS:
1. If the user asks for a study plan, timetable, or completion schedule, always provide:
   - A detailed day-by-day or week-by-week breakdown
   - Realistic time estimates for each section/chapter
   - Study goals and milestones
   - Review and practice sessions
   - Tips for effective learning

2. If the user asks any question about the document content:
   - Always provide a complete and helpful answer
   - Use specific information from the document
   - Include relevant examples or explanations
   - If information is not in the document, clearly state that

3. For study plans, consider:
   - The document length and complexity
   - Reasonable daily study hours (2-4 hours typically)
   - Include breaks and review sessions
   - Progressive difficulty levels
   - Practice and assessment opportunities

4. Always structure your response clearly with:
   - Clear headings and sections
   - Bullet points for easy reading
   - Actionable steps
   - Time estimates

5. Be encouraging and motivational while being realistic about time commitments.

Provide a comprehensive, well-structured response:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def create_qa_chain(vector_store, api_key, doc_info):
    if vector_store is None:
        return None
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            max_output_tokens=2048, 
            google_api_key=api_key
        )
        
      
        enhanced_prompt = create_enhanced_prompt()
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),  # Retrieve more context
            chain_type_kwargs={"prompt": enhanced_prompt},
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

def detect_question_type(question):
    """Detect if the user is asking for a study plan or other type of question"""
    study_plan_keywords = [
        'study plan', 'timetable', 'schedule', 'complete', 'finish', 'learn',
        'roadmap', 'how many days', 'study guide', 'plan to study'
    ]
    
    question_lower = question.lower()
    is_study_plan = any(keyword in question_lower for keyword in study_plan_keywords)
    
    return 'study_plan' if is_study_plan else 'general_question'

def enhance_study_plan_question(question, doc_info):
    """Enhance the user's question with document context for better study planning"""
    if doc_info:
        enhanced_question = f"""
        {question}
        
        Additional Context about the document:
        - Document has approximately {doc_info.get('estimated_pages', 'unknown')} pages
        - Word count: approximately {doc_info.get('word_count', 'unknown')} words
        - Estimated chapters/sections: {doc_info.get('estimated_chapters', 'unknown')}
        
        Please create a comprehensive study plan considering this document size and structure.
        """
        return enhanced_question
    return question

# Streamlit UI
st.set_page_config(page_title="AI Study Planner", layout="wide")
st.title("📚 AI-Powered Study Planner")

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your Google Gemini API Key:", type="password")
uploaded_file = st.sidebar.file_uploader("Upload your study material (PDF/TXT):", type=['txt', 'pdf'])

if uploaded_file:
    file_size = len(uploaded_file.getvalue()) / 1024
    st.sidebar.info(f"📄 {uploaded_file.name} ({file_size:.1f} KB)")

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
    st.session_state.vector_store = None
    st.session_state.documents_loaded = False
    st.session_state.doc_info = None

# Document loading
if st.sidebar.button("Load Documents") or not st.session_state.documents_loaded:
    if not api_key:
        st.error("Please enter your Google Gemini API key.")
    else:
        with st.spinner("Processing documents..."):
            result = load_documents(uploaded_file)
            if result[0] is None:
                st.stop()
            
            docs, doc_info = result
            st.info(f"📚 Loaded {len(docs)} chunks")
            
            # Display document information
            if doc_info and uploaded_file:
                st.sidebar.markdown("### 📊 Document Info")
                st.sidebar.write(f"📄 Pages: {doc_info.get('estimated_pages', 'N/A')}")
                st.sidebar.write(f"📝 Words: {doc_info.get('word_count', 'N/A'):,}")
                st.sidebar.write(f"📚 Chapters: {doc_info.get('estimated_chapters', 'N/A')}")
            
            vector_store = build_vector_store(docs)
            if vector_store:
                qa_chain = create_qa_chain(vector_store, api_key, doc_info)
                if qa_chain:
                    st.session_state.vector_store = vector_store
                    st.session_state.qa_chain = qa_chain
                    st.session_state.documents_loaded = True
                    st.session_state.doc_info = doc_info
                    st.success("✅ Ready to answer your questions!")

# Main interface
if st.session_state.qa_chain and st.session_state.documents_loaded:
    st.success("🎯 Ask your question here and get the answer using Prasad AI")
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📅 Create Study Plan"):
            st.session_state.quick_question = "Create a comprehensive study plan to complete this document. Include a detailed timetable with daily goals and milestones."
    with col2:
        if st.button("📋 Summarize Content"):
            st.session_state.quick_question = "Provide a comprehensive summary of this document including main topics, key concepts, and learning objectives."
    with col3:
        if st.button("🎯 Learning Roadmap"):
            st.session_state.quick_question = "Create a learning roadmap for mastering the content in this document, including prerequisites and advanced topics."
    
    # Text input area
    default_question = getattr(st.session_state, 'quick_question', '')
    user_input = st.text_area(
        "Enter your study topic, question, or ask for a plan:",
        value=default_question,
        placeholder="Examples:\n• 'Create a 7-day study plan for this Python book'\n• 'How to complete this PDF in 2 weeks with daily schedule?'\n• 'Explain the main concepts in chapter 3'\n• 'What are the key topics I should focus on?'",
        height=150
    )
    
    if 'quick_question' in st.session_state:
        del st.session_state.quick_question
    
    if st.button("🚀 Get Answer", type="primary"):
        if user_input.strip():
            with st.spinner("Generating comprehensive response..."):
                try:
                    # Detect question type and enhance if needed
                    question_type = detect_question_type(user_input)
                    
                    if question_type == 'study_plan':
                        enhanced_question = enhance_study_plan_question(user_input, st.session_state.doc_info)
                    else:
                        enhanced_question = user_input
                    
                    # Get response
                    result = st.session_state.qa_chain({"query": enhanced_question})
                    
                    # Display response
                    st.markdown("### 📝 Response")
                    st.markdown(result['result'])
                    
                    # Show source information if available
                    if 'source_documents' in result and result['source_documents']:
                        with st.expander("📚 Source Information"):
                            for i, doc in enumerate(result['source_documents'][:3]):
                                st.markdown(f"**Source {i+1}:**")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    
                    # Download option
                    st.download_button(
                        "📥 Download Response", 
                        data=result['result'], 
                        file_name="study_response.txt", 
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.info("Please try rephrasing your question or check your API key.")
        else:
            st.warning("Please enter a valid question or prompt.")
else:
    st.info("🧩 Please upload a document and load it first to get started.")

# Help section
with st.expander("ℹ️ How to Use This AI Study Planner"):
    st.markdown("""
    ### 🚀 Getting Started
    1. **Get API Key**: Get a Gemini API Key from [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. **Enter API Key**: Paste your API key in the sidebar
    3. **Upload Document**: Upload a PDF or TXT file with your study material
    4. **Load Documents**: Click "Load Documents" to process your file
    5. **Ask Questions**: Use the interface to ask questions or request study plans
    
    ### 📚 What You Can Ask
    **Study Planning:**
    - "Create a 7-day study plan for this Python book"
    - "How to complete this PDF in 2 weeks with daily schedule?"
    - "Give me a comprehensive learning roadmap"
    - "What's the best way to study this material?"
    
    **Content Questions:**
    - "Summarize the main topics in this document"
    - "Explain the concept discussed in chapter 3"
    - "What are the key points I should focus on?"
    - "Give me practice questions for this topic"
    
    ### ✨ Features
    - **Smart Study Plans**: Get detailed timetables with daily goals
    - **Comprehensive Answers**: Every question gets a complete response
    - **Document Analysis**: Automatic analysis of document structure
    - **Source References**: See which parts of your document were used
    - **Download Results**: Save your study plans and answers
    """)

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: For best results, be specific about your study goals and timeline when asking for study plans!")