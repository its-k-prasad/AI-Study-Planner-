# 📚 AI-Powered Study Planner
An interactive AI-based web application that helps students generate personalized study plans, summaries, and question-based answers from uploaded PDF or text documents using Google Gemini, LangChain, and Hugging Face Embeddings.

<br>

🚀 Features
📄 PDF/TXT Upload: Upload your own study material.

🧠 Google Gemini LLM: Generate intelligent, detailed answers and study schedules.

📚 Smart Study Plans: Day-by-day or week-by-week learning roadmaps with goals and tips.

🔍 Semantic Search: Extract contextually relevant information from documents.

✍️ Comprehensive Q&A: Ask any question based on document content.

🧾 Document Insights: Analyze word count, chapters, sections, and estimated study effort.

💾 Downloadable Output: Save study plans or responses for offline use.

🛠️ Tech Stack
Area	Technologies
Frontend/UI	Streamlit
LLM	Google Gemini API (via LangChain)
Embeddings	HuggingFace Transformers (sentence-transformers/all-MiniLM-L6-v2)
Vector Store	FAISS
Text Processing	PyPDF2, LangChain TextSplitter
Prompt Engineering	LangChain PromptTemplate
Environment	Python, dotenv

🧑‍💻 How to Use
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/ai-study-planner.git
cd ai-study-planner
2. Install Dependencies
Make sure you have Python 3.8+ and pip installed.

bash
Copy
Edit
pip install -r requirements.txt
3. Set Up Google Gemini API Key
Create a .env file in the root directory and add your API key:

bash
Copy
Edit
GOOGLE_API_KEY=your_google_gemini_api_key
🔑 You can get your API key from Google AI Studio

4. Run the App
bash
Copy
Edit
streamlit run app.py
📂 File Structure
bash
Copy
Edit
📁 ai-study-planner/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (add manually)
└── README.md               # Project documentation
📸 Screenshots
<details> <summary>🖼️ Click to view screenshots</summary>



</details>
💡 Sample Prompts
"Create a 10-day study plan for this data science book"

"Summarize all chapters with key takeaways"

"How much time will it take to complete this material?"

"Explain the concepts in chapter 3"

"What are the most important topics in this document?"

📌 Tips
Speak clearly and limit PDFs to under ~100 pages for best results.

Tailor your prompt: the more specific your goal, the better the AI's plan.

Use TXT files for plain content or lecture notes.
