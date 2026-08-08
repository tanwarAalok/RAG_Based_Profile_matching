import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables!")

class ResumeMetadata(BaseModel):
    name: str = Field(description="Full name of the candidate")
    skills: list[str] = Field(description="List of technical and soft skills")
    experience_years: int = Field(description="Total years of professional experience")
    education: str = Field(description="Highest degree and institution")

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-3-haiku",
    temperature=0
)
embeddings = OpenAIEmbeddings(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model="openai/text-embedding-3-small"
)
vector_store = Chroma(
    collection_name="resumes",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


def load_pdf(file_path: str) -> list[Document]:
    reader = PdfReader(file_path)
    docs = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": file_path, "page": page_num}
            ))

    return docs


def extract_metadata(text: str) -> dict:
    parser = PydanticOutputParser(pydantic_object=ResumeMetadata)

    prompt = PromptTemplate(
        template="""Extract candidate information from the following resume text. 
        You must output ONLY valid JSON. Do not include any conversational text, greetings, or markdown blocks.

        {format_instructions}

        Resume Text:
        {text}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 3. Create the extraction chain
    chain = prompt | llm | parser

    try:
        # Run the chain
        result = chain.invoke({"text": text[:3000]})
        return result.model_dump()  # Note: .dict() is deprecated in Pydantic V2, use .model_dump()
    except Exception as e:
        print(f"Extraction failed for chunk: {e}")
        return {"name": "Unknown", "skills": [], "experience_years": 0, "education": "Unknown"}


def process_resumes(directory_path: str):
    """Loads, chunks, and stores resumes with metadata."""
    # Intelligent chunking: respects paragraphs and sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "•", " ", ""]
    )

    all_files = os.listdir(directory_path)
    pdf_files = [f for f in all_files if f.endswith(".pdf")]

    for filename in pdf_files[20:]:
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing: {filename}")

            # Load
            docs = load_pdf(file_path)
            full_text = " ".join([d.page_content for d in docs])

            # Extract Metadata
            metadata = extract_metadata(full_text)
            metadata["source"] = file_path

            # Chunk
            chunks = text_splitter.split_documents(docs)

            # Embed & Store (attaching metadata to each chunk)
            for chunk in chunks:
                chunk.metadata = {
                    "name": metadata["name"],
                    "skills": ", ".join(metadata["skills"]),
                    "experience_years": metadata["experience_years"],
                    "education": metadata["education"],
                    "source": file_path
                }

            vector_store.add_documents(chunks)
            print(f"Added {len(chunks)} chunks for {metadata['name']}")


if __name__ == "__main__":
    process_resumes("./resumes")