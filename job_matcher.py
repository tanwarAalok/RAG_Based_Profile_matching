import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables!")


class MatchResult(BaseModel):
    candidate_name: str = Field(description="Full name of candidate")
    resume_path: str = Field(description="File path of the candidate's resume")
    match_score: int = Field(description="Score from 0 to 100 based on fit")
    matched_skills: list[str] = Field(description="List of skills matching the JD")
    relevant_excerpts: list[str] = Field(description="Relevant text excerpts from resume")
    reasoning: str = Field(description="Explanation of why candidate matched or did not match")


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

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-3-haiku",
    temperature=0
)

parser = PydanticOutputParser(pydantic_object=MatchResult)

prompt_template = PromptTemplate(
    template="""You are an expert ATS recruitment evaluator. Compare the candidate's resume excerpts against the job description.

Job Description:
{job_description}

Candidate Name: {candidate_name}
Candidate Skills: {skills}

Resume Excerpts:
{excerpts}

{format_instructions}

Return ONLY valid JSON matching the schema above. Do not include markdown code block tags or extra conversational text.
""",
    input_variables=["job_description", "candidate_name", "skills", "excerpts"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt_template | llm | parser


def match_job(job_description: str, top_k: int = 10, min_experience: int = 0):
    """Retrieves matches and generates JSON output."""

    search_kwargs = {"k": top_k * 2}  # Retrieve more chunks to account for deduplication
    if min_experience > 0:
        search_kwargs["filter"] = {"experience_years": {"$gte": min_experience}}

    # Perform Semantic Search
    results = vector_store.similarity_search_with_score(job_description, **search_kwargs)

    # Deduplicate candidates across chunks
    unique_candidates = {}
    for doc, distance in results:
        name = doc.metadata.get("name", "Unknown")
        if name not in unique_candidates:
            unique_candidates[name] = {
                "metadata": doc.metadata,
                "excerpts": [doc.page_content],
                "base_score": max(0, int((1.0 - distance) * 100))
            }
        else:
            unique_candidates[name]["excerpts"].append(doc.page_content)

    final_matches = []

    # Evaluate candidates with LLM
    for name, data in unique_candidates.items():
        excerpts_text = "\n---\n".join(data["excerpts"])
        skills = data["metadata"].get("skills", "")

        try:
            evaluation = chain.invoke({
                "job_description": job_description,
                "candidate_name": name,
                "skills": skills,
                "excerpts": excerpts_text
            })

            match_dict = evaluation.model_dump()
            match_dict["resume_path"] = data["metadata"].get("source", "")
            final_matches.append(match_dict)

        except Exception as e:
            print(f"Failed to evaluate candidate {name}: {e}")

    # Sort by match score descending
    final_matches = sorted(final_matches, key=lambda x: x["match_score"], reverse=True)

    output = {
        "job_description": job_description.strip(),
        "top_matches": final_matches[:top_k]
    }

    return json.dumps(output, indent=2)


if __name__ == "__main__":
    jd = """
    Looking for a Senior Data Scientist. 
    Must have 5+ years of experience in Python, Machine Learning, and building RAG pipelines.
    Experience with LangChain and vector databases (Chroma/Pinecone) is highly preferred.
    """

    result_json = match_job(jd, top_k=3, min_experience=5)
    print(result_json)