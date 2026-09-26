from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from groq import Groq
from dotenv import load_dotenv
import os


chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("my_docs")

model = SentenceTransformer("all-MiniLM-L6-v2")


# d = {}
splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,      # characters per chunk
        chunk_overlap=50     # overlap between chunks
    )

load_dotenv()

api_key = os.getenv("api_key")
client = Groq(api_key=api_key)

def load_pdf(pdf:str):
    all_chunks = []
    all_metadatas = []
    from pypdf import PdfReader
    reader = PdfReader(pdf)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        text = text.replace("\n", " ")
        # d[page_num+1] = text
        chunks = splitter.split_text(text)   # returns list[str]
        # print(chunks)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source":pdf,"page":page_num+1})
    return all_chunks,all_metadatas


# print(all_chunks[0])
# print("The page number of this chunk is:",all_metadatas[0]["page"])
def embed(all_chunks:list,all_metadatas:list):
    all_embeddings = []
    all_embeddings = model.encode(all_chunks)   # returns list of vectors (numpy arrays)
        # chunk_embedding.append(embeddings)

    collection.add(
        ids=[f"chunk_{i+1}" for i in range(len(all_chunks))],
        embeddings=all_embeddings.tolist(),
        documents=all_chunks,
        metadatas=all_metadatas
    )

def retriever(question:str,no_of_results:int):
    query_embedding = model.encode([question])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=no_of_results
    )
    return results["documents"][0],results["metadatas"][0]

def build_context(chunks, metadatas):
    context_parts = []
    for chunk, meta in zip(chunks, metadatas):
        context_parts.append(f"[Page {meta['page']}]\n{chunk}")
    return "\n\n---\n\n".join(context_parts)

def build_prompt(query, context):
    return f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {query}

Answer:"""

def llm_answer(prompt):
    response = client.chat.completions.create(
        model = "openai/gpt-oss-120b",
        messages = [{"role":"user","content":prompt}],
        reasoning_effort="high"
    )
    return response.choices[0].message.content

if collection.count() == 0:
    print("Indexing for the first time...")
    all_chunks, all_metadatas = load_pdf("ED_Exp_7.pdf")
    embed(all_chunks, all_metadatas)

def rag_answer(question,no_of_results=3):
    chunks,metadatas = retriever(question,no_of_results)
    context = build_context(chunks,metadatas)
    prompt = build_prompt(question,context)
    result = llm_answer(prompt)
    return result,metadatas

x = "yes"
while x == "yes":
    question = input("Enter your question:")
    result,source = rag_answer(question)
    print(result)
    print("Sources:")
    # print(source)
    l = []
    for i in range(len(source)):
        if source[i]["page"] not in l:
            print("doc used:",source[i]["source"])
            print("page number:",source[i]["page"])
            l.append(source[i]["page"])
            print("----------------------------------------")

    x = input("U want to ask more questions(Yes or No):")
    x = x.lower()