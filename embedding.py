from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

# d = {}
splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # characters per chunk
        chunk_overlap=50     # overlap between chunks
    )

all_chunks = []
all_metadatas = []


reader = PdfReader("ED_Exp_7.pdf")
for page_num, page in enumerate(reader.pages):
    text = page.extract_text()
    text = text.replace("\n", " ")
    # d[page_num+1] = text
    chunks = splitter.split_text(text)   # returns list[str]
    # print(chunks)
    for chunk in chunks:
        all_chunks.append(chunk)
        all_metadatas.append({"source":"ED_Exp_7.pdf","page":page_num+1})

# print(all_chunks[0])
# print("The page number of this chunk is:",all_metadatas[0]["page"])

all_embeddings = []
model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good default
all_embeddings = model.encode(all_chunks)   # returns list of vectors (numpy arrays)
    # chunk_embedding.append(embeddings)

collection.add(
    ids=[f"chunk_{i+1}" for i in range(len(all_chunks))],
    embeddings=all_embeddings.tolist(),
    documents=all_chunks,
    metadatas=all_metadatas
)


query_embedding = model.encode(["what is Digital Prototyoing?"])
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3
)
print(results["documents"])
print(results["metadatas"])



# print(d)