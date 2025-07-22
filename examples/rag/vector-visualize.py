import os
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

load_dotenv(override=True)

model = "text-embedding-3-small"
db_name = "vector_db"


def generate_documents():
    urls = [
        "https://en.wikipedia.org/wiki/William_Shakespeare",
        "https://en.wikipedia.org/wiki/Leonardo_da_Vinci",
    ]
    documents = []
    for url in urls:
        loader = WebBaseLoader(url)
        documents.extend(loader.load())
    for doc in documents:
        # Just to prevent unexpected openai costs if the page size becomes too big
        doc.page_content = doc.page_content[:100_000]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def get_vectorstore():
    embedding = OpenAIEmbeddings(model=model)
    if os.path.exists(db_name):
        vectorstore = Chroma(persist_directory=db_name, embedding_function=embedding)
    else:
        documents = generate_documents()
        vectorstore = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=db_name)
    vector_dimensions = len(vectorstore._collection.get(limit=1, include=["embeddings"])["embeddings"][0])
    print(f"Vectorstore documents: {vectorstore._collection.count()}")
    print(f"Vector dimensions: {vector_dimensions}")
    return vectorstore


def build_colors(titles):
    palette = px.colors.qualitative.Plotly
    title_to_color = {title: palette[index % len(palette)] for index, title in enumerate(set(titles))}
    return [title_to_color[title] for title in titles]


def display(vectorstore):
    result = vectorstore._collection.get(include=['embeddings', 'documents', 'metadatas'])
    vectors = np.array(result['embeddings'])
    documents = result['documents']
    titles = [metadata['title'] for metadata in result['metadatas']]
    colors = build_colors(titles)
    tsne = TSNE(n_components=2, random_state=123)
    reduced_vectors = tsne.fit_transform(vectors)
    go.Figure(
        layout=go.Layout(
            title='Demo',
            scene=dict(xaxis_title='x', yaxis_title='y'),
            width=800,
            height=600,
            margin=dict(r=20, b=10, l=10, t=40)),
        data=[go.Scatter(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            mode='markers',
            marker=dict(size=5, color=colors, opacity=0.8),
            text=[f"Title: {t}<br>Text: {d[:100]}..." for t, d in zip(titles, documents)],
            hoverinfo='text',
        )]).show()


vectorstore = get_vectorstore()
display(vectorstore)
