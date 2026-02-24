import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader

# 500 worked better than 1000, less noise in results
# 50 overlap so we don't lose context at boundaries
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_csv(file_path: str, metadata_columns: list = []) -> List[Document]:
    """Load CSV as documents. metadata_columns are kept as filters later."""
    print(f"Loading: {file_path}")
    loader = CSVLoader(
        file_path=file_path,
        metadata_columns=metadata_columns
    )
    return loader.load()


def load_pdfs(folder_path: str) -> List[Document]:
    """Load all PDFs from a folder, page by page."""
    docs = []
    pdfs = list(Path(folder_path).glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs in {folder_path}, skipping")
        return []

    for pdf in pdfs:
        print(f"  → {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())

    return docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split docs into smaller chunks for retrieval.
    Using RecursiveCharacterTextSplitter because it respects
    natural boundaries (paragraphs → sentences → words)
    before making a hard cut. Much better than basic splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Got {len(chunks)} chunks from {len(documents)} docs")
    return chunks


def load_and_chunk_all(data_folder: str = "./data/raw") -> List[Document]:
    """
    Entry point for the ingestion pipeline.
    Loads defect records, SAP data, and any PDFs then chunks everything.
    vectorstore.py calls this to build the knowledge base.
    """
    all_docs = []

    # defect records from Kaggle AI4I dataset
    defect_path = os.path.join(data_folder, "defect_records.csv")
    if os.path.exists(defect_path):
        defect_docs = load_csv(defect_path, metadata_columns=["Type"])
        all_docs.extend(defect_docs)
        print(f"Defect records: {len(defect_docs)} rows loaded")

    # SAP maintenance history mocked for now
    # TODO: replace with live SAP RFC call in production
    sap_path = os.path.join(data_folder, "sap_maintenance.csv")
    if os.path.exists(sap_path):
        sap_docs = load_csv(sap_path)
        all_docs.extend(sap_docs)
        print(f"SAP records: {len(sap_docs)} rows loaded")

    # PDFs maintenance manuals if present
    manuals_path = os.path.join(data_folder, "maintenance_manuals")
    if os.path.exists(manuals_path):
        pdf_docs = load_pdfs(manuals_path)
        all_docs.extend(pdf_docs)

    if not all_docs:
        print("WARNING: nothing to ingest, check data folder")
        return []

    return chunk_documents(all_docs)


if __name__ == "__main__":
    chunks = load_and_chunk_all()
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"\nSample:\n{chunks[0].page_content[:200]}")
    print(f"Metadata: {chunks[0].metadata}")