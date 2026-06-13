import re

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from rag_lab.documents import Document


def load_pdf(file, name: str = "") -> Document:
    reader = PdfReader(file)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return Document(text=text, metadata={"source": name or "pdf"})


def load_text(content: str, name: str = "") -> Document:
    return Document(text=content, metadata={"source": name or "text"})


def load_url(url: str) -> Document:
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # 去掉導覽列、script 等雜訊，只留正文
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n"))
    return Document(text=text.strip(), metadata={"source": url})
