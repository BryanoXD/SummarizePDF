#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# TOKENIZAÇÃO / STOPWORDS
# =========================

STOPWORDS = {
    "o","a","os","as","um","uma","uns","umas","e","ou","mas","se","entao","então",
    "de","da","do","das","dos","para","por","com","como","em","no","na","nos","nas",
    "ao","aos","à","às","entre","durante","antes","depois",
    "é","são","foi","foram","ser","sendo","sido","isso","isto",
    "eu","voce","você","ele","ela","nos","nós","eles","elas",
    "meu","minha","seu","sua","não","nao","sim","mais","menos","tambem","também"
}

def word_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-záàâãéêíóôõúüç ]+", " ", text)
    return [t for t in text.split() if len(t) > 2 and t not in STOPWORDS]


# =========================
# LIMPEZA DE TEXTO
# =========================

def fix_pdf_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "\n")

    # remove hifenização quebrada
    text = re.sub(r"(\w)\s*-\s*\n\s*(\w)", r"\1\2", text)

    # normaliza espaços
    text = re.sub(r"[ \t]+", " ", text)

    # marca parágrafos
    text = re.sub(r"\n\s*\n+", "\n<PARA>\n", text)

    # une linhas quebradas
    text = re.sub(r"\n(?!<PARA>)", " ", text)

    # restaura parágrafos
    text = text.replace("<PARA>", "\n\n")

    return text.strip()


def sentence_split(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 30]


# =========================
# PDF EXTRACTION
# =========================

def extract_pdf_text(path: str) -> Tuple[str, int]:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages), len(reader.pages)


# =========================
# SEÇÕES / CAPÍTULOS
# =========================

@dataclass
class Section:
    title: str
    text: str


def looks_like_heading(line: str) -> bool:
    line = line.strip()
    if len(line) < 5 or len(line) > 80:
        return False
    if re.match(r"^\d+(\.\d+)?\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ]", line):
        return True
    if line.lower() in {
        "introdução","introducao","metodologia","metodo",
        "resultados","discussão","discussao",
        "conclusão","conclusao",
        "considerações finais","consideracoes finais",
        "referências","referencias"
    }:
        return True
    if line.isupper():
        return True
    return False


def split_into_sections(text: str) -> List[Section]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sections = []
    current_title = "Documento"
    buffer = []

    for line in lines:
        if looks_like_heading(line):
            if buffer:
                sections.append(Section(current_title, " ".join(buffer)))
                buffer = []
            current_title = line
        else:
            buffer.append(line)

    if buffer:
        sections.append(Section(current_title, " ".join(buffer)))

    return sections


# =========================
# RESUMO COESO (BLOCOS)
# =========================

def summarize_cohesive(text: str, target_sentences=8, window=2) -> str:
    sentences = sentence_split(text)
    if len(sentences) <= target_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()

    picked = set()
    blocks = []

    for idx in np.argsort(scores)[::-1]:
        if idx in picked:
            continue
        start = max(0, idx - window)
        end = min(len(sentences) - 1, idx + window)

        blocks.append((start, end))
        for i in range(start, end + 1):
            picked.add(i)

        if len(blocks) >= 3:
            break

    blocks.sort()
    result = []
    for s, e in blocks:
        result.extend(sentences[s:e+1])

    return " ".join(result[:target_sentences])


def summarize_section(text: str, sentences=6) -> List[str]:
    summary = summarize_cohesive(text, target_sentences=sentences, window=2)
    return sentence_split(summary)


# =========================
# MARKDOWN
# =========================

def make_markdown(path, pages, sections, overall, section_summaries):
    lines = []
    lines.append(f"# Resumo — {os.path.basename(path)}\n")
    lines.append(f"- Páginas: {pages}")
    lines.append(f"- Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## Resumo Geral\n")
    lines.append(overall or "_Resumo não disponível._\n")

    lines.append("\n## Resumo por Capítulo\n")

    for sec in sections:
        lines.append(f"### {sec.title}\n")
        bullets = section_summaries.get(sec.title, [])
        if bullets:
            for b in bullets:
                lines.append(f"- {b}")
        else:
            lines.append("_Sem conteúdo suficiente._")
        lines.append("")

    return "\n".join(lines)


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("-o", "--output")
    parser.add_argument("--sentences", type=int, default=8)
    parser.add_argument("--section-sentences", type=int, default=6)
    args = parser.parse_args()

    text, pages = extract_pdf_text(args.pdf)
    text = fix_pdf_text(text)

    sections = split_into_sections(text)

    overall = summarize_cohesive(text, target_sentences=args.sentences)

    section_summaries = {}
    for sec in sections:
        if len(sec.text) < 400:
            section_summaries[sec.title] = []
        else:
            section_summaries[sec.title] = summarize_section(
                sec.text, sentences=args.section_sentences
            )

    md = make_markdown(args.pdf, pages, sections, overall, section_summaries)

    out = args.output or os.path.splitext(args.pdf)[0] + "_summary.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"✅ Resumo gerado em: {out}")


if __name__ == "__main__":
    main()
