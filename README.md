# JEPA & Representation Collapse: A Self-Supervised Learning Overview

This repository contains a school project paper analyzing **Joint Embedding Predictive Architectures (JEPAs)**. The paper explores how JEPAs differ from traditional generative self-supervised learning by predicting representations in latent space rather than reconstructing inputs. It specifically addresses the critical challenge of **representation collapse**, outlining mathematical definitions, mitigation strategies (such as architectural asymmetry and SIGReg), and current limitations in the literature.

## 📄 Abstract

> Joint Embedding Predictive Architectures (JEPAs) represent a significant advancement in self-supervised learning. Unlike traditional generative approaches that reconstruct inputs in the original modality (e.g., pixels), JEPAs predict representations directly in latent space. While JEPAs offer high sample efficiency and robustness compared to generative methods, they introduce a critical vulnerability: representation collapse. Preventing this failure mode is essential to ensure that learned embeddings remain informative and preserve fine-grained features for downstream tasks. This overview describes the JEPA framework, provides mathematical definitions of collapse, and outlines mitigation paradigms : including historical contrastive approaches, architectural asymmetry, masking strategies, explicit regularization, and state-of-the-art methods like SIGReg. Finally, we present their limitations discussed in the literature.

## 🛠️ Prerequisites

To compile this project, you need **Typst** installed on your system. Typst is a modern typesetting system that combines the power of LaTeX with the simplicity of Markdown.

### Option 1: Install via Package Manager (Recommended)

- **macOS**: `brew install typst`
- **Ubuntu/Debian**:
  ```bash
  sudo apt install typst
  # Or download the latest release from GitHub if not in repos
  ```
- **Windows**: Download the binary from [Typst Releases](https://github.com/typst/typst/releases) and add it to your PATH.

### Option 2: Install via Cargo (Rust)

If you have Rust installed:

```bash
cargo install typst-cli
```

## 📂 Project Structure

```text
├── paper.pdf       # Compiled output (generated after build)
├── paper.typ       # Main source file for the paper
├── README.md       # This file
└── refs.bib        # Bibliography references in BibTeX format
```

## 🚀 How to Build & Compile

### 1. Install Required Typst Packages

This project uses specific Typst packages for IEEE formatting and diagramming. Ensure you have internet access during the first compile, as Typst will automatically download these dependencies:

- `@preview/charged-ieee`: For IEEE-style formatting.
- `@preview/fletcher`: For creating diagrams (used in JEPA architecture visualization).
- `@preview/cetz`: For vector graphics and custom drawings.

### 2. Compile the Paper

Navigate to the project directory and run:

```bash
typst compile paper.typ
```

This will generate `paper.pdf` from `paper.typ`.

### 3. Watch Mode (For Development)

If you are editing the paper, use watch mode to automatically recompile on save:

```bash
typst watch paper.typ
```
