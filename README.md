# Multi-Agent Research Report Writing System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain Integration](https://img.shields.io/badge/LangChain-0.1.0-green)]()
[![Gradio Interface](https://img.shields.io/badge/Interface-Gradio-FF4B4B)]()

An AI-powered system that collaboratively writes research reports using multiple specialized agents through planning, research, generation, and iterative refinement.

## ✨ Features

- **Multi-Agent Architecture**: Four specialized agents (Planner, Researcher, Writer, Editor) working in sequence
- **Iterative Refinement**: Up to 5 revision cycles with automated quality checks
- **Web Research Integration**: Real-time data gathering using Tavily Search API
- **Quality Control**: Automated reflection/critique system for continuous improvement
- **User-Friendly Interface**: Gradio web interface for interactive report generation

## 🚀 Quick Start

### Prerequisites
```bash
pip install langchain langgraph tavily-python gradio google-generativeai
