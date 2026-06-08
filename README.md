# AI-Based-Medical-Report-Explanation-Agent

<img width="1600" height="900" alt="WhatsApp Image 2026-06-08 at 6 04 16 PM" src="https://github.com/user-attachments/assets/a04e9252-c79e-4ac7-b348-d30aa213b83e" />

<img width="1600" height="900" alt="WhatsApp Image 2026-06-08 at 6 05 26 PM" src="https://github.com/user-attachments/assets/214dbeb5-735f-48e5-9f70-873903cb9b82" />

**Overview**
The Medical Report AI Agent is an application that analyzes medical laboratory reports and generates structured, easy-to-understand explanations for patients. Users can upload a PDF lab report or manually enter lab values, and the system uses the LLaMA-3.3-70B model  with Chain-of-Thought prompting to classify each value as HIGH / NORMAL / LOW, explain its clinical significance in plain English, and produce a downloadable PDF report.


**Features**

Dual Input Support — Upload PDF lab reports or paste values directly as text
AI-Powered Analysis — Uses LLaMA-3.3-70B with structured CoT prompting
Status Classification — Each lab value tagged as HIGH / NORMAL / LOW with reference ranges
Downloadable PDF Reports — Clean formatted PDF export via ReportLab
Chat Interface — Conversational UI with real-time AI responses
Safe by Design — No diagnosis, no medication recommendations, mandatory disclaimers


**System Architecture**

User
 │
 ├─ PDF Upload ──► PyPDF2 Text Extraction ──► /api/analyze-pdf
 │
 └─ Text Input ──────────────────────────► /api/analyze
                                                │
                                    (LLaMA-3.3-70B)
                                    CoT System Prompt
                                                │
                                      Structured Analysis
                                    HIGH / NORMAL / LOW
                                                │
                              ┌─────────────────────────┐
                              │  Web Interface Display  │
                              └─────────────────────────┘
                                                │
                                       generate-pdf
                                                │
                                      ReportLab PDF Export
                                      (Download to User)

**Installation**
Requirements

Python 3.10+
pip

