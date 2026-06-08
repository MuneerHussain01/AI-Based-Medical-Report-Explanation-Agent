"""
Medical Report Explanation AI Agent
====================================
Backend: Flask + Groq API (LLaMA 3.3 70B)
"""

from flask import Flask, request, jsonify, render_template_string, send_file
import requests
import json
import os
import io
import base64
from datetime import datetime

# Try importing PDF libraries
try:
    import PyPDF2
    PDF_READ_AVAILABLE = True
except ImportError:
    PDF_READ_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    PDF_GEN_AVAILABLE = True
except ImportError:
    PDF_GEN_AVAILABLE = False

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# GROQ API KEY — hardcoded in backend
# ─────────────────────────────────────────────────────────────
GROQ_API_KEY = "Put Your API Key"

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Medical Report Explanation AI Agent designed to help patients understand their medical laboratory reports.

Your responsibilities:
1. Analyze medical lab values (blood tests, urine tests, liver function, thyroid, lipid profile, etc.)
2. Compare each value against standard reference ranges
3. Clearly mark each result as: HIGH, NORMAL, or LOW
4. Explain in simple, non-technical English what each value means
5. Identify which organ/body system each test relates to
6. Mention possible general causes of abnormal values (do NOT diagnose)
7. Provide general wellness tips
8. Always strongly recommend consulting a qualified doctor

OUTPUT FORMAT (always follow this structure):

## Report Summary
Brief overview of the report in 2-3 sentences.

## Value-by-Value Analysis
For each test value provided:
- **Test Name**: [value] [unit]
  - Status: HIGH / NORMAL / LOW
  - Normal Range: [range]
  - What it means: [simple explanation]
  - Related to: [organ/system]

## Key Observations
- List the most important findings

## General Health Tips
- Practical lifestyle suggestions based on the report

## When to See a Doctor
- Specific warning signs from this report that need medical attention

## Important Disclaimer
Always end with: "This analysis is for educational purposes only. Please consult a qualified medical doctor for proper diagnosis and treatment."

RULES:
- Never diagnose any disease definitively
- Never prescribe or recommend specific medications
- Use simple, clear English
- Be compassionate and reassuring in tone
- If values are critically abnormal, urge immediate medical attention
"""


# ─────────────────────────────────────────────────────────────
# GROQ API CALL
# ─────────────────────────────────────────────────────────────
def call_groq_api(user_message):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.3,
        "max_tokens": 2048
    }
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    if response.status_code == 200:
        data = response.json()
        return {"success": True, "response": data["choices"][0]["message"]["content"]}
    else:
        error_data = response.json()
        return {"success": False, "error": error_data.get("error", {}).get("message", "API call failed")}


# ─────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes):
    if not PDF_READ_AVAILABLE:
        return None, "PyPDF2 not installed"
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip(), None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────
# PDF REPORT GENERATION
# ─────────────────────────────────────────────────────────────
def generate_pdf_report(analysis_text):
    if not PDF_GEN_AVAILABLE:
        return None, "reportlab not installed"
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=50, leftMargin=50,
            topMargin=50, bottomMargin=50
        )

        styles = getSampleStyleSheet()
        story = []

        # Header
        header_style = ParagraphStyle(
            'Header', parent=styles['Normal'],
            fontSize=18, fontName='Helvetica-Bold',
            textColor=colors.HexColor('#1a4a7a'),
            spaceAfter=4, alignment=TA_CENTER
        )
        sub_style = ParagraphStyle(
            'Sub', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#607080'),
            spaceAfter=2, alignment=TA_CENTER
        )
        story.append(Paragraph("Medical Report Analysis", header_style))
        story.append(Paragraph("AI-Powered Lab Report Explanation", sub_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", sub_style))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2d7dd2')))
        story.append(Spacer(1, 14))

        # Body text
        body_style = ParagraphStyle(
            'Body', parent=styles['Normal'],
            fontSize=10, leading=16,
            textColor=colors.HexColor('#1a2840'),
            spaceAfter=6
        )
        h2_style = ParagraphStyle(
            'H2', parent=styles['Normal'],
            fontSize=12, fontName='Helvetica-Bold',
            textColor=colors.HexColor('#1a4a7a'),
            spaceBefore=12, spaceAfter=6
        )

        # Remove emojis and non-Latin1 characters that ReportLab can't handle
        import re
        def clean_text(t):
            # Remove emoji and other symbols outside basic Latin + Latin Extended
            return re.sub(r'[^\x00-\xFF]', '', t)
        analysis_text = clean_text(analysis_text)

        for line in analysis_text.split('\n'):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 4))
            elif line.startswith('## '):
                story.append(Paragraph(line[3:], h2_style))
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#d0dcea')))
                story.append(Spacer(1, 4))
            elif line.startswith('### '):
                bold_style = ParagraphStyle('Bold', parent=body_style, fontName='Helvetica-Bold', fontSize=10)
                story.append(Paragraph(line[4:], bold_style))
            elif line.startswith('- '):
                bullet_style = ParagraphStyle('Bullet', parent=body_style, leftIndent=16, bulletIndent=6)
                story.append(Paragraph(f"• {line[2:]}", bullet_style))
            else:
                # Strip bold markers
                clean = line.replace('**', '')
                story.append(Paragraph(clean, body_style))

        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#d0dcea')))
        disclaimer_style = ParagraphStyle(
            'Disc', parent=styles['Normal'],
            fontSize=8, textColor=colors.HexColor('#99aabb'),
            spaceBefore=8, alignment=TA_CENTER
        )
        story.append(Paragraph(
            "WARNING: This report is for educational purposes only and does not replace professional medical advice. "
            "Always consult a qualified doctor for diagnosis and treatment.",
            disclaimer_style
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Medical Report AI Agent</title>
<style>
  :root {
    --primary: #1a4a7a;
    --primary-light: #2d7dd2;
    --primary-pale: #e8f2fb;
    --success: #27ae60;
    --success-bg: #e8f8ee;
    --danger: #c0392b;
    --danger-bg: #fdecea;
    --warning: #e67e22;
    --warning-bg: #fff8e1;
    --gray: #607080;
    --border: #d0dcea;
    --bg: #f0f4fa;
    --white: #ffffff;
    --text: #1a2840;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); min-height: 100vh; color: var(--text); }

  .container { max-width: 980px; margin: 0 auto; display: flex; flex-direction: column; min-height: 100vh; background: var(--white); box-shadow: 0 4px 40px rgba(26,74,122,0.10); border-radius: 0 0 16px 16px; }

  /* HEADER */
  header {
    background: linear-gradient(120deg, #0f2f55 0%, #1a4a7a 50%, #2d7dd2 100%);
    color: white;
    padding: 22px 32px;
    display: flex;
    align-items: center;
    gap: 18px;
    border-radius: 0;
  }
  .header-logo {
    width: 52px; height: 52px;
    background: rgba(255,255,255,0.15);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    border: 1px solid rgba(255,255,255,0.25);
    box-shadow: 0 2px 12px rgba(0,0,0,0.15);
  }
  .header-info h1 { font-size: 21px; font-weight: 700; letter-spacing: -0.3px; }
  .header-info p { font-size: 12px; opacity: 0.7; margin-top: 3px; }
  .header-pill {
    margin-left: auto;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 30px;
    padding: 6px 16px;
    font-size: 11px;
    font-weight: 600;
    display: flex; align-items: center; gap: 6px;
  }
  .status-dot { width: 7px; height: 7px; background: #4ade80; border-radius: 50%; box-shadow: 0 0 6px #4ade80; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

  /* QUICK PROMPTS */
  .quick-section { padding: 12px 24px; background: #f8faff; border-bottom: 1px solid var(--border); }
  .quick-label { font-size: 11px; font-weight: 700; color: var(--gray); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 9px; }
  .quick-btns { display: flex; gap: 8px; flex-wrap: wrap; }
  .quick-btn {
    background: var(--white);
    border: 1px solid var(--border);
    color: var(--primary);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.18s;
    font-weight: 500;
  }
  .quick-btn:hover { background: var(--primary-light); color: white; border-color: var(--primary-light); transform: translateY(-1px); }

  /* CHAT AREA */
  .chat-area { flex: 1; overflow-y: auto; padding: 28px 32px; display: flex; flex-direction: column; gap: 22px; min-height: 400px; }

  .welcome-card {
    text-align: center;
    padding: 38px 28px;
    border: 1.5px dashed var(--border);
    border-radius: 20px;
    background: linear-gradient(135deg, #f0f6ff 0%, #e8f2fb 100%);
    margin: 8px 0;
  }
  .welcome-card .icon { font-size: 60px; margin-bottom: 16px; filter: drop-shadow(0 4px 8px rgba(26,74,122,0.15)); }
  .welcome-card h2 { color: var(--primary); font-size: 22px; margin-bottom: 10px; font-weight: 700; }
  .welcome-card p { color: var(--gray); font-size: 14px; line-height: 1.8; }
  .welcome-features { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-top: 20px; }
  .feat-chip {
    background: white;
    border: 1px solid var(--border);
    color: var(--primary);
    border-radius: 30px;
    padding: 7px 16px;
    font-size: 12px;
    font-weight: 500;
    display: flex; align-items: center; gap: 6px;
  }

  /* MESSAGES */
  .msg { display: flex; gap: 12px; animation: slideIn 0.28s ease; }
  .msg.user { flex-direction: row-reverse; }
  @keyframes slideIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

  .avatar { width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 17px; flex-shrink: 0; margin-top: 4px; }
  .avatar.bot { background: var(--primary-pale); border: 1px solid var(--border); }
  .avatar.user { background: linear-gradient(135deg, var(--primary-light), var(--primary)); color: white; }

  .bubble { max-width: 82%; }
  .bubble-inner { padding: 14px 18px; border-radius: 18px; font-size: 14px; line-height: 1.75; }
  .bubble.bot .bubble-inner { background: #f4f8fe; border: 1px solid var(--border); border-bottom-left-radius: 4px; }
  .bubble.user .bubble-inner { background: linear-gradient(135deg, var(--primary-light), var(--primary)); color: white; border-bottom-right-radius: 4px; }
  .bubble-meta { font-size: 11px; color: #aab; margin-top: 5px; padding: 0 4px; display: flex; align-items: center; gap: 8px; }
  .msg.user .bubble-meta { justify-content: flex-end; }

  /* Download button inside message */
  .download-btn {
    background: var(--primary-light);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    transition: background 0.2s;
    margin-top: 10px;
    text-decoration: none;
  }
  .download-btn:hover { background: var(--primary); }

  /* BOT MARKDOWN */
  .bubble.bot h2 { font-size: 14px; font-weight: 700; color: var(--primary); margin: 16px 0 8px; padding-top: 14px; border-top: 1px solid var(--border); }
  .bubble.bot h2:first-child { margin-top: 0; padding-top: 0; border-top: none; }
  .bubble.bot h3 { font-size: 13px; font-weight: 700; color: #345; margin: 10px 0 4px; }
  .bubble.bot strong { font-weight: 700; color: var(--primary); }
  .bubble.bot ul { padding-left: 20px; margin: 6px 0; }
  .bubble.bot ul li { margin: 4px 0; }
  .bubble.bot p { margin: 5px 0; }

  /* STATUS TAGS */
  .tag { display: inline-block; border-radius: 5px; padding: 2px 9px; font-size: 11px; font-weight: 700; letter-spacing: 0.3px; }
  .tag-high { background: var(--danger-bg); color: var(--danger); border: 1px solid #f5bcb8; }
  .tag-low  { background: var(--success-bg); color: var(--success); border: 1px solid #b8e8c8; }
  .tag-norm { background: var(--primary-pale); color: var(--primary); border: 1px solid #c0d8f0; }

  /* TYPING */
  .typing-wrap { display: flex; gap: 12px; align-items: center; }
  .typing-bubble { background: #f4f8fe; border: 1px solid var(--border); border-radius: 18px; border-bottom-left-radius: 4px; padding: 14px 18px; display: flex; gap: 5px; align-items: center; }
  .dot { width: 8px; height: 8px; background: #90aabe; border-radius: 50%; animation: bounce 1.2s infinite; }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-7px)} }

  /* INPUT AREA */
  .input-section { padding: 16px 24px; border-top: 1px solid var(--border); background: var(--white); }

  /* PDF Upload Zone */
  .upload-zone {
    border: 1.5px dashed var(--border);
    border-radius: 12px;
    padding: 11px 16px;
    text-align: center;
    cursor: pointer;
    background: #f8faff;
    font-size: 12px;
    color: var(--gray);
    transition: all 0.2s;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  .upload-zone:hover, .upload-zone.dragover { border-color: var(--primary-light); background: var(--primary-pale); color: var(--primary); }
  .upload-zone.has-file { border-color: var(--success); background: var(--success-bg); color: var(--success); }
  #pdfFileInput { display: none; }

  .input-row { display: flex; gap: 10px; align-items: flex-end; }
  textarea { flex: 1; border: 1.5px solid var(--border); border-radius: 14px; padding: 12px 16px; font-size: 14px; font-family: inherit; resize: none; min-height: 48px; max-height: 140px; outline: none; color: var(--text); transition: border 0.2s; line-height: 1.5; }
  textarea:focus { border-color: var(--primary-light); box-shadow: 0 0 0 3px rgba(45,125,210,0.1); }
  textarea::placeholder { color: #b0bec5; }
  .send-btn {
    background: linear-gradient(135deg, var(--primary-light), var(--primary));
    color: white;
    border: none;
    border-radius: 13px;
    width: 50px; height: 50px;
    font-size: 20px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s;
    flex-shrink: 0;
    box-shadow: 0 3px 10px rgba(45,125,210,0.3);
  }
  .send-btn:hover { transform: translateY(-1px); box-shadow: 0 5px 14px rgba(45,125,210,0.4); }
  .send-btn:disabled { background: #b0c4de; cursor: not-allowed; box-shadow: none; transform: none; }
  .input-hint { font-size: 11px; color: #aab; margin-top: 8px; }

  /* DISCLAIMER */
  .disclaimer { text-align: center; font-size: 11px; color: #99a; padding: 10px 20px 16px; background: var(--white); border-top: 1px solid var(--border); }

  /* ERROR */
  .error-bubble { background: var(--danger-bg); border: 1px solid #f5bcb8; border-radius: 12px; padding: 12px 16px; font-size: 13px; color: var(--danger); }

  /* SCROLLBAR */
  .chat-area::-webkit-scrollbar { width: 5px; }
  .chat-area::-webkit-scrollbar-track { background: transparent; }
  .chat-area::-webkit-scrollbar-thumb { background: #ccd; border-radius: 10px; }

  @media (max-width: 600px) {
    .chat-area { padding: 16px; }
    header { padding: 16px; }
    .input-section { padding: 12px 14px; }
  }
</style>
</head>
<body>
<div class="container">

  <!-- HEADER -->
  <header>
    <div class="header-logo">🏥</div>
    <div class="header-info">
      <h1>Medical Report AI Agent</h1>
      <p>Intelligent Lab Report Analysis &nbsp;·&nbsp; Powered by LLaMA 3.3 70B</p>
    </div>
    <div class="header-pill">
      <span class="status-dot"></span> Online
    </div>
  </header>

  <!-- QUICK PROMPTS -->
  <div class="quick-section">
    <div class="quick-label">Quick Examples</div>
    <div class="quick-btns">
      <button class="quick-btn" onclick="setQ('CBC Blood Test: WBC 11.5 K/uL, RBC 4.2 M/uL, Hemoglobin 11.8 g/dL, Hematocrit 36%, Platelets 180 K/uL, Neutrophils 75%, Lymphocytes 18%')">🩸 CBC Blood Test</button>
      <button class="quick-btn" onclick="setQ('Diabetes Panel: Fasting Blood Glucose 126 mg/dL, HbA1c 7.2%, Post-meal glucose 198 mg/dL')">🍬 Diabetes Panel</button>
      <button class="quick-btn" onclick="setQ('Liver Function Test: ALT 65 U/L, AST 72 U/L, Total Bilirubin 2.1 mg/dL, ALP 130 U/L, Total Protein 6.8 g/dL, Albumin 3.9 g/dL')">🫀 Liver Function</button>
      <button class="quick-btn" onclick="setQ('Thyroid Function: TSH 6.8 mIU/L, Free T4 0.7 ng/dL, Free T3 2.8 pg/mL')">🦋 Thyroid Test</button>
      <button class="quick-btn" onclick="setQ('Lipid Profile: Total Cholesterol 240 mg/dL, LDL 160 mg/dL, HDL 38 mg/dL, Triglycerides 220 mg/dL, VLDL 44 mg/dL')">💊 Lipid Profile</button>
      <button class="quick-btn" onclick="setQ('Urine Routine: Color Yellow, Turbidity Cloudy, Protein +1, Glucose Nil, Pus Cells 10-15/hpf, RBC 2-3/hpf')">🧪 Urine Test</button>
      <button class="quick-btn" onclick="setQ('Kidney Function: Serum Creatinine 1.8 mg/dL, BUN 28 mg/dL, Uric Acid 8.2 mg/dL, eGFR 55 mL/min')">🫘 Kidney Function</button>
    </div>
  </div>

  <!-- CHAT AREA -->
  <div class="chat-area" id="chatArea">
    <div class="welcome-card" id="welcomeCard">
      <div class="icon">🩺</div>
      <h2>Welcome to Medical Report AI Agent</h2>
      <p>Paste your lab report values or upload a PDF — the AI will explain<br>each result in simple, easy-to-understand English.</p>
      <div class="welcome-features">
        <div class="feat-chip">✅ HIGH / NORMAL / LOW detection</div>
        <div class="feat-chip">📋 Step-by-step value analysis</div>
        <div class="feat-chip">📤 Upload PDF reports</div>
        <div class="feat-chip">📥 Download analysis as PDF</div>
        <div class="feat-chip">⚠️ Always recommends a doctor</div>
      </div>
    </div>
  </div>

  <!-- INPUT AREA -->
  <div class="input-section">
    <!-- PDF Upload -->
    <div class="upload-zone" id="uploadZone" onclick="document.getElementById('pdfFileInput').click()" ondragover="onDragOver(event)" ondragleave="onDragLeave(event)" ondrop="onDrop(event)">
      <span>📄</span>
      <span id="uploadLabel">Upload PDF Report &nbsp;(click or drag & drop)</span>
    </div>
    <input type="file" id="pdfFileInput" accept=".pdf" onchange="onFileSelect(event)">

    <div class="input-row">
      <textarea id="userInput"
        placeholder="Or type / paste your lab values here... (e.g. WBC 11.5, Hemoglobin 13.2 g/dL, Platelets 180)"
        rows="2"
        onkeydown="handleKey(event)"
        oninput="autoResize(this)"></textarea>
      <button class="send-btn" id="sendBtn" onclick="sendMessage()" title="Send">&#10148;</button>
    </div>
    <div class="input-hint">Press Enter to send &nbsp;|&nbsp; Shift+Enter for new line</div>
  </div>

  <div class="disclaimer">
    ⚠️ This AI agent is for educational purposes only. It does not replace professional medical advice. Always consult a qualified doctor for diagnosis and treatment.
  </div>

</div>

<script>
let isLoading = false;
let chatHistory = [];
let lastAnalysisText = '';

// ─────────────────────────────────────────────
// PDF UPLOAD HANDLERS
// ─────────────────────────────────────────────
function onDragOver(e) {
  e.preventDefault();
  document.getElementById('uploadZone').classList.add('dragover');
}
function onDragLeave(e) {
  document.getElementById('uploadZone').classList.remove('dragover');
}
function onDrop(e) {
  e.preventDefault();
  document.getElementById('uploadZone').classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type === 'application/pdf') handlePDF(file);
  else alert('Please drop a valid PDF file.');
}
function onFileSelect(e) {
  const file = e.target.files[0];
  if (file) handlePDF(file);
}

async function handlePDF(file) {
  if (file.size > 10 * 1024 * 1024) { alert('PDF too large. Max 10 MB.'); return; }
  const zone = document.getElementById('uploadZone');
  zone.classList.add('has-file');
  document.getElementById('uploadLabel').textContent = `✅ ${file.name} — uploading…`;

  const reader = new FileReader();
  reader.onload = async function(e) {
    const base64 = e.target.result.split(',')[1];
    if (!base64 || isLoading) return;

    isLoading = true;
    document.getElementById('sendBtn').disabled = true;

    const area = document.getElementById('chatArea');
    const welcome = document.getElementById('welcomeCard');
    if (welcome) welcome.remove();

    addMessage('user', `📄 Uploaded PDF: <strong>${escHtml(file.name)}</strong>`, true);
    showTyping();
    document.getElementById('uploadLabel').textContent = `📄 ${file.name} (uploaded)`;

    try {
      const response = await fetch('/api/analyze-pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pdf_base64: base64, filename: file.name })
      });
      const data = await response.json();
      removeTyping();

      if (data.success) {
        lastAnalysisText = data.response;
        const formatted = formatBotResponse(data.response);
        addBotMessage(formatted);
        chatHistory.push({ role: 'assistant', content: data.response });
      } else {
        addMessage('bot', `<div class="error-bubble"><strong>Error:</strong> ${escHtml(data.error)}</div>`, true);
      }
    } catch (err) {
      removeTyping();
      addMessage('bot', `<div class="error-bubble"><strong>Error:</strong> ${escHtml(err.message)}</div>`, true);
    }
    isLoading = false;
    document.getElementById('sendBtn').disabled = false;
  };
  reader.readAsDataURL(file);
}

// ─────────────────────────────────────────────
// SEND TEXT MESSAGE
// ─────────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById('userInput');
  const text = input.value.trim();
  if (!text || isLoading) return;

  isLoading = true;
  document.getElementById('sendBtn').disabled = true;
  input.value = '';
  input.style.height = 'auto';

  addMessage('user', text);
  chatHistory.push({ role: 'user', content: text });
  showTyping();

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, history: chatHistory.slice(-6) })
    });
    const data = await response.json();
    removeTyping();

    if (data.success) {
      lastAnalysisText = data.response;
      const formatted = formatBotResponse(data.response);
      addBotMessage(formatted);
      chatHistory.push({ role: 'assistant', content: data.response });
    } else {
      addMessage('bot', `<div class="error-bubble"><strong>Error:</strong> ${escHtml(data.error)}<br><br>Please try again.</div>`, true);
    }
  } catch (err) {
    removeTyping();
    addMessage('bot', `<div class="error-bubble"><strong>Connection Error:</strong> ${escHtml(err.message)}</div>`, true);
  }
  isLoading = false;
  document.getElementById('sendBtn').disabled = false;
  input.focus();
}

// ─────────────────────────────────────────────
// ADD BOT MESSAGE WITH DOWNLOAD BUTTON
// ─────────────────────────────────────────────
function addBotMessage(htmlContent) {
  const area = document.getElementById('chatArea');
  const wrap = document.createElement('div');
  wrap.className = 'msg bot';
  wrap.innerHTML = `
    <div class="avatar bot">🤖</div>
    <div class="bubble bot">
      <div class="bubble-inner">${htmlContent}</div>
      <div class="bubble-meta">
        ${getTime()}
        <button class="download-btn" onclick="downloadReport(this)">⬇️ Download PDF</button>
      </div>
    </div>`;
  area.appendChild(wrap);
  area.scrollTop = area.scrollHeight;
}

// ─────────────────────────────────────────────
// DOWNLOAD PDF REPORT
// ─────────────────────────────────────────────
async function downloadReport(btn) {
  if (!lastAnalysisText) { alert('No analysis to download yet.'); return; }
  btn.disabled = true;
  btn.textContent = '⏳ Generating PDF…';
  try {
    const response = await fetch('/api/generate-pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: lastAnalysisText })
    });
    if (!response.ok) throw new Error('PDF generation failed');
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'medical_report_analysis.pdf';
    a.click();
    URL.revokeObjectURL(url);
    btn.textContent = '✅ Downloaded!';
    btn.style.background = '#27ae60';
    setTimeout(() => { btn.textContent = '⬇️ Download PDF'; btn.style.background = ''; btn.disabled = false; }, 3000);
  } catch (err) {
    alert('PDF download failed: ' + err.message);
    btn.textContent = '⬇️ Download PDF';
    btn.disabled = false;
  }
}

// ─────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────
function addMessage(role, content, isHtml = false) {
  const area = document.getElementById('chatArea');
  const welcome = document.getElementById('welcomeCard');
  if (welcome) welcome.remove();

  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const icon = role === 'user' ? '👤' : '🤖';
  const innerHtml = isHtml ? content : escHtml(content);
  wrap.innerHTML = `
    <div class="avatar ${role}">${icon}</div>
    <div class="bubble ${role}">
      <div class="bubble-inner">${innerHtml}</div>
      <div class="bubble-meta">${getTime()}</div>
    </div>`;
  area.appendChild(wrap);
  area.scrollTop = area.scrollHeight;
}

function showTyping() {
  const area = document.getElementById('chatArea');
  const d = document.createElement('div');
  d.id = 'typingWrap'; d.className = 'typing-wrap';
  d.innerHTML = `<div class="avatar bot">🤖</div>
    <div class="typing-bubble">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
      <span style="font-size:12px;color:#607080;margin-left:8px;">Analyzing report…</span>
    </div>`;
  area.appendChild(d);
  area.scrollTop = area.scrollHeight;
}
function removeTyping() { const t = document.getElementById('typingWrap'); if (t) t.remove(); }

function autoResize(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 140) + 'px'; }
function setQ(text) { const ta = document.getElementById('userInput'); ta.value = text; autoResize(ta); ta.focus(); }
function handleKey(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } }
function escHtml(t) { return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function getTime() { return new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' }); }

function formatBotResponse(text) {
  let html = escHtml(text);
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
  html = html.replace(/\\bSTATUS:\\s*HIGH\\b/gi,   '<span class="tag tag-high">&#9650; HIGH</span>');
  html = html.replace(/\\bStatus:\\s*HIGH\\b/gi,   '<span class="tag tag-high">&#9650; HIGH</span>');
  html = html.replace(/\\bSTATUS:\\s*LOW\\b/gi,    '<span class="tag tag-low">&#9660; LOW</span>');
  html = html.replace(/\\bStatus:\\s*LOW\\b/gi,    '<span class="tag tag-low">&#9660; LOW</span>');
  html = html.replace(/\\bSTATUS:\\s*NORMAL\\b/gi, '<span class="tag tag-norm">&#10003; NORMAL</span>');
  html = html.replace(/\\bStatus:\\s*NORMAL\\b/gi, '<span class="tag tag-norm">&#10003; NORMAL</span>');
  html = html.replace(/\\b(HIGH)\\b/g,   '<span class="tag tag-high">&#9650; HIGH</span>');
  html = html.replace(/\\b(LOW)\\b/g,    '<span class="tag tag-low">&#9660; LOW</span>');
  html = html.replace(/\\b(NORMAL)\\b/g, '<span class="tag tag-norm">&#10003; NORMAL</span>');
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\\/li>)/gs, '<ul>$1</ul>');
  html = html.replace(/<\\/ul>\\n<ul>/g, '');
  html = html.replace(/\\n\\n/g, '</p><p>');
  html = html.replace(/\\n/g, '<br>');
  return html;
}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return HTML_PAGE


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data received"}), 400
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({"success": False, "error": "No message provided"}), 400
    result = call_groq_api(user_message)
    return jsonify(result)


@app.route('/api/analyze-pdf', methods=['POST'])
def analyze_pdf():
    data = request.get_json()
    if not data or 'pdf_base64' not in data:
        return jsonify({"success": False, "error": "No PDF data received"}), 400

    try:
        pdf_bytes = base64.b64decode(data['pdf_base64'])
    except Exception:
        return jsonify({"success": False, "error": "Invalid PDF data"}), 400

    text, err = extract_text_from_pdf(pdf_bytes)
    if err:
        return jsonify({"success": False, "error": f"Could not read PDF: {err}"}), 400
    if not text or len(text.strip()) < 20:
        return jsonify({"success": False, "error": "Could not extract text from PDF. The PDF may be image-based or protected."}), 400

    message = f"Here is a medical lab report extracted from a PDF file:\n\n{text}\n\nPlease analyze this report and explain each value."
    result = call_groq_api(message)
    return jsonify(result)


@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"success": False, "error": "No text provided"}), 400

    pdf_buffer, err = generate_pdf_report(data['text'])
    if err:
        return jsonify({"success": False, "error": f"PDF generation failed: {err}"}), 500

    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='medical_report_analysis.pdf'
    )


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "running", "agent": "Medical Report AI Agent", "version": "2.0"})


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  Medical Report AI Agent v2.0")
    print("  Server: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
