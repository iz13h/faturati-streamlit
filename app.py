# app.py — واجهة محسّنة واحترافية
import os, io, base64
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from openai import OpenAI

# -------------------- إعدادات عامة --------------------
st.set_page_config(
    page_title="فاتورتي | استخراج اسم المنتج والتاريخ والكمية",
    page_icon="🧾",
    layout="wide",
)

APP_TITLE = "فاتورتي"
MODEL_DEFAULT = "gpt-5"

# -------------------- تنسيقات (CSS) --------------------
st.markdown("""
<style>
:root { --brand:#2563eb; --card:rgba(255,255,255,.06); --line:rgba(255,255,255,.12); }
html, body, .stApp { background: radial-gradient(1200px 600px at 0% 0%, rgba(37,99,235,.18), rgba(37,99,235,.06) 40%, #0e1326 100%); color:#e6e9f0; }
.block-container { padding-top: 1.3rem; }
.header {
  display:flex; align-items:center; gap:.9rem; padding:18px 20px; border-radius:18px;
  border:1px solid var(--line); background: var(--card); box-shadow:0 10px 30px rgba(0,0,0,.25);
}
.logo { width:40px; height:40px; border-radius:12px; background:linear-gradient(135deg, var(--brand), #4f46e5); display:flex; align-items:center; justify-content:center; font-weight:700; }
.card { background: var(--card); border:1px solid var(--line); border-radius:16px; padding:18px; }
.badge { display:inline-flex; gap:.5rem; align-items:center; border:1px solid var(--line); border-radius:999px; padding:.25rem .6rem; font-size:.78rem; opacity:.9; }
hr.dash { border:none; border-top:1px dashed var(--line); margin:1rem 0; }
.stButton>button { background:var(--brand) !important; color:#fff !important; border-radius:12px !important; font-weight:600 !important; padding:.6rem 1.1rem !important; box-shadow:0 6px 18px rgba(37,99,235,.25); }
.stDownloadButton>button{ border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)

# -------------------- مخطط الخرج --------------------
INVOICE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "product_name": {"type": "string", "description": "اسم أو وصف المنتج الرئيسي"},
        "date": {"type": "string", "description": "تاريخ الشراء/الفاتورة بصيغة YYYY-MM-DD إن أمكن"},
        "quantity": {
            "anyOf": [{"type": "integer"}, {"type": "number"}, {"type": "string"}],
            "description": "الكمية — فضّل رقم، وإن تعذّر فالنص الظاهر"
        },
    },
    "required": ["product_name", "date", "quantity"],
    "additionalProperties": False
}

SYSTEM_PROMPT = """You are an expert invoice parser.
Extract ONLY these three fields: product_name, date, quantity.
- If multiple items exist, pick the PRIMARY item with the highest total or the first line item.
- Normalize date to YYYY-MM-DD when possible.
- If a field is missing or unknowable, set it to an empty string.
Produce STRICT JSON conforming to the provided JSON Schema, no extra keys.
"""

# -------------------- وظائف مساعدة --------------------
def get_client() -> OpenAI | None:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ لا يوجد مفتاح OpenAI. أضفه من Settings → Secrets باسم OPENAI_API_KEY.")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"تعذّر إنشاء عميل OpenAI: {e}")
        return None

def pdf_to_images(file_bytes: bytes, dpi: int = 220) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            imgs.append(img)
    except Exception:
        pass
    return imgs

def image_msg(img: Image.Image) -> Dict[str, Any]:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"role":"user","content":[
        {"type":"input_text","text":"Extract product_name, date, and quantity from this invoice/receipt image."},
        {"type":"input_image","image_data":b64,"mime_type":"image/png"}
    ]}

def text_msg(text: str) -> Dict[str, Any]:
    return {"role":"user","content":[
        {"type":"input_text","text":"Extract product_name, date, and quantity from the following invoice text:\n\n"+text}
    ]}

def call_gpt_parse(client: OpenAI, messages: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
    """يستخدم Structured Outputs (يتطلب openai>=1.43)."""
    resp = client.responses.create(
        model=model_id,
        reasoning={"effort": "low"},
        response_format={
            "type":"json_schema",
            "json_schema":{"name":"invoice_schema","schema":INVOICE_SCHEMA,"strict":True}
        },
        messages=[{"role":"system","content":SYSTEM_PROMPT}, *messages]
    )
    # المخرجات المهيكلة مباشرة
    try:
        data = resp.output_parsed
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # بديل: قراءة النص وتحويله JSON
    txt = getattr(resp, "output_text", "") or ""
    if not txt:
        try:
            parts = resp.output[0].content
            if parts and hasattr(parts[0], "text"): txt = parts[0].text
        except Exception: pass
    if not txt: raise RuntimeError("No text returned from model.")
    import json
    s, e = txt.find("{"), txt.rfind("}")
    if s == -1 or e == -1: raise RuntimeError("Model did not return valid JSON.")
    return json.loads(txt[s:e+1])

def parse_file(client: OpenAI, file) -> Dict[str, Any]:
    suffix = os.path.splitext(file.name)[1].lower()
    model = st.session_state.get("model_id", MODEL_DEFAULT)
    if suffix == ".pdf":
        b = file.read()
        imgs = pdf_to_images(b, dpi=220)
        if imgs:
            return call_gpt_parse(client, [image_msg(imgs[0])], model)
        # PDF نصي: جرّب نص
        txt = ""
        try:
            reader = PdfReader(io.BytesIO(b))
            for p in reader.pages: txt += (p.extract_text() or "") + "\n"
        except Exception: pass
        if txt.strip():
            return call_gpt_parse(client, [text_msg(txt)], model)
        return {"product_name":"","date":"","quantity":""}
    else:
        img = Image.open(file).convert("RGB")
        return call_gpt_parse(client, [image_msg(img)], model)

# -------------------- الهيدر --------------------
st.markdown(f"""
<div class="header">
  <div class="logo">🧾</div>
  <div>
    <div style="font-weight:800; font-size:1.2rem;">{APP_TITLE}</div>
    <div style="opacity:.85; font-size:.92rem;">استخراج <b>اسم المنتج</b> و<b>تاريخ الفاتورة</b> و<b>الكمية</b> من PDF/صور باستخدام GPT-5</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.write("")

# -------------------- الإعدادات --------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    cols = st.columns([1,1,2])
    with cols[0]:
        st.session_state["model_id"] = st.text_input("Model ID", value=MODEL_DEFAULT, help="مثل: gpt-5 أو gpt-5-thinking")
    with cols[1]:
        st.markdown("**حالة المفتاح**")
        st.caption("يُقرأ من Settings → Secrets باسم OPENAI_API_KEY.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- الرفع والتنفيذ --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### ارفع الفواتير")
files = st.file_uploader("PDF, JPG, PNG, WEBP (حتى 200MB لكل ملف)", type=["pdf","jpg","jpeg","png","webp"], accept_multiple_files=True)
st.markdown('</div>', unsafe_allow_html=True)

client = get_client()
results: List[Dict[str, Any]] = []

if st.button("استخراج الحقول", disabled=not files or client is None):
    with st.spinner("جارِ تحليل الفواتير..."):
        for f in files:
            try:
                data = parse_file(client, f)
                results.append({"file": f.name,
                                "product_name": data.get("product_name",""),
                                "date": data.get("date",""),
                                "quantity": data.get("quantity","")})
            except Exception as e:
                results.append({"file": f.name, "product_name":"", "date":"", "quantity":"", "error": str(e)})

# -------------------- النتائج --------------------
if results:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### النتائج")
    df = pd.DataFrame(results, columns=["file","product_name","date","quantity","error"])
    st.dataframe(df, use_container_width=True, height=360)
    st.download_button("⬇️ تنزيل CSV", df.to_csv(index=False).encode("utf-8-sig"), "invoices_extracted.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

st.caption("ملاحظة: لملفات PDF المصوّرة، يتم تحويل الصفحة إلى صورة داخليًا (PyMuPDF). عند فشل الاستخراج سيظهر السبب في عمود error.")
