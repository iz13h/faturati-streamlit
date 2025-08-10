
import os
import io
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
from pypdf import PdfReader

from openai import OpenAI

APP_TITLE = "فاتورتي — استخراج (الاسم، التاريخ، الكمية)"
APP_DESC = "ارفع فواتيرك (PDF/صور)، وسيستخرج التطبيق **اسم المنتج + التاريخ + الكمية** باستخدام GPT-5 مع مخرجات منظّمة (JSON Schema)."

MODEL_DEFAULT = "gpt-5"

INVOICE_SCHEMA = {
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

def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY", None)
    if not api_key:
        st.error("ضع مفتاح OpenAI في Secrets أو البيئة.")
        return None
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"تعذّر إنشاء عميل OpenAI: {e}")
        return None

def pdf_to_images(file_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    """Render PDF pages to PIL Images using PyMuPDF (works on Streamlit Cloud without poppler)."""
    imgs: List[Image.Image] = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            mat = fitz.Matrix(dpi/72, dpi/72)  # scale to desired dpi
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            imgs.append(img)
        return imgs
    except Exception:
        # fallback try pypdf to get text and let model parse text directly
        try:
            text = ""
            reader = PdfReader(io.BytesIO(file_bytes))
            for p in reader.pages:
                text += (p.extract_text() or "") + "\n"
            if text.strip():
                # handled later as text path
                return []
        except Exception:
            pass
        return []

def build_vision_message_for_image(img: Image.Image) -> Dict[str, Any]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    import base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Extract product_name, date, and quantity from this invoice/receipt image."},
            {"type": "input_image", "image_data": image_b64, "mime_type": "image/png"}
        ]
    }

def build_text_message_for_pdf_text(text: str) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Extract product_name, date, and quantity from the following invoice text:\n\n"+text}
        ]
    }

def call_gpt_parse(client: OpenAI, messages: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
    resp = client.responses.create(
        model=model_id,
        reasoning={"effort": "low"},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "invoice_schema",
                "schema": INVOICE_SCHEMA,
                "strict": True
            }
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *messages
        ]
    )
    try:
        data = resp.output_parsed
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    # Fallback to raw JSON in text
    try:
        parts = resp.output[0].content
        if parts and hasattr(parts[0], "text"):
            import json
            return json.loads(parts[0].text)
    except Exception as e:
        raise RuntimeError(f"لم نحصل على مخرجات منظّمة: {e}")

def parse_file(client: OpenAI, file) -> Dict[str, Any]:
    name = file.name
    suffix = os.path.splitext(name)[1].lower()
    model_id = st.session_state.get("model_id", MODEL_DEFAULT)

    # PDFs: convert to images (vision) else extract text
    if suffix == ".pdf":
        file_bytes = file.read()
        images = pdf_to_images(file_bytes, dpi=220)
        if images:
            # Use first page only (MVP).
            msg = build_vision_message_for_image(images[0])
            return call_gpt_parse(client, [msg], model_id)
        else:
            # Try text extraction
            text = ""
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
                for p in reader.pages:
                    text += (p.extract_text() or "") + "\n"
            except Exception:
                pass
            if text.strip():
                msg = build_text_message_for_pdf_text(text)
                return call_gpt_parse(client, [msg], model_id)
            # If completely empty, return blanks
            return {"product_name": "", "date": "", "quantity": ""}
    else:
        # images
        img = Image.open(file).convert("RGB")
        msg = build_vision_message_for_image(img)
        return call_gpt_parse(client, [msg], model_id)

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🧾", layout="wide")

# Subtle styling
st.markdown("""
<style>
:root {
  --card-bg: #ffffff;
}
.reportview-container .main .block-container{padding-top:2rem; padding-bottom:2rem;}
.stApp header {background: rgba(255,255,255,0);}
/* Card */
.card {
  background: var(--card-bg);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 18px;
  padding: 20px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.04);
}
.badge {
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.12);
  font-size: 0.78rem;
}
hr {border: none; border-top: 1px dashed rgba(0,0,0,0.1); margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"### {APP_TITLE}")
st.caption(APP_DESC)

with st.sidebar:
    st.markdown("#### الإعدادات")
    st.session_state["model_id"] = st.text_input("Model ID", value=MODEL_DEFAULT, help="اكتب موديل GPT-5 المتاح بحسابك (مثال: gpt-5, gpt-5-thinking)")
    st.write("—")
    st.markdown("**OpenAI API Key** يتم قراءته من Secrets أو متغيرات البيئة.")
    if "OPENAI_API_KEY" not in st.secrets and not os.getenv("OPENAI_API_KEY"):
        st.info("أضف المفتاح في **Settings → Secrets** (Streamlit Cloud) باسم OPENAI_API_KEY.")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### ارفع الفواتير")
files = st.file_uploader("يدعم: PDF, JPG, PNG, WEBP", type=["pdf","jpg","jpeg","png","webp"], accept_multiple_files=True)
st.markdown("</div>", unsafe_allow_html=True)

client = get_openai_client()

results: List[Dict[str, Any]] = []
if st.button("استخراج الحقول", disabled=not files or client is None):
    with st.spinner("جار استخراج البيانات..."):
        for f in files:
            try:
                data = parse_file(client, f)
                results.append({
                    "file": f.name,
                    "product_name": data.get("product_name",""),
                    "date": data.get("date",""),
                    "quantity": data.get("quantity","")
                })
            except Exception as e:
                results.append({
                    "file": f.name, "product_name":"", "date":"", "quantity":"", "error": str(e)
                })

if results:
    df = pd.DataFrame(results, columns=["file","product_name","date","quantity","error"])
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### النتائج")
    st.dataframe(df, use_container_width=True, height=320)
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("تنزيل CSV", data=csv, file_name="invoices_extracted.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("ملاحظة: للـPDF المصوّرة، نحول الصفحة إلى صورة داخليًا (PyMuPDF). إذا فشل الاستخراج، سيظهر السطر مع خانة خطأ.")
