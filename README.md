
# فاتورتي — استخراج (الاسم، التاريخ، الكمية)

واجهة ويب بسيطة بـ **Streamlit** لاستخراج 3 حقول من الفواتير (PDF/صور) باستخدام **OpenAI Responses API (GPT-5)** مع مخرجات منظّمة.

## الميزات
- رفع عدة ملفات دفعة واحدة (PDF/صور).
- تحويل PDF المصوّرة إلى صور داخليًا عبر PyMuPDF (بدون Poppler).
- استخدام JSON Schema لضمان **product_name / date / quantity** فقط.
- عرض النتائج في جدول + تنزيل CSV.

## التشغيل محليًا
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...  # أو ضعها في .streamlit/secrets.toml
streamlit run app.py
```

## النشر على Streamlit Cloud
1. ارفع هذا المجلد إلى GitHub كـ Repo.
2. من Streamlit Cloud، اختر "New app" وحدد الـRepo و الفرع وملف `app.py`.
3. في "Settings → Secrets"، أضف:
```
OPENAI_API_KEY="sk-..."
```
4. شغل التطبيق، بيطلع لك رابط مباشر.

> ملاحظة: لتغيير الموديل استخدم مربع النص في الشريط الجانبي (مثل `gpt-5` أو `gpt-5-thinking`).

## ملاحظات
- لو بعض الفواتير بدون نص واضح أو تصوير رديء، قد تحتاج إعادة الرفع بصور أوضح.
- إذا أردت استخراج **كل البنود** بدل بند واحد، نقدر نعدّل الـSchema ونستخرج قائمة عناصر.
