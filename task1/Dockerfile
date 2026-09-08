# ⚠️ هيكل عام غير مؤكد التطابق الكامل مع قالب Grand Challenge الرسمي.
# يجب استبداله بمحتوى templates/task1/Dockerfile الفعلي من
# github.com/Bangulli/TopAneu-26 بعد تحميل المستودع، لضمان التوافق مع
# main.py ونظام تشغيل GC (المسارات /input و /output الخاصة بهم).

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /opt/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py vessel_preprocessing.py ./
COPY models/ ./models/

# نقطة الدخول: يجب مطابقتها لما يتوقعه main.py الرسمي من GC
# (main.py غالبًا يُستورد من هنا ويستدعي predict() في inference.py)
ENTRYPOINT ["python", "-c", "import inference; print('inference module loaded OK')"]
