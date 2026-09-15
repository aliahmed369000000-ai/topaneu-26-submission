FROM --platform=linux/amd64 pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user main.py inference.py model.py vessel_preprocessing.py /opt/app/

# نسخة احتياطية اختيارية للأوزان داخل الصورة (إن وُجدت وقت البناء)
# على GC المصدر الأساسي: /opt/ml/model من Model tarball
COPY --chown=user:user models/ /opt/app/models/

ENTRYPOINT ["python", "main.py"]
