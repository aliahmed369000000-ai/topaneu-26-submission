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

# الكود الأساسي
COPY --chown=user:user main.py /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user model.py /opt/app/
COPY --chown=user:user vessel_preprocessing.py /opt/app/

# الأوزان (يجب أن تكون موجودة قبل البناء أو تُرفع كـ Model على GC)
# في GC تُستخرج الأوزان إلى /opt/ml/model تلقائيًا إذا رفعتها كـ Model tarball
COPY --chown=user:user models/ /opt/app/models/

ENTRYPOINT ["python", "main.py"]
