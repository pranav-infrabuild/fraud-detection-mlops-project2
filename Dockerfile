# Base image - Python 3.11 ka chhota version
FROM python:3.11-slim

# Box ke andar working folder
WORKDIR /app

# Pehle requirements copy karo aur install karo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ab baaki code aur model copy karo
COPY src/ ./src/
COPY models/ ./models/

# API is port pe chalega
EXPOSE 5001

# Box start hote hi yeh command chalegi
CMD ["python", "src/predict.py"]
