# 🚀 Streamlit App Deployment Guide

This guide will help you deploy your TextRank + SimpleGNN Streamlit app to various platforms.

## 📋 Prerequisites

- ✅ Streamlit app is working locally
- ✅ All dependencies are installed
- ✅ Code is ready for deployment

## 🌐 Deployment Options

### 1. Streamlit Cloud (Recommended - Free)

**Step 1: Prepare Your Repository**
```bash
# Create a new GitHub repository
git init
git add .
git commit -m "Initial commit: TextRank + SimpleGNN Streamlit App"
git branch -M main
git remote add origin https://github.com/yourusername/textrank-gnn-summarizer.git
git push -u origin main
```

**Step 2: Deploy to Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `streamlit_app.py`
6. Click "Deploy"

**Step 3: Configure Environment**
- **Python version**: 3.9
- **Requirements file**: `streamlit_requirements.txt`
- **App URL**: Will be provided by Streamlit Cloud

### 2. Heroku Deployment

**Step 1: Create Heroku Files**

Create `Procfile`:
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

Create `runtime.txt`:
```
python-3.9.18
```

**Step 2: Deploy to Heroku**
```bash
# Install Heroku CLI
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-app-name

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Open app
heroku open
```

### 3. Docker Deployment

**Step 1: Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY streamlit_requirements.txt .
RUN pip install -r streamlit_requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the application
CMD ["streamlit", "run", "streamlit_app.py"]
```

**Step 2: Build and Run Docker**
```bash
# Build image
docker build -t textrank-gnn-app .

# Run container
docker run -p 8501:8501 textrank-gnn-app

# Access at http://localhost:8501
```

### 4. Google Cloud Run

**Step 1: Create Cloud Run Files**

Create `Dockerfile` (same as above)

Create `.dockerignore`:
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
```

**Step 2: Deploy to Cloud Run**
```bash
# Install Google Cloud CLI
# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud run deploy textrank-gnn-app \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501
```

### 5. AWS Elastic Beanstalk

**Step 1: Create EB Files**

Create `.ebextensions/python.config`:
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: streamlit_app.py
  aws:elasticbeanstalk:application:environment:
    PYTHONPATH: "/var/app/current:$PYTHONPATH"
```

**Step 2: Deploy to EB**
```bash
# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init -p python-3.9 textrank-gnn-app

# Create environment
eb create textrank-gnn-env

# Deploy
eb deploy
```

## 🔧 Configuration Files

### Streamlit Configuration (`.streamlit/config.toml`)
```toml
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Requirements File (`streamlit_requirements.txt`)
```
streamlit>=1.28.0
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
networkx>=2.6.0
scikit-learn>=1.0.0
rouge-score>=0.1.2
nltk>=3.6.0
tqdm>=4.62.0
```

## 📊 Performance Optimization

### For Production Deployment

1. **Reduce Model Complexity**
   ```python
   # In streamlit_app.py, reduce epochs for faster processing
   epochs = st.sidebar.slider("Training Epochs", 5, 15, 8)
   ```

2. **Add Caching**
   ```python
   @st.cache_data
   def load_model():
       # Load pre-trained model
       pass
   ```

3. **Memory Management**
   ```python
   # Clear cache periodically
   if st.button("Clear Cache"):
       st.cache_data.clear()
   ```

4. **Error Handling**
   ```python
   try:
       # Your code here
   except Exception as e:
       st.error(f"An error occurred: {e}")
   ```

## 🔒 Security Considerations

1. **Environment Variables**
   ```bash
   # Set sensitive data as environment variables
   export API_KEY=your_api_key
   ```

2. **Input Validation**
   ```python
   # Validate user input
   if len(input_text) > 10000:
       st.error("Text too long. Please use shorter text.")
   ```

3. **Rate Limiting**
   ```python
   # Add rate limiting for API calls
   import time
   time.sleep(1)  # Add delay between requests
   ```

## 📈 Monitoring and Analytics

### Add Analytics
```python
# Track usage
import streamlit as st

def track_usage():
    if 'usage_count' not in st.session_state:
        st.session_state.usage_count = 0
    st.session_state.usage_count += 1
```

### Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_summarization(text_length, model_type):
    logger.info(f"Summarization: {text_length} chars, {model_type}")
```

## 🚀 Quick Deploy Commands

### Streamlit Cloud (Easiest)
```bash
# 1. Push to GitHub
git push origin main

# 2. Deploy via Streamlit Cloud web interface
# Go to share.streamlit.io
```

### Local Testing
```bash
# Test locally first
streamlit run streamlit_app.py

# Test with different port
streamlit run streamlit_app.py --server.port=8502
```

### Docker Quick Deploy
```bash
# Build and run in one command
docker build -t textrank-app . && docker run -p 8501:8501 textrank-app
```

## 📞 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find and kill process
   lsof -ti:8501 | xargs kill -9
   ```

2. **Memory Issues**
   ```bash
   # Increase memory limit
   export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
   ```

3. **Dependency Conflicts**
   ```bash
   # Create fresh environment
   python -m venv fresh_env
   source fresh_env/bin/activate
   pip install -r streamlit_requirements.txt
   ```

4. **NLTK Data Missing**
   ```bash
   # Download NLTK data
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## 🎯 Success Checklist

- [ ] App runs locally without errors
- [ ] All dependencies are in requirements.txt
- [ ] Configuration files are properly set
- [ ] Environment variables are configured
- [ ] Error handling is implemented
- [ ] Performance is optimized
- [ ] Security measures are in place
- [ ] Monitoring is set up
- [ ] Documentation is updated

## 🌟 Next Steps

1. **Deploy to Streamlit Cloud** (Recommended for beginners)
2. **Set up custom domain** (Optional)
3. **Add analytics and monitoring**
4. **Implement user feedback system**
5. **Scale based on usage**

---

**🚀 Your TextRank + SimpleGNN Streamlit App is ready for deployment!** 

Choose the deployment option that best fits your needs and follow the steps above. Streamlit Cloud is recommended for quick and easy deployment with zero configuration required. 