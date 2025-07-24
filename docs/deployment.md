# Deployment Guide

## Quick Deployment Options

### 1. Streamlit Cloud (Free - Recommended for MVP)

1. **Push to GitHub**:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/charity-client-finder.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select repository: `charity-client-finder`
   - Set main file: `charity_search_app.py`
   - Deploy

3. **Configure Qdrant Cloud**:
   - Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
   - Create a cluster
   - Add environment variables in Streamlit Cloud:
     - `QDRANT_URL`: Your Qdrant cluster URL
     - `QDRANT_API_KEY`: Your Qdrant API key

### 2. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t charity-client-finder .
docker run -p 8501:8501 charity-client-finder
```

### 3. Cloud Providers

#### Google Cloud Run
```bash
gcloud run deploy charity-client-finder \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### AWS ECS with Fargate
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t charity-client-finder .
docker tag charity-client-finder:latest <account>.dkr.ecr.us-east-1.amazonaws.com/charity-client-finder:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/charity-client-finder:latest
```

#### Heroku
```bash
# Install Heroku CLI and login
heroku create charity-client-finder
git push heroku main
```

## Environment Configuration

### Required Environment Variables
```bash
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_api_key
ENVIRONMENT=production
```

### Optional Environment Variables
```bash
STREAMLIT_SERVER_PORT=8501
DEFAULT_SCORE_THRESHOLD=0.3
DEFAULT_FUZZY_THRESHOLD=70
LOG_LEVEL=INFO
```

## Production Checklist

- [ ] Set up Qdrant Cloud cluster
- [ ] Configure environment variables
- [ ] Set up monitoring/logging
- [ ] Configure SSL/HTTPS
- [ ] Set up domain name
- [ ] Configure backup strategy
- [ ] Set up CI/CD pipeline
- [ ] Add authentication (if required)
- [ ] Performance testing
- [ ] Security audit

## Monitoring & Maintenance

### Health Checks
- Application: `http://localhost:8501/_stcore/health`
- Qdrant: `http://localhost:6333/health`

### Logs
- Application logs: `/app/logs/`
- Streamlit logs: Check container logs
- Qdrant logs: Check Qdrant Cloud dashboard

### Performance Metrics
- Search response time: < 1 second
- Memory usage: < 2GB
- CPU usage: < 80%
- Qdrant cluster health: Monitor in cloud dashboard 