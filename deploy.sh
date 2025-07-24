#!/bin/bash

# Charity Client Finder - Quick Deployment Script
# This script helps you get started with the charity search application

set -e

echo "🚀 Charity Client Finder - Deployment Script"
echo "============================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Check if .env file exists, if not copy from example
if [ ! -f ".env" ]; then
    if [ -f "env.example" ]; then
        echo "📋 Creating .env file from example..."
        cp env.example .env
        echo "✅ .env file created. You can customize it if needed."
    else
        echo "⚠️  No env.example file found. Using default configuration."
    fi
fi

# Check if data files exist
if [ ! -f "data/backfill_data_ccn/ClientCCNs.xlsx" ]; then
    echo "❌ Client lookup data not found: data/backfill_data_ccn/ClientCCNs.xlsx"
    echo "   Please ensure your data files are in the correct location."
    exit 1
fi

if [ ! -f "data/charity_commission_data/publicextract.charity.json" ]; then
    echo "❌ Charity commission data not found: data/charity_commission_data/publicextract.charity.json"
    echo "   Please ensure your charity commission data files are in the correct location."
    exit 1
fi

echo "✅ Data files found"

# Build and start services
echo "🏗️  Building Docker containers..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Services are running!"
    echo ""
    echo "🎉 Deployment Complete!"
    echo "======================="
    echo "📱 Streamlit App: http://localhost:8501"
    echo "🔍 Qdrant Dashboard: http://localhost:6333/dashboard"
    echo ""
    echo "📋 Useful Commands:"
    echo "   Stop services:    docker-compose down"
    echo "   View logs:        docker-compose logs -f"
    echo "   Restart:          docker-compose restart"
    echo "   Build fresh:      docker-compose down && docker-compose build --no-cache && docker-compose up -d"
    echo ""
    echo "🔧 First time setup:"
    echo "   1. Visit http://localhost:8501"
    echo "   2. Go to 'Data Management' tab"
    echo "   3. Click 'Load and Index Data' to initialize the database"
    echo "   4. Once indexing is complete, use the 'Search' tab"
else
    echo "❌ Services failed to start. Check logs with: docker-compose logs"
    exit 1
fi
