#!/bin/bash

# Charity Client Finder - Development Workflow Helper
# Usage: ./dev-workflow.sh [command]

case "$1" in
    "start-local")
        echo "🚀 Starting local development server..."
        echo "📍 Connecting to: localhost:6333 (local Qdrant)"
        echo "🌐 App URL: http://localhost:8501"
        pkill -f streamlit 2>/dev/null
        sleep 2
        env -i PATH="/opt/anaconda3/bin:/opt/anaconda3/condabin:/usr/bin:/bin" bash -c "
            cd /Users/onursahil/Documents/Developer/charity-client-finder && 
            source /opt/anaconda3/etc/profile.d/conda.sh && 
            conda activate search_env && 
            streamlit run charity_search_app.py
        " &
        echo "✅ Local server starting... Visit http://localhost:8501"
        ;;
        
    "stop-local")
        echo "🛑 Stopping local development server..."
        pkill -f streamlit
        echo "✅ Local server stopped"
        ;;
        
    "deploy")
        echo "🚀 Deploying to production..."
        echo "📝 Adding changes to git..."
        git add .
        read -p "💬 Enter commit message: " commit_msg
        git commit -m "$commit_msg"
        echo "📤 Pushing to GitHub..."
        git push origin main
        echo "✅ Deployed! Streamlit Cloud will auto-update in ~2 minutes"
        echo "🌐 Check your deployed app for updates"
        ;;
        
    "status")
        echo "📊 Development Environment Status:"
        echo ""
        echo "🔹 Local Streamlit:"
        if pgrep -f "streamlit run charity_search_app.py" > /dev/null; then
            echo "   ✅ Running at http://localhost:8501"
        else
            echo "   ❌ Not running"
        fi
        
        echo ""
        echo "🔹 Local Qdrant:"
        if curl -s http://localhost:6333/health > /dev/null; then
            count=$(curl -s http://localhost:6333/collections/charity_commission | jq -r '.result.points_count // 0')
            echo "   ✅ Running at localhost:6333 ($count records)"
        else
            echo "   ❌ Not running"
        fi
        
        echo ""
        echo "🔹 Git Status:"
        if git status --porcelain | grep -q .; then
            echo "   📝 Uncommitted changes:"
            git status --short
        else
            echo "   ✅ All changes committed"
        fi
        ;;
        
    "help"|*)
        echo "🎯 Charity Client Finder - Development Workflow"
        echo ""
        echo "Commands:"
        echo "  start-local  🚀 Start local development server (localhost:6333)"
        echo "  stop-local   🛑 Stop local development server"
        echo "  deploy       📤 Deploy changes to production (GitHub + Streamlit Cloud)"
        echo "  status       📊 Show status of local environment"
        echo "  help         ❓ Show this help message"
        echo ""
        echo "Typical workflow:"
        echo "  1. ./dev-workflow.sh start-local     # Start local testing"
        echo "  2. Make your code changes"
        echo "  3. Test at http://localhost:8501"
        echo "  4. ./dev-workflow.sh deploy          # Deploy when ready"
        ;;
esac 