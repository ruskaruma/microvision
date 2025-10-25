#!/bin/bash
echo "microvision Cache Cleanup Script"
echo "=================================="
show_sizes() {
    echo "Current directory sizes:"
    echo "------------------------"
    if [ -d "src/__pycache__" ]; then
        echo "src/__pycache__: $(du -sh src/__pycache__ 2>/dev/null | cut -f1)"
    fi
    if [ -d "notebooks/.ipynb_checkpoints" ]; then
        echo "notebooks/.ipynb_checkpoints: $(du -sh notebooks/.ipynb_checkpoints 2>/dev/null | cut -f1)"
    fi
    if [ -d "notebooks/data" ]; then
        echo "notebooks/data: $(du -sh notebooks/data 2>/dev/null | cut -f1)"
    fi
    if [ -d "experiments" ]; then
        echo "experiments: $(du -sh experiments 2>/dev/null | cut -f1)"
    fi
    if [ -d "mv-venv" ]; then
        echo "mv-venv: $(du -sh mv-venv 2>/dev/null | cut -f1)"
    fi
    echo ""
}
echo "Before cleanup:"
show_sizes

echo "Cleaning Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -type f -delete 2>/dev/null || true
find . -name "*.pyo" -type f -delete 2>/dev/null || true
echo "Cleaning Jupyter checkpoints..."
rm -rf notebooks/.ipynb_checkpoints 2>/dev/null || true
echo "Cleaning experiment directories..."
rm -rf experiments/logs 2>/dev/null || true
rm -rf experiments/checkpoints 2>/dev/null || true
rm -rf experiments/models 2>/dev/null || true


echo "Cleaning temporary files..."
find . -name "*.tmp" -type f -delete 2>/dev/null || true
find . -name "*.log" -type f -delete 2>/dev/null || true
find . -name ".DS_Store" -type f -delete 2>/dev/null || true

echo "Cleaning virtual environment cache..."
if [ -d "mv-venv/lib/python3.12/site-packages" ]; then
    find mv-venv/lib/python3.12/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find mv-venv/lib/python3.12/site-packages -name "*.pyc" -type f -delete 2>/dev/null || true
fi

echo ""
echo "Cleanup completed!"
echo ""

echo "After cleanup:"
show_sizes

echo "Cache cleanup finished!"
echo ""
echo "What was cleaned:"
echo "  • Python __pycache__ directories"
echo "  • Jupyter notebook checkpoints"
echo "  • Experiment logs and checkpoints"
echo "  • Temporary files"
echo "  • Virtual environment cache files"
echo ""
echo "To clean downloaded data (CIFAR-10), uncomment the data cleanup line in this script"

##just run the fule as ./cleanup_cache.sh and it will clean the model storage cache and other things as well
## i guess i finally mastered the shell scripting 
