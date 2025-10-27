#!/bin/bash
# microvision cleanup script for maiden runs

echo "🧹 cleaning microvision for maiden run..."

# clean python cache
echo "cleaning python cache..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# clean jupyter cache
echo "cleaning jupyter cache..."
rm -rf .ipynb_checkpoints/ 2>/dev/null || true
jupyter lab clean 2>/dev/null || true

# clean experiment data
echo "cleaning experiment data..."
rm -rf experiments/logs/* 2>/dev/null || true
rm -rf experiments/checkpoints/* 2>/dev/null || true
rm -rf experiments/models/* 2>/dev/null || true

# clean dataset cache (optional - uncomment if you want to re-download)
# echo "cleaning dataset cache..."
# rm -rf data/cifar-10-batches-py/ 2>/dev/null || true
# rm -f data/cifar-10-python.tar.gz 2>/dev/null || true

# clean system caches
echo "cleaning system caches..."
rm -rf ~/.cache/torch/ 2>/dev/null || true
rm -rf ~/.cache/matplotlib/ 2>/dev/null || true

# clean tensorboard logs
echo "cleaning tensorboard logs..."
rm -rf runs/ 2>/dev/null || true

# clean any temporary files
echo "cleaning temporary files..."
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.log" -delete 2>/dev/null || true

echo " cleanup complete! ready for maiden run."
echo ""
echo " to start fresh:"
echo "1. source mv-venv/bin/activate"
echo "2. jupyter lab"
echo "3. run notebooks 01-08 in order"
