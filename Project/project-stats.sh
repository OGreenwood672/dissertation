#!/bin/bash

GRAND_TOTAL=0

count_section() {
    SECTION_NAME=$1
    FILE_LIST=$2

    echo "=========================================="
    echo "  $SECTION_NAME"
    echo "=========================================="

    if [ -z "$FILE_LIST" ]; then
        echo "No files found."
        SECTION_COUNT=0
    else
        echo "$FILE_LIST" | xargs wc -l | sort -nr
        SECTION_COUNT=$(echo "$FILE_LIST" | xargs wc -l | tail -n 1 | awk '{print $1}')
    fi

    echo "------------------------------------------"
    echo "  $SECTION_NAME Total: $SECTION_COUNT lines"
    echo "------------------------------------------"
    echo ""

    GRAND_TOTAL=$((GRAND_TOTAL + SECTION_COUNT))
}

CONFIG_FILES=$(find configs -type f \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.toml" \))
count_section "CONFIGS" "$CONFIG_FILES"

SIM_FILES=$(find environment environment-rs \
    -name "target" -prune -o \
    -type f \( -name "*.py" -o -name "*.rs" -o -name "*.toml" \) \
    -print)
count_section "SIMULATION (Rust)" "$SIM_FILES"

CTRL_FILES=$(find controller -name "*.py" -not -path "*/__pycache__/*")
count_section "CONTROLLER" "$CTRL_FILES"

FRONTEND_FILES=$(find frontend \
    -name "node_modules" -prune -o \
    -name "package-lock.json" -prune -o \
    -name "dist" -prune -o \
    -type f \( -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" -o -name "*.html" -o -name "*.css" -o -name "*.py" \) -print)
count_section "FRONTEND (Source Only)" "$FRONTEND_FILES"

LOG_FILES=$(find logging_utils -name "*.py" -not -path "*/__pycache__/*")
count_section "LOGGING UTILS" "$LOG_FILES"

echo "##########################################"
echo "  GRAND TOTAL LINES OF CODE: $GRAND_TOTAL"
echo "##########################################"