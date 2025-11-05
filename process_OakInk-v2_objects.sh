#!/bin/bash

# 配置参数
INPUT_ROOT="dataset/OakInk-v2/object_preview/align_ds"
OUTPUT_ROOT="dataset/OakInk-v2/coacd_object_preview/align_ds"
SCRIPT="scripts/coacd_process.py"

# 其他参数
MAX_HULL=32
SEED=1
MCTS_ITER=2000
MCTS_DEPTH=5
THRESHOLD=0.07

# 创建输出根目录
mkdir -p "$OUTPUT_ROOT"

# 查找所有 .obj 和 .ply 文件
find "$INPUT_ROOT" \( -name "*.obj" -o -name "*.ply" \) -type f | while read input_file; do
    # 获取相对于 INPUT_ROOT 的相对路径（包含文件名）
    rel_path="${input_file#$INPUT_ROOT/}"
    
    # 构造输出文件路径
    output_file="$OUTPUT_ROOT/$rel_path"
    
    # 创建输出目录
    output_dir=$(dirname "$output_file")
    mkdir -p "$output_dir"
    
    # 打印正在处理的文件
    echo "Processing: $input_file"
    echo "      ->  $output_file"
    
    # 运行 Python 脚本
    python "$SCRIPT" \
        -i "$input_file" \
        -o "$output_file" \
        --max-convex-hull "$MAX_HULL" \
        --seed "$SEED" \
        -mi "$MCTS_ITER" \
        -md "$MCTS_DEPTH" \
        -t "$THRESHOLD"
    
    # 检查执行是否成功
    if [ $? -eq 0 ]; then
        echo "✅ Success: $rel_path"
    else
        echo "❌ Failed: $rel_path"
    fi
    echo "---"
done

echo "All files processed."