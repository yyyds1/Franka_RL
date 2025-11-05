#!/bin/bash

# 配置路径
ROOT_DIR="dataset/OakInk-v2/coacd_object_preview/align_ds"

# 颜色常量（可自定义）
COLOR_R="1.0"
COLOR_G="0.423529411765"
COLOR_B="0.0392156862745"
COLOR_A="1.0"
MATERIAL_NAME="obj_color"

# 递归查找所有 .obj 和 .ply 文件
find "$ROOT_DIR" \( -name "*.obj" -o -name "*.ply" \) -type f | while read -r mesh_file; do
    # 获取文件所在目录、文件名（无扩展名）、扩展名
    dir=$(dirname "$mesh_file")
    filename=$(basename "$mesh_file")
    name="${filename%.*}"
    ext="${filename##*.}"

    # 输出 URDF 文件路径
    urdf_file="$dir/$name.urdf"

    # mesh 标签中使用的相对文件名（保持原始格式）
    mesh_filename="$name.$ext"

    # 生成 URDF 内容
    cat > "$urdf_file" << EOF
<?xml version="1.0"?>
<robot name="design">
  <material name="$MATERIAL_NAME">
      <color rgba="$COLOR_R $COLOR_G $COLOR_B $COLOR_A"/>
  </material>
  <link name="base">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="$mesh_filename" scale="1 1 1"/>
      </geometry>
      <material name="$MATERIAL_NAME"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="$mesh_filename" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
EOF

    # 提示输出
    echo "✅ Generated: $urdf_file"

done

echo "🎉 All URDF files generated."