"""預覽模型的前置處理（Blender）：貼圖縮到 SIZE、拿掉法線貼圖，輸出未壓縮 GLB，之後交給 gltfpack 減面＋meshopt 壓縮。
用法：blender -b --factory-startup -P scripts/make-preview-src.py -- <完整未壓縮.glb> <輸出.glb> [SIZE=256]
"""
import sys
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
size = int(argv[2]) if len(argv) > 2 else 256
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
for mat in bpy.data.materials:
    if not mat.use_nodes:
        continue
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type == "NORMAL_MAP":  # 預覽不需要法線貼圖
            for l in list(n.outputs[0].links):
                nt.links.remove(l)
for img in bpy.data.images:
    if img.size[0] > size:
        img.scale(size, size)
        img.file_format = "JPEG"
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", export_yup=True, export_image_format="JPEG", export_jpeg_quality=80)
print("DONE")
