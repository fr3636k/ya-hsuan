import bpy, sys
src, dst = sys.argv[sys.argv.index("--")+1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
bpy.ops.export_scene.gltf(filepath=dst, export_format='GLB', export_yup=True,
  export_draco_mesh_compression_enable=True, export_draco_mesh_compression_level=7,
  export_draco_position_quantization=16, export_draco_normal_quantization=10, export_draco_texcoord_quantization=14,
  export_image_format='AUTO')
print("DONE")
