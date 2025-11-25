#!/usr/bin/env python3
import os
import sys
import argparse
import multiprocessing as mp
import traceback
import platform

# 说明：pip 安装的 bpy 在 Linux 下使用 fork 方式通常更可靠；spawn 方式会重新执行入口脚本，
# 可能导致无法定位内部的 _bpy 扩展从而出现 ModuleNotFoundError。故默认在 Linux 上强制使用 fork。
# 在非 Linux 平台若需要并行，建议改用外层多进程用 subprocess 调用，或保持单进程。

import bpy

def parse_args():
    # Blender 在 '--' 之前的是自己的参数，我们只取 '--' 之后的
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(description="Batch apply Smooth + Decimate modifiers to OBJ files.")
    parser.add_argument("--input_dir", "-i", required=True, help="Input directory with .obj files")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory for processed .obj files")
    parser.add_argument("--smooth_iterations", "--laplacian_iterations", dest="smooth_iterations", type=int, default=1,
                        help="Iterations for Smooth modifier")
    parser.add_argument("--smooth_factor", "--laplacian_lambda", dest="smooth_factor", type=float, default=1.0,
                        help="Factor for Smooth modifier")
    parser.add_argument("--decimate_ratio", type=float, default=0.5, help="Ratio for Decimate modifier (0~1)")
    parser.add_argument("--suffix", type=str, default=None, help="Suffix for output file name")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel processes (>=1)")
    parser.add_argument("--chunk_size", type=int, default=1, help="How many files a worker processes sequentially (>=1)")
    args = parser.parse_args(argv)
    return args

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # 删除可能残留的网格数据等
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

def import_obj(obj_path):
    bpy.ops.wm.obj_import(filepath=obj_path)

def apply_modifiers_to_mesh_objects(smooth_iterations, smooth_factor, decimate_ratio):
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        print(f"Processing object: {obj.name}")

        # 普通平滑修改器
        smooth_mod = obj.modifiers.new(name="Smooth", type='SMOOTH')
        smooth_mod.iterations = smooth_iterations
        smooth_mod.factor = smooth_factor

        bpy.ops.object.modifier_apply(modifier=smooth_mod.name)
        print(f"Applied Smooth modifier to {obj.name}")

        # 精简修改器
        dec_mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
        dec_mod.ratio = decimate_ratio
        dec_mod.use_collapse_triangulate = True  # 按需要打开

        bpy.ops.object.modifier_apply(modifier=dec_mod.name)
        print(f"Applied Decimate modifier to {obj.name}")

        obj.select_set(False)

def export_obj(output_path):
    # 选中所有要导出的对象
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    bpy.ops.wm.obj_export(
        filepath=output_path,
        export_selected_objects=True,
        export_materials=False
    )

def process_one_file(in_path, out_path, smooth_iterations, smooth_factor, decimate_ratio):
    try:
        clean_scene()
        import_obj(in_path)
        apply_modifiers_to_mesh_objects(
            smooth_iterations=smooth_iterations,
            smooth_factor=smooth_factor,
            decimate_ratio=decimate_ratio,
        )
        export_obj(out_path)
        return (in_path, True, "")
    except Exception as e:
        tb = traceback.format_exc()
        return (in_path, False, f"{e}\n{tb}")

def worker_run(batch, params):
    results = []
    for in_path, out_path in batch:
        r = process_one_file(
            in_path=in_path,
            out_path=out_path,
            smooth_iterations=params['smooth_iterations'],
            smooth_factor=params['smooth_factor'],
            decimate_ratio=params['decimate_ratio'],
        )
        results.append(r)
    return results

def main():
    args = parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    obj_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".obj")]
    if not obj_files:
        print(f"No .obj files found in {input_dir}")
        return

    print(f"Found {len(obj_files)} .obj files in {input_dir}")

    # 构建任务列表 (in_path, out_path)
    tasks = []
    for fname in sorted(obj_files):
        in_path = os.path.join(input_dir, fname)
        base, ext = os.path.splitext(fname)
        out_name = base + (args.suffix or "") + ext
        out_path = os.path.join(output_dir, out_name)
        tasks.append((in_path, out_path))

    num_workers = max(1, args.num_workers)
    chunk_size = max(1, args.chunk_size)

    # 将任务按 chunk_size 分块
    batches = [tasks[i:i+chunk_size] for i in range(0, len(tasks), chunk_size)]
    print(f"Using {num_workers} workers, {len(batches)} batches, chunk_size={chunk_size}")

    params = dict(
        smooth_iterations=args.smooth_iterations,
        smooth_factor=args.smooth_factor,
        decimate_ratio=args.decimate_ratio,
    )

    if num_workers == 1:
        # 单进程串行执行，保持原先行为但走统一代码路径
        all_results = []
        for batch in batches:
            res = worker_run(batch, params)
            all_results.extend(res)
    else:
        # 多进程：Linux 下优先使用 fork，避免 spawn 重新加载导致 _bpy 丢失
        if platform.system().lower() == 'linux':
            try:
                mp.set_start_method('fork')
            except RuntimeError:
                pass
        else:
            print("Non-Linux platform detected; falling back to single-process due to bpy spawn limitations.")
            num_workers = 1
            all_results = []
            for batch in batches:
                res = worker_run(batch, params)
                all_results.extend(res)
            # 汇总并返回
            success = 0
            for idx, (in_path, ok, info) in enumerate(all_results, 1):
                out_path = [t[1] for t in tasks if t[0] == in_path][0]
                if ok:
                    print(f"[{idx}/{len(all_results)}] OK: {in_path} -> {out_path}")
                    success += 1
                else:
                    print(f"[{idx}/{len(all_results)}] FAIL: {in_path} -> {out_path}\n{info}")
            print(f"Done. Success {success}/{len(all_results)}; Failed {len(all_results)-success}")
            return
        with mp.Pool(processes=num_workers) as pool:
            all_results_nested = pool.starmap(worker_run, [(batch, params) for batch in batches])
        all_results = [item for sub in all_results_nested for item in sub]

    # 汇总输出
    success = 0
    for idx, (in_path, ok, info) in enumerate(all_results, 1):
        out_path = [t[1] for t in tasks if t[0] == in_path][0]
        if ok:
            print(f"[{idx}/{len(all_results)}] OK: {in_path} -> {out_path}")
            success += 1
        else:
            print(f"[{idx}/{len(all_results)}] FAIL: {in_path} -> {out_path}\n{info}")

    print(f"Done. Success {success}/{len(all_results)}; Failed {len(all_results)-success}")

if __name__ == "__main__":
    main()