import argparse
from multiprocessing import cpu_count, freeze_support, get_context
from pathlib import Path

# try:
import bpy  # pylint: disable=import-error
import bmesh  # pylint: disable=import-error
import addon_utils
# except ImportError as exc:  # pragma: no cover - provides a clear error upfront
#     raise SystemExit(
#         "无法导入 bpy。请在 Blender Python 环境或安装了 bpy 模块的解释器中运行此脚本。"
#     ) from exc


def ensure_obj_importer_enabled():
    try:
        addon_utils.enable("io_scene_obj")
    except Exception:
        # 如果加载失败，后续在调用 import_scene.obj 时会抛出更明确的异常
        pass


def clear_blender_data():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    data_blocks = (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.images,
    )
    for collection in data_blocks:
        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def detect_floating_geometry(obj_path):
    clear_blender_data()

    try:
        bpy.ops.import_scene.obj(  # type: ignore[attr-defined]
            filepath=str(obj_path),
            use_edges=True,
            use_split_objects=True,
            use_split_groups=False,
            axis_forward="-Z",
            axis_up="Y",
        )
    except Exception as exc:
        raise RuntimeError(f"导入 OBJ 失败: {obj_path}") from exc

    floating_points = 0
    floating_edges = 0

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        if mesh is None:
            continue

        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()

            for vert in bm.verts:
                if not vert.link_faces and not vert.link_edges:
                    floating_points += 1

            for edge in bm.edges:
                if not edge.link_faces:
                    floating_edges += 1
        finally:
            bm.free()

    return {
        "floating_points": floating_points,
        "floating_lines": floating_edges,
        "has_floating": bool(floating_points or floating_edges),
    }


def _detect_worker(obj_path):
    obj_path = Path(obj_path)
    ensure_obj_importer_enabled()
    try:
        res = detect_floating_geometry(obj_path)
        return {
            "path": obj_path,
            "has_floating": res["has_floating"],
            "floating_points": res["floating_points"],
            "floating_lines": res["floating_lines"],
            "error": None,
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "path": obj_path,
            "has_floating": False,
            "floating_points": 0,
            "floating_lines": 0,
            "error": str(exc),
        }


def find_obj_files(root_dir):
    return sorted(path for path in Path(root_dir).rglob("*.obj") if path.is_file())


def write_report(output_path, summary_lines):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(summary_lines), encoding="utf-8")


def scan_directory(input_dir, output_path, processes=None):
    obj_files = find_obj_files(input_dir)
    total = len(obj_files)
    if total == 0:
        print(f"在 {input_dir} 未找到任何 OBJ 文件。")
        write_report(Path(output_path), ["未找到 OBJ 文件。"])
        return

    worker_count = processes or cpu_count()
    floating_results = []
    errors = []

    ctx = get_context("spawn")
    with ctx.Pool(processes=worker_count) as pool:
        for result in pool.imap_unordered(_detect_worker, [str(p) for p in obj_files]):
            if result["error"]:
                errors.append(result)
                continue
            if result["has_floating"]:
                floating_results.append(result)

    floating_results.sort(key=lambda item: str(item["path"]))
    lines = [
        f"扫描目录: {input_dir}",
        f"检测 OBJ 数量: {total}",
        f"存在漂浮几何的 OBJ 数量: {len(floating_results)}",
        "",
        "存在漂浮几何的文件列表:",
    ]

    if floating_results:
        for item in floating_results:
            lines.append(
                f"{item['path']}: 漂浮点={item['floating_points']} 漂浮线={item['floating_lines']}"
            )
    else:
        lines.append("无")

    if errors:
        lines.append("")
        lines.append("解析失败的文件:")
        for item in errors:
            lines.append(f"{item['path']}: {item['error']}")

    write_report(Path(output_path), lines)

    print(f"检测完成，结果写入: {output_path}")
    print(f"共检测 OBJ 文件数量: {total}")
    print(f"发现漂浮几何的 OBJ 数量: {len(floating_results)}")
    if errors:
        print(f"有 {len(errors)} 个文件处理失败，详细信息见输出文件。")


def parse_arguments():
    parser = argparse.ArgumentParser(description="使用 Blender 检测 OBJ 文件中的漂浮几何。")
    parser.add_argument("input_dir", type=Path, help="包含 OBJ 子文件夹的根目录")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("floating_report.txt"),
        help="结果输出的 txt 路径，默认写在当前目录",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=None,
        help="并行进程数，默认使用 CPU 核心数",
    )

    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"输入路径 {args.input_dir} 不是有效的目录。")

    return args


def main():
    freeze_support()
    ensure_obj_importer_enabled()
    args = parse_arguments()
    scan_directory(args.input_dir, args.output, args.processes)


if __name__ == "__main__":
    main()
