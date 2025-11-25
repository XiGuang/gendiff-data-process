import argparse
import math
import os
import sys
import tempfile
import subprocess
from typing import List


def chunk_list(lst: List[str], n: int) -> List[List[str]]:
    if n <= 1:
        return [lst]
    k, m = divmod(len(lst), n)
    chunks = [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
    # remove empty chunks
    return [c for c in chunks if c]


def find_subdirs_with_yaml(parent: str) -> List[str]:
    names = []
    for d in os.listdir(parent):
        p = os.path.join(parent, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'data.yaml')):
            names.append(d)
    names.sort()
    return names


def build_child_cmd(python_exec: str, script_path: str, args, subdirs_file: str, inner_workers: int) -> List[str]:
    cmd = [
        python_exec,'-m',
        script_path,
        "--folder", args.folder,
        "--building_model_folder", args.building_model_folder,
        # "--ground_model_folder", args.ground_model_folder,
        "--sample_points_num", str(args.sample_points_num),
        "--workers", str(inner_workers),
        "--subdirs_file", subdirs_file,
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Split YAML subfolders across multiple processes to accelerate processing.")
    parser.add_argument("--folder", type=str, required=True, help="包含多个 data.yaml 子目录的父路径")
    parser.add_argument("--building_model_folder", type=str, required=True, help="建筑模型OBJ根目录")
    # parser.add_argument("--ground_model_folder", type=str, required=True, help="地面模型OBJ根目录")
    parser.add_argument("--sample_points_num", type=int, default=65536, help="每个模型采样的点数")
    parser.add_argument("--procs", type=int, default=2, help="并行的外层进程数（即同时启动的实例数）")
    parser.add_argument("--inner_workers", type=int, default=0, help="传递给子进程的 --workers（<=0 自动=CPU核数/进程数）")
    parser.add_argument("--python_exec", type=str, default=sys.executable, help="用于运行子任务的Python可执行文件；如需使用blender嵌入python，请填写其python路径或包装脚本")
    parser.add_argument("--script", type=str, default="process_change_model.py", help="子进程运行的脚本路径")
    parser.add_argument("--dry_run", action="store_true", help="仅打印将要执行的命令，不实际运行")

    args = parser.parse_args()

    all_names = find_subdirs_with_yaml(args.folder)
    if not all_names:
        print("未发现任何包含 data.yaml 的子目录，退出。")
        return 0

    procs = max(1, args.procs)
    chunks = chunk_list(all_names, procs)

    cpus = os.cpu_count() or 1
    if args.inner_workers and args.inner_workers > 0:
        inner_workers = args.inner_workers
    else:
        inner_workers = max(1, cpus // procs)

    processes = []
    temp_files = []

    for i, sublist in enumerate(chunks):
        # 写临时文件给子进程
        tf = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=f"_subdirs_{i}.txt")
        for name in sublist:
            tf.write(name + "\n")
        tf.flush()
        tf.close()
        temp_files.append(tf.name)

        cmd = build_child_cmd(args.python_exec, args.script, args, tf.name, inner_workers)
        print(f"[启动{i+1}/{len(chunks)}] 子目录数量={len(sublist)} -> {cmd}")
        if args.dry_run:
            continue
        p = subprocess.Popen(cmd)
        processes.append(p)

    if args.dry_run:
        print("dry_run 模式，未启动实际进程")
        return 0

    # 等待所有子进程完成
    rc_total = 0
    for p in processes:
        rc = p.wait()
        rc_total |= (rc != 0)

    # 清理临时文件
    for f in temp_files:
        try:
            os.unlink(f)
        except OSError:
            pass

    if rc_total:
        print("部分子进程返回非零退出码，请检查日志")
        return 1

    print("全部子进程完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
