#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import signal
from iros.paths import OUTPUT_ROOT

PORTS = [7860, 7861, 7862, 7863, 7864]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_SCRIPT = os.path.join(SCRIPT_DIR, 'sam2_labeling_ui.py')
LOG_DIR = OUTPUT_ROOT / "logs" / "annotation"

processes = []

def kill_existing_processes():

    print("检查并清理占用端口的进程...")
    for port in PORTS:
        try:

            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            print(f"  已终止端口 {port} 上的进程 PID={pid}")
                        except:
                            pass
        except:
            pass
    time.sleep(1)

def launch_instance(port):

    print(f"启动端口 {port} 上的实例...")
    env = os.environ.copy()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = str(LOG_DIR / f'gradio_ui_{port}.log')
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            [sys.executable, APP_SCRIPT, '--port', str(port)],
            cwd=SCRIPT_DIR,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT
        )
    processes.append((port, process, log_file))
    print(f"  ✓ 端口 {port} 已启动 (PID={process.pid}, 日志: {log_file})")
    return process

def main():
    print("=" * 60)
    print("启动5个Gradio UI实例")
    print("=" * 60)

    kill_existing_processes()

    print("\n启动实例...")
    for port in PORTS:
        launch_instance(port)
        time.sleep(3)                     

    print("\n" + "=" * 60)
    print("所有实例已启动！")
    print("=" * 60)
    print("\n访问地址:")
    for port in PORTS:
        print(f"  - http://localhost:{port}")
    print("\nSSH端口转发 (在你的本地电脑上运行):")
    for i, port in enumerate(PORTS, 1):
        local_port = 7860 + i - 1
        print(f"  ssh -L {local_port}:localhost:{port} nuounuou@172.26.211.82")
    print("\n日志文件:")
    for port, proc, log_file in processes:
        print(f"  - 端口 {port}: {log_file}")
    print("\n查看日志:")
    for port, proc, log_file in processes:
        print(f"  tail -f {log_file}")
    print("\n按 Ctrl+C 停止所有实例")
    print("=" * 60)

    try:

        while True:
            time.sleep(1)

            for port, proc, log_file in processes:
                if proc.poll() is not None:
                    print(f"\n⚠️  端口 {port} 上的实例已退出 (返回码: {proc.returncode})")
                    print(f"   查看日志: tail -f {log_file}")
    except KeyboardInterrupt:
        print("\n\n正在停止所有实例...")
        for port, proc, log_file in processes:
            try:
                proc.terminate()
                print(f"  已终止端口 {port} 上的进程")
            except:
                pass

        time.sleep(2)

        for port, proc, log_file in processes:
            if proc.poll() is None:
                try:
                    proc.kill()
                    print(f"  已强制终止端口 {port} 上的进程")
                except:
                    pass

        print("\n所有实例已停止")

if __name__ == '__main__':
    main()
