import os
import subprocess
from pathlib import Path
from iros.paths import FRAME_ROOT, VIDEO_ROOT

def extract_frames_from_video(video_path, output_dir, frame_interval=2):

    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, video_name)
    os.makedirs(output_path, exist_ok=True)

    print(f"处理视频: {video_path}")
    print(f"输出目录: {output_path}")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select='not(mod(n,{frame_interval}))'",
        "-vsync", "0",
        "-q:v", "2",
        "-start_number", "0",
        os.path.join(output_path, "%05d.jpg")
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print(f"[OK] 完成: {video_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 错误: {video_name}")
        print(f"  错误信息: {e.stderr}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract image frames from every video in a directory")
    parser.add_argument("--input", type=Path, default=VIDEO_ROOT, help="Input video directory")
    parser.add_argument("--output", type=Path, default=FRAME_ROOT, help="Output frame directory")
    parser.add_argument("--interval", type=int, default=2, help="Keep every Nth frame (paper: every other frame, N=2)")
    args = parser.parse_args()
    videos_dir = str(args.input)
    output_base_dir = str(args.output)
    frame_interval = args.interval

    if frame_interval < 1:
        parser.error("--interval must be at least 1")
    if not os.path.isdir(videos_dir):
        parser.error(f"input directory does not exist: {videos_dir}")

    os.makedirs(output_base_dir, exist_ok=True)

    video_extensions = [".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV"]
    video_files = []

    for file in os.listdir(videos_dir):
        if any(file.endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(videos_dir, file))

    if len(video_files) == 0:
        print(f"在 {videos_dir} 目录下未找到视频文件")
    else:
        print(f"找到 {len(video_files)} 个视频文件")
        print(f"每 {frame_interval} 帧抽取一张\n")

        success_count = 0
        for video_path in video_files:
            if extract_frames_from_video(video_path, output_base_dir, frame_interval):
                success_count += 1

        print(f"\n处理完成！成功: {success_count}/{len(video_files)}")
