"""Build the same native tools for Android or for host differential validation."""
import argparse, os, shutil, subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=['android', 'host'], default='android')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ndk', type=Path)
    ap.add_argument('--cmake')
    args = ap.parse_args()
    sdk = Path(os.environ.get('ANDROID_HOME', Path.home() / 'Library/Android/sdk'))

    def latest(root):
        candidates = sorted(root.iterdir(), key=lambda p: tuple((int(n) for n in p.name.split('.') if n.isdigit()))) if root.is_dir() else []
        if not candidates:
            raise FileNotFoundError(root)
        return candidates[-1]
    cmake = args.cmake or shutil.which('cmake') or str(latest(sdk / 'cmake') / 'bin/cmake')
    source = Path(__file__).resolve().parents[1]
    command = [cmake, '-S', str(source), '-B', str(args.output.resolve()), '-DCMAKE_BUILD_TYPE=Release']
    if args.target == 'android':
        ndk = args.ndk or (Path(os.environ['ANDROID_NDK_HOME']) if os.environ.get('ANDROID_NDK_HOME') else latest(sdk / 'ndk'))
        command += ['-DCMAKE_TOOLCHAIN_FILE=' + str(ndk / 'build/cmake/android.toolchain.cmake'), '-DANDROID_ABI=arm64-v8a', '-DANDROID_PLATFORM=android-26', '-DANDROID_STL=c++_static']
    subprocess.run(command, check=True)
    subprocess.run([cmake, '--build', str(args.output.resolve()), '-j', '4'], check=True)
if __name__ == '__main__':
    main()
