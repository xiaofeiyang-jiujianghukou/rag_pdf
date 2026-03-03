# fix_env.py
import subprocess
import sys

print("🔍 正在检测当前 Python 环境...")
print(f"Python 路径: {sys.executable}")
print(f"Python 版本: {sys.version}")

packages = ["langchain-alibaba", "dashscope", "langchain-openai", "langchain-community"]

for pkg in packages:
    print(f"\n📦 正在安装/升级 {pkg} ...")
    # 使用 sys.executable 确保包安装在当前运行脚本的 Python 环境中
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])

print("\n✅ 所有依赖安装完成！请重新运行你的主程序。")