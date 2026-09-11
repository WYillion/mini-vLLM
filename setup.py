from setuptools import setup, find_packages

setup(
    name="mini-vllm",
    version="0.1.0",
    description="A simplified vLLM inference framework implementation",
    author="Wang Yilin",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "scipy>=1.11.0",
        "accelerate>=0.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mini-vllm=mini_vllm.api.server:main",
        ],
    },
)
