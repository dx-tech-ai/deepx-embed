from setuptools import setup, find_packages

setup(
    name="deepx-embed",
    version="1.0.0",
    description="DeepX Embedding v1.0 — Vietnamese Legal Retrieval with Linear Attention",
    author="DX Tech Asia",
    author_email="contact@dxtech.jp",
    url="https://github.com/dx-tech-ai/deepx-embed",
    packages=find_packages() + ["modeling"],
    py_modules=["config"],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.30",
        "huggingface_hub",
        "numpy",
        "einops",
    ],
    extras_require={
        "train": ["bitsandbytes", "triton"],
        "fla": ["fla"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
