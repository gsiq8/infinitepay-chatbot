{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 \
# setup.py\
from setuptools import setup, find_packages\
\
setup(\
    name="infinitepay-chatbot",\
    version="1.0.0",\
    description="AI-powered customer support chatbot for InfinitePay using RAG",\
    author="Your Name",\
    author_email="your.email@example.com",\
    packages=find_packages(),\
    install_requires=[\
        "requests>=2.31.0",\
        "beautifulsoup4>=4.12.2",\
        "lxml>=4.9.3",\
        "sentence-transformers>=2.2.2",\
        "scikit-learn>=1.3.0",\
        "numpy>=1.24.3",\
        "pandas>=2.0.3",\
        "fastapi>=0.103.1",\
        "uvicorn>=0.23.2",\
        "python-multipart>=0.0.6",\
        "python-dotenv>=1.0.0",\
        "pydantic>=2.3.0",\
        "aiofiles>=23.2.1",\
    ],\
    extras_require=\{\
        "dev": [\
            "pytest>=7.4.2",\
            "pytest-asyncio>=0.21.1",\
        ]\
    \},\
    python_requires=">=3.8",\
    classifiers=[\
        "Development Status :: 4 - Beta",\
        "Intended Audience :: Developers",\
        "License :: OSI Approved :: MIT License",\
        "Programming Language :: Python :: 3.8",\
        "Programming Language :: Python :: 3.9",\
        "Programming Language :: Python :: 3.10",\
        "Programming Language :: Python :: 3.11",\
    ],\
)\
}