import importlib.metadata
packages = [
    "ipykernel",
    "langchain-community",
    "langchain-core",
    "langchain-google-genai",
    "langchain-groq",
    "langchain-openai",
    "langgraph",
    "tavily-python",
    "wikipedia",
]
for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} (not installed)")