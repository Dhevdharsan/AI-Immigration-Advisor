import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "gpt-4o")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://immigration_navigator:local_dev_only@localhost:5432/immigration_navigator"
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
