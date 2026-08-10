import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "gpt-4o")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")

# For mechanical, non-judgment retrieval steps -- query expansion (paraphrasing a question) and
# re-ranking (sorting passages by relevance) -- not the nuanced reasoning GENERATOR_MODEL is
# reserved for (drafting the actual answer, weighing a source conflict). Kept as its own
# variable, not reused from JUDGE_MODEL, so the two can be tuned independently even though they
# default to the same cheaper model today.
MECHANICAL_MODEL = os.environ.get("MECHANICAL_MODEL", "gpt-4o-mini")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://immigration_navigator:local_dev_only@localhost:5432/immigration_navigator"
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# Found live: a real user request and this session's own background eval run both landed on
# the account's shared tokens-per-minute cap at the same time, and the SDK's default
# max_retries=2 wasn't enough headroom -- a 429 surfaced as a raw crash instead of quietly
# retrying. The SDK already backs off between attempts and honors the API's own retry-after
# hint, so raising max_retries here (once, for every agent) is enough on its own.
_OPENAI_MAX_RETRIES = 5


def get_openai_client() -> OpenAI:
    """Every agent constructs its OpenAI client through here so rate-limit resilience is
    configured once, not copy-pasted at each of the six call sites that used to construct
    `OpenAI(api_key=OPENAI_API_KEY)` directly."""
    return OpenAI(api_key=OPENAI_API_KEY, max_retries=_OPENAI_MAX_RETRIES)
