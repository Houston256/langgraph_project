import openai

from assistant.config import settings

# Provider answers worth another try: credit holds (402), rate limits, timeouts, 5xx.
RETRY_STATUS = {402, 408, 409, 429, 500, 502, 503, 504}


def retry_delay(error: BaseException, attempt: int) -> float | None:
    """Seconds to wait before retrying the run, or None if the error is final."""
    e: BaseException | None = error
    while e is not None:
        if isinstance(e, openai.APIStatusError) and e.status_code in RETRY_STATUS:
            header = e.response.headers.get("retry-after")
            try:
                wait = float(header) if header else 2.0**attempt
            except ValueError:
                wait = 2.0**attempt
            return min(wait, settings.run_retry_max_seconds)
        if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError)):
            return min(2.0**attempt, settings.run_retry_max_seconds)
        e = e.__cause__
    return None
