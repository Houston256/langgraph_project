"""python -m assistant.migrations: migrates the database and exits."""

import asyncio
import logging

from assistant.db import migrate_db

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    asyncio.run(migrate_db())
