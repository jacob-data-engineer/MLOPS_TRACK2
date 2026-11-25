"""Independent development worker: no HTTP server or browser connection required."""
import asyncio
from .. import record_store
from . import preparation_jobs, ticker


async def preparation_loop():
    while True:
        try:
            await asyncio.to_thread(preparation_jobs.tick)
        except Exception as exc:
            print(f'Preparation worker: {exc}', flush=True)
        try:
            from . import training_variants
            await asyncio.to_thread(training_variants.tick)
        except Exception as exc:
            print(f'Training variants worker: {exc}', flush=True)
        await asyncio.sleep(3)


async def main():
    record_store.verify_ready()
    print('Independent SQL training worker started', flush=True)
    await asyncio.gather(ticker.queue_poll_loop(), preparation_loop())


if __name__ == '__main__':
    asyncio.run(main())
