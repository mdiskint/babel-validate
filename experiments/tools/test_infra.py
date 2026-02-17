import asyncio
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
semaphore = asyncio.Semaphore(2)

async def test_call(i):
    async with semaphore:
        print(f"[{i}] Acquiring semaphore...")
        res = await client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": "say hi"}],
            max_tokens=5
        )
        print(f"[{i}] Done: {res.choices[0].message.content}")

async def main():
    tasks = [test_call(i) for i in range(5)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
