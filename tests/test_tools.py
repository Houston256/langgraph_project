from src.search.qdrant import query_product
import asyncio
import time


async def test_my_async_tool():
    # Define your parameters
    params = {
        "name": "chic lyžařské brýle",
        "description": "elegantní brýle na lyže vhodné na ski opening",
        "groups": ["BRÝLE"],
        "genders": ["Dámské"],
    }

    # Measure execution time
    start = time.perf_counter()
    result = await query_product.ainvoke(params)
    end = time.perf_counter()
    result = "\n".join([str(x["name"] + " " + x["uuid"]) for x in result])
    print(f"Execution time: {end - start:.4f} seconds")

    print("Params:")
    print(params)

    print("Result:")
    print(result)


asyncio.run(test_my_async_tool())
