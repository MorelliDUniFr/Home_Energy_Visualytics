# simulate_cache_writer.py
from diskcache import Cache
import os
import time
import random

cache = Cache(os.path.join('../data', "diskcache"))

print("Starting to write random live_power values every 10 seconds...")

try:
    while True:
        # Generate a random power value between 100 and 500 (adjust as needed)
        random_power = round(random.uniform(100, 500), 2)
        cache.set("live_power", random_power)
        print(f"Written live_power: {random_power}")
        time.sleep(10)
except KeyboardInterrupt:
    print("Stopped by user.")
