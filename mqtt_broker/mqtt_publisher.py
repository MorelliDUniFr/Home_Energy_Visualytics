import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime

broker = "localhost"  # if local mosquitto
port = 1883
topic = "tele/gPlugDI_1E533C/SENSOR"

client = mqtt.Client()
client.connect(broker, port, 60)
client.loop_start()

def generate_fake_payload():
    # Generate data similar to your IoT device JSON structure
    payload = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "z.Power": 100 + 10 * (time.time() % 10),  # some dummy changing value
        # Add other fields as needed matching expected structure
    }
    return json.dumps(payload)

try:
    while True:
        payload = generate_fake_payload()
        client.publish(topic, payload)
        print(f"Published: {payload}")
        time.sleep(10)  # publish every 10 seconds like your real device
except KeyboardInterrupt:
    client.loop_stop()
    client.disconnect()
