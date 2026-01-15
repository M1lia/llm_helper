import opentracing
import websocket
import os
from dotenv import load_dotenv

load_dotenv()

OPEN_API_KEY = os.environ.get('OPEN_API_KEY')


