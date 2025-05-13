import asyncio
import logging
from quart import Quart
from quart_cors import cors
from threading import Thread

from config import async_read_indexes, init_env_and_logging, server_settings
from agent_workflow_structured_answer import builder #initialize a static workflow
from routes import register_routes

app = Quart(__name__)
app = cors(app, allow_origin="*")  # Allow CORS from all origins

init_env_and_logging()

@app.before_serving
async def before_serving():
    # Ensure indexes are loaded before serving requests
    status, indexes_loaded = server_settings.get_status()
    if not indexes_loaded:
        print("Waiting for indexes to load...")
        await async_read_indexes()  # This will ensure indexes are loaded
        print("Indexes loaded.")

        # Now the status should be updated correctly
        status, indexes_loaded = server_settings.get_status()
        logging.info(f"Server status after loading indexes: {status}")
    else:
        logging.info("Indexes already loaded, server is ready.")
        
# Register routes here
register_routes(app)




if __name__ == "__main__":
    app.run()
