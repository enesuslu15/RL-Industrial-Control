import asyncio
import logging
from asyncua import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OPCUA_Client")

class OPCUAClient:
    def __init__(self, url: str):
        self.url = url
        self.client = Client(url=self.url)
        self.nodes = {}

    async def connect(self):
        try:
            await self.client.connect()
            logger.info(f"Connected to OPC UA Server at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to {self.url}: {e}")
            raise e

    async def disconnect(self):
        await self.client.disconnect()
        logger.info("Disconnected from OPC UA Server")

    async def register_node(self, name: str, node_id: str):
        """Registers a node so we can read/write to it easily."""
        try:
            node = self.client.get_node(node_id)
            self.nodes[name] = node
            logger.info(f"Registered node '{name}' with ID '{node_id}'")
        except Exception as e:
            logger.error(f"Failed to register node {node_id}: {e}")

    async def read_value(self, name: str):
        if name not in self.nodes:
            logger.error(f"Node '{name}' not registered.")
            return None
        value = await self.nodes[name].read_value()
        return value

    async def write_value(self, name: str, value, variant_type):
        from asyncua import ua
        if name not in self.nodes:
            logger.error(f"Node '{name}' not registered.")
            return
        
        dv = ua.DataValue(ua.Variant(value, variant_type))
        await self.nodes[name].write_value(dv)

# Example usage
async def main():
    # S7-1200 PLC IP adresini giriniz (Örn: 192.168.0.1)
    client = OPCUAClient("opc.tcp://192.168.0.1:4840")
    await client.connect()
    
    # Gerçek PLC Node ID'leri
    await client.register_node("Temperature", "ns=4;i=3")
    await client.register_node("HeaterPower", "ns=4;i=2")
    
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
