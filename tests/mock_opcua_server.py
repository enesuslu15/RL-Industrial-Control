import asyncio
import logging
from asyncua import Server, ua

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Mock_OPCUA_Server")

async def main():
    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://127.0.0.1:4841/freeopcua/server/")
    
    # Setup our own namespace, not really necessary but should as spec
    uri = "http://examples.freeopcua.github.io"
    idx = await server.register_namespace(uri)

    # Populating our address space
    myobj = await server.nodes.objects.add_object(idx, "TankSimulation")
    
    # Variables: Temperature (Sensor), HeaterPower (Actuator)
    myvar_temp = await myobj.add_variable(idx, "Temperature", 20.0)
    myvar_power = await myobj.add_variable(idx, "HeaterPower", 0.0)
    
    # Set variables to be writable by clients
    await myvar_temp.set_writable()
    await myvar_power.set_writable()

    logger.info("Starting Mock OPC UA Server...")
    async with server:
        # Simulate simple heating process
        temperature = 20.0
        ambient_temp = 20.0
        while True:
            await asyncio.sleep(0.5)  # Update every 500ms
            
            # Read current power from server (in case client updated it)
            power = await myvar_power.read_value()
            
            # Very basic tank simulation:
            # Heating: +0.1 degree per % power
            # Cooling (loss to ambient): -0.05 degree difference
            heating_effect = power * 0.05
            cooling_effect = (temperature - ambient_temp) * 0.02
            
            temperature += heating_effect - cooling_effect
            
            # Write new temperature to server
            await myvar_temp.write_value(temperature)
            logger.info(f"Power: {power:.1f}%, Temp: {temperature:.2f} C")

if __name__ == "__main__":
    asyncio.run(main())
