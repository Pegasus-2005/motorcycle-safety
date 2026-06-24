import logging
import time
from typing import Dict, Optional

# Configure clean logging format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("AlertSystem")


def send_alert(
    message: str,
    location: Optional[Dict[str, float]] = None,
    severity: str = "LOW",
) -> bool:
    """Dispatches a safety alert.

    For the M1 Mac software prototype, this logs the alert to the console.
    When deployed to the Raspberry Pi hardware, this will interface with the
    SIM800L GSM module over UART serial.

    Args:
        message: The textual payload of the alert.
        location: Dict containing 'latitude' and 'longitude' (optional).
        severity: Severity level (e.g., 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL').

    Returns:
        bool: True if alert was dispatched successfully, False otherwise.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Build the output log structure
    border = "=" * 60
    log_msg = f"\n{border}\n"
    log_msg += f"🚨 SYSTEM ALERT - [{severity.upper()}] | {timestamp}\n"
    log_msg += f"Message: {message}\n"

    if location:
        lat = location.get("latitude")
        lon = location.get("longitude")
        log_msg += f"GPS Coordinates: Lat {lat}, Lon {lon}\n"
        log_msg += f"Map Link: https://maps.google.com/?q={lat},{lon}\n"

    log_msg += f"{border}\n"

    # Log to local console/files
    if severity.upper() in ["HIGH", "CRITICAL"]:
        logger.error(log_msg)
    else:
        logger.info(log_msg)

    # =========================================================================
    # SIM800L GSM MODULE INTEGRATION PLACEHOLDER
    # =========================================================================
    # Once the hardware arrives and this is run on a Raspberry Pi, the code
    # below can be uncommented and integrated.
    #
    # try:
    #     import serial
    #     # Open hardware serial port (UART on Raspberry Pi GPIO pins 14/15)
    #     # Typically '/dev/ttyS0' or '/dev/ttyAMA0'
    #     ser = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=1)
    #
    #     phone_number = "+91XXXXXXXXXX"  # Define emergency contact number
    #
    #     # Set SMS format to text mode
    #     ser.write(b"AT+CMGF=1\r")
    #     time.sleep(0.5)
    #
    #     # Send SMS command with destination phone number
    #     sms_command = f'AT+CMGS="{phone_number}"\r'
    #     ser.write(sms_command.encode())
    #     time.sleep(0.5)
    #
    #     # Send message body
    #     sms_body = f"Alert: {message}"
    #     if location:
    #         sms_body += f"\nLoc: http://maps.google.com/?q={lat},{lon}"
    #     ser.write(sms_body.encode())
    #     time.sleep(0.5)
    #
    #     # Write ASCII 'Substitute' character (Ctrl+Z / 0x1A) to send the message
    #     ser.write(b"\x1a")
    #     time.sleep(1.0)
    #
    #     response = ser.read(ser.in_waiting).decode()
    #     if "OK" in response or "+CMGS" in response:
    #         logger.info("SIM800L SMS dispatch reported success.")
    #     else:
    #         logger.warning(f"SIM800L SMS send state uncertain. Response: {response}")
    #
    # except Exception as e:
    #     # During prototype on Mac, this block handles the missing serial library/hardware
    #     pass
    # =========================================================================

    return True