	
import serial

ser = serial.serial_for_url('rfc2217://localhost:4000', baudrate=115200, timeout=2)

with open('wokwi_readings.csv', 'w') as f:
    print("Capturing... Ctrl+C to stop")
    try:
        while True:
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                print(line)
                f.write(line + '\n')
                f.flush()
    except KeyboardInterrupt:
        print("\nStopped. Saved to wokwi_readings.csv")