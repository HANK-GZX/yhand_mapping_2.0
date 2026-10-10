import asyncio
import os
import sys
try:
    from pynput.keyboard import Listener
except Exception:
    Listener = None
from mocap_api import *

EVENT_POLL_SLEEP_SEC = 0.005

class MCPBase:
    def __init__(self):
        # Initialize current command and running command status
        self.current_command = -1  # Current command being executed
        self.capture_key = False  # Capture status flag
        self.connent_key = False  # Connection status flag
        self.auto_start_capture = (
            str(os.environ.get("MOCAP_AUTO_START_CAPTURE", "")).strip().lower() in ("1", "true", "yes", "on")
            or (not sys.stdin.isatty())
        )
        self.auto_start_requested = False
        self.app = MCPApplication()  # Create application instance
        settings = MCPSettings()  # Create settings instance
        
        # Configure BVH data format to binary
        settings.set_bvh_data(MCPBvhData.Binary)
        # Enable BVH data transformation
        settings.set_bvh_transformation(MCPBvhDisplacement.Enable)
        # Set rotation order to YZX
        settings.set_bvh_rotation(MCPBvhRotation.YXZ)
        # Configure UDP data transmission address and port
        settings.SetSettingsUDPEx('192.168.3.246', 8002)
        settings.SetSettingsUDPServer('192.168.3.110', 7003)
        
        # Apply configuration to application instance and open connection
        self.app.set_settings(settings)
        self.app.open()

    def get_current_command_title(self):
        # Return the title corresponding to the current command
        if self.current_command == EMCPCommand.CommandStartCapture:
            return 'Start Capture'
        elif self.current_command == EMCPCommand.CommandStopCapture:
            return 'Stop Capture'
        elif self.current_command == EMCPCommand.CommandCalibrateMotion:
            return 'Calibrate Motion'
        elif self.current_command == EMCPCommand.CommandResumeOriginalHandsPosture:
            return 'Resume Hands'
        elif self.current_command == EMCPCommand.CommandClearZeroMotionDrift:
            return 'Reset 0 Motion Drift'
        elif self.current_command == EMCPCommand.CommandResumeOriginalPosture:
            return 'Resume Body'
        elif self.current_command == EMCPCommand.CommandZeroPosition:
            return 'Zero Position'
        else:
            return 'None'

    def check_current_command(self, running_command):
        # Check if the specified command can be run
        if self.connent_key == False:
            print('Link failure.')  # Print error message if not connected
            return False
        elif self.current_command != -1:
            # If another command is running, prohibit executing other commands
            print(f'Pending command {self.get_current_command_title()} is running.')
            return False
        # Commands other than start capture need capture to be started first
        elif self.capture_key == False and running_command != EMCPCommand.CommandStartCapture:
            print('Please start capture command first.')
            return False
        return True

    def running_command(self, running_command):
        # Execute the specified command
        if self.check_current_command(running_command) == True:            
            self.app.queue_command(running_command)  # Execute command
            self.current_command = running_command  # Update current command
            print(f'Pending command {self.get_current_command_title()} is running.')

    def handleNotify(self, notifyData):
        # Handle notification events
        if notifyData._notify == MCPEventNotify.Notify_SystemUpdated:
            mcpSystem = MCPSystem(notifyData._notifyHandle)  # Get system information
            print(f'MasterInfo : ( Version : {mcpSystem.get_master_version()}, SerialNumber : {mcpSystem.get_master_serial_number()} )')

    def handleResult(self, commandRespond):
        # Handle command results
        command = MCPCommand()
        _commandHandle = commandRespond._commandHandle
        ret_code = command.get_result_code(_commandHandle)
        if ret_code != 0:
            ret_msg = command.get_result_message(_commandHandle)
            print(f'ResultCode: {self.get_current_command_title()}, ResultMessage: {ret_msg}')  # Print command execution error message
        else:
            print(f'{self.get_current_command_title()} done.')  # Print command execution success message
            if self.current_command == EMCPCommand.CommandStopCapture:
                self.capture_key = False
        command.destroy_command(_commandHandle)  # Destroy command handle
        self.current_command = -1  # Reset current command

    def handleRunning(self, commandRespond):
        # Handle calibration progress
        _calibrateProgressHandle = MCPCommand().get_progress(commandRespond._commandHandle)
        progressHandle = MCPCalibrateMotionProgress(_calibrateProgressHandle)
        count = progressHandle.get_count_of_support_poses()

        str_poses = "Support poses:"
        for i in range(count):
            name = progressHandle.get_name_of_support_poses(i)
            str_poses += name
            if i + 1 != count:
                str_poses += ", "
        str_poses += " : "     
        result_progress = 0
        # Get calibration step and calibration name
        result_current_step, result_p_name = progressHandle.get_step_current_pose()
        if result_current_step == MCPCalibrateMotionProgressStep.CalibrateMotionProgressStep_Countdown:
            # Get calibration countdown
            result_countdown, result_p_name = progressHandle.get_countdown_current_pose()
            str_poses += (f"Calibration-({result_p_name})-Countdown {result_countdown}.")
        elif result_current_step == MCPCalibrateMotionProgressStep.CalibrateMotionProgressStep_Progress:
            # Get calibration progress
            result_progress, result_p_name = progressHandle.get_progress_current_pose()
            str_poses += (f"Calibration-({result_p_name})-Progress {result_progress}.")

        elif result_current_step == MCPCalibrateMotionProgressStep.CalibrateMotionProgressStep_Prepare:
            str_poses += (f"Calibration-({result_p_name}) is preparing.")
        else:
            str_poses += (f'Calibration-Unknown({result_current_step}) is running.')

        print(str_poses) 

        if result_progress == 100:
            print('Waiting for the completion of the calibration of human posture command execution')   

    async def update(self):
        try: 
            # Asynchronous update function to handle event loop
            while True:
                evts = self.app.poll_next_event()  # Get next event
                for evt in evts:
                    self.connent_key = True
                    if (
                        self.auto_start_capture
                        and (not self.auto_start_requested)
                        and (not self.capture_key)
                        and self.current_command == -1
                    ):
                        print("Auto-start capture enabled, sending Start Capture command")
                        self.running_command(EMCPCommand.CommandStartCapture)
                        self.auto_start_requested = True
                    if evt.event_type == MCPEventType.AvatarUpdated:
                        if self.current_command != EMCPCommand.CommandCalibrateMotion and self.current_command == -1:
                            self.capture_key = True
                            self.handleAvatar(evt.event_data.avatar_handle)
                    elif evt.event_type == MCPEventType.Notify:
                        self.handleNotify(evt.event_data.notifyData)
                    elif evt.event_type == MCPEventType.CommandReply:
                        if evt.event_data.commandRespond._replay == MCPReplay.MCPReplay_Response:
                            print('MCPReplay_Response')
                        elif evt.event_data.commandRespond._replay == MCPReplay.MCPReplay_Running:
                            self.handleRunning(evt.event_data.commandRespond)
                        elif evt.event_data.commandRespond._replay == MCPReplay.MCPReplay_Result:
                            self.handleResult(evt.event_data.commandRespond)
                    elif evt.event_type == MCPEventType.RigidBodyUpdated:
                        print('rigid body updated')
                await asyncio.sleep(EVENT_POLL_SLEEP_SEC)
        except Exception as e:
            print(f"An error occurred: {e}")    

    async def main_async(self):
        # Main asynchronous function
        main = self
        loop = asyncio.get_event_loop()

        def dispatch_command(key_name):
            key_name = str(key_name).strip().lower()
            if not key_name:
                return True
            print(f"Key pressed: {key_name}")
            if key_name == 'n':
                main.running_command(EMCPCommand.CommandStartCapture)
            elif key_name == 'f':
                main.running_command(EMCPCommand.CommandStopCapture)
            elif key_name == 'c':
                main.running_command(EMCPCommand.CommandCalibrateMotion)
            elif key_name == 'r':
                main.running_command(EMCPCommand.CommandResumeOriginalHandsPosture)
            elif key_name == '0':
                main.running_command(EMCPCommand.CommandClearZeroMotionDrift)
            elif key_name == 'o':
                main.running_command(EMCPCommand.CommandResumeOriginalPosture)
            elif key_name == 'z':
                main.running_command(EMCPCommand.CommandZeroPosition)
            elif key_name in ('esc', 'q', 'quit', 'exit'):
                print("Exit command received, exiting program")
                return False
            else:
                print(
                    "Unknown command. Use one of: "
                    "n(start), f(stop), c(calibrate), r(resume hands), "
                    "0(reset drift), o(resume body), z(zero position), q(exit)"
                )
            return True

        # Keyboard event handler function
        def on_key_press(key):
            try:
                key_name = key.char.lower()
                return dispatch_command(key_name)
            except AttributeError:
                if key == key.esc:
                    print("ESC key pressed, exiting program")
                    return False  # Exit listener

        asyncio.run_coroutine_threadsafe(main.update(), loop)  # Start event update
        print(
            "Press N to Start Capture, C to calibrate, R to Resume Hands, "
            "0 to Reset 0 Motion Drift, O to Resume Body, Z to Zero Position, "
            "F to Stop Capture, press ESC to exit program"
        )
        if self.auto_start_capture:
            print("Non-interactive mode detected: auto-start capture is enabled.")

        if Listener is None:
            print("pynput is unavailable, keyboard listener disabled.")

        if Listener is not None:
            with Listener(on_press=on_key_press) as listener:
                if sys.stdin.isatty():
                    print(
                        "Terminal fallback is enabled: you can also type "
                        "n/f/c/r/0/o/z/q then press Enter."
                    )
                    while True:
                        cmd = await loop.run_in_executor(None, input, "> ")
                        if dispatch_command(cmd) is False:
                            listener.stop()
                            break
                else:
                    while True:
                        await asyncio.sleep(1.0)
        else:
            if sys.stdin.isatty():
                print(
                    "Terminal fallback is enabled: you can also type "
                    "n/f/c/r/0/o/z/q then press Enter."
                )
                while True:
                    cmd = await loop.run_in_executor(None, input, "> ")
                    if dispatch_command(cmd) is False:
                        break
            else:
                while True:
                    await asyncio.sleep(1.0)
