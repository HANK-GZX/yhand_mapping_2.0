"""Real DDS shutdown regression tests. The SDK executable is always a fake driver."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


PACKAGE = Path(__file__).resolve().parents[1]


def record(event, **values):
    with open(os.environ['PRETIGHTEN_TEST_EVENTS'], 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(event=event, **values)) + '\n')


def fake_driver():
    import rclpy
    from devices_pkg.msg import YHandMsg
    from rclpy.signals import SignalHandlerOptions

    stopping = False

    def on_signal(signum, _frame):
        nonlocal stopping
        record('signal', signum=signum)
        if os.environ.get('PRETIGHTEN_TEST_IGNORE') != '1':
            stopping = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, on_signal)
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('fake_pretighten_driver')
    publisher = node.create_publisher(YHandMsg, sys.argv[2], 10)
    positions = [float(i * 2) for i in range(1, 29)]
    zeroed = False
    mode = os.environ.get('PRETIGHTEN_TEST_ZERO_MODE', 'success')
    if mode == 'cached_zero':
        positions[:] = [0.0] * 28

    def command(message):
        nonlocal zeroed
        record('command', motors=[
            [motor.pos, motor.vel, motor.tor, motor.kp, motor.kd]
            for motor in message.motors
        ], pretighten=message.pretightenforce.flag,
            zero_torque=message.pretightenforce.torque, zero_ms=message.pretightenforce.ms)
        if message.pretightenforce.flag:
            record('sdk_set_zero_started')
            time.sleep(0.3)  # SDK callback blocks until all 28 Set_Zero calls finish.
            zeroed = True
            if mode not in ('unchanged', 'no_feedback'):
                positions[:] = [0.0] * 28
            if mode == 'bad_motor':
                positions[27] = 12.0
            if mode == 'nan_motor':
                positions[27] = float('nan')
            record('sdk_set_zero_completed', motors=list(range(1, 29)))
        if zeroed and mode in ('no_feedback', 'cached_zero'):
            return
        if zeroed and mode == 'one_frame' and not message.pretightenforce.flag:
            return
        response = YHandMsg()
        for motor, pos in zip(response.motors, positions):
            motor.pos = pos
        if zeroed and mode == 'motor_error':
            response.motors[27].error = 7
        publisher.publish(response)
        record('feedback', zeroed=zeroed)

    node.create_subscription(YHandMsg, sys.argv[3], command, 10)
    record('ready', pid=os.getpid(), controller_pid=os.getppid(), pgid=os.getpgrp())
    try:
        while not stopping:
            rclpy.spin_once(node, timeout_sec=0.01)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def launch_fixture(driver_prefix):
    from launch import LaunchService

    spec = importlib.util.spec_from_file_location(
        'pretighten_launch', PACKAGE / 'launch' / 'yhand_pretighten_launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Redirect the actual launch's SDK path. Never execute the hardware SDK in tests.
    module.get_package_prefix = lambda _package: driver_prefix
    module.get_package_share_directory = lambda _package: str(PACKAGE)
    description = module.generate_launch_description()
    from launch.actions import ExecuteProcess
    from launch.substitutions import TextSubstitution
    for entity in description.entities:
        if isinstance(entity, ExecuteProcess):
            # Short timeout only in tests; launch still exercises the real script.
            entity.cmd.extend([[TextSubstitution(text='--zero-timeout')],
                               [TextSubstitution(text='1.5')]])
    service = LaunchService(noninteractive=False)
    service.include_launch_description(description)
    return service.run()


class PretightenShutdownTest(unittest.TestCase):
    def check_shutdown(self, trigger, ignore_driver=False, standalone=False, zero_mode='success'):
        with tempfile.TemporaryDirectory(prefix='pretighten-test-') as directory:
            root = Path(directory)
            events_path = root / 'events.jsonl'
            driver = root / 'lib' / 'devices_pkg' / 'Y_Hand_Node'
            driver.parent.mkdir(parents=True)
            driver.write_text(
                f'#!{sys.executable}\nimport runpy\n'
                f'runpy.run_path({str(Path(__file__).resolve())!r}, run_name="fake_driver_fixture")\n',
                encoding='utf-8',
            )
            driver.chmod(0o755)
            env = dict(os.environ, PRETIGHTEN_TEST_EVENTS=str(events_path),
                       PRETIGHTEN_TEST_IGNORE=str(int(ignore_driver)),
                       PRETIGHTEN_TEST_ZERO_MODE=zero_mode,
                       ROS_LOG_DIR=str(root / 'ros-log'), ROS_LOCALHOST_ONLY='1',
                       ROS_DOMAIN_ID=str(100 + os.getpid() % 100),
                       PYTHONDONTWRITEBYTECODE='1')
            if standalone:
                cmd = [sys.executable, str(PACKAGE / 'scripts' / 'yhand_pretighten.py'),
                       '--driver-executable', str(driver), '--driver-node-name', 'y_hand_node',
                       '--cmd-topic', 'yhand_left_cmd', '--feedback-topic', 'yhand_left_feedback',
                       '--dev-config', 'unused by fake driver', '--zero-timeout', '1.5']
            else:
                cmd = [sys.executable, str(Path(__file__).resolve()), '--launch-fixture', directory]
            with (root / 'output.log').open('w+') as output:
                process = subprocess.Popen(cmd, env=env, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                ready = None
                try:
                    def read_events():
                        if not events_path.exists():
                            return []
                        return [json.loads(line) for line in events_path.read_text().splitlines()
                                if line.endswith('}')]

                    deadline = time.monotonic() + 15
                    while True:
                        events = read_events()
                        ready = next((e for e in events if e['event'] == 'ready'), None)
                        if any(e['event'] == 'command' and e['motors'][0][2] < 0 for e in events):
                            break
                        if process.poll() is not None or time.monotonic() > deadline:
                            self.fail('Fake driver did not receive pretighten current')
                        time.sleep(0.02)
                    self.assertNotEqual(ready['pgid'], process.pid)
                    started = time.monotonic()
                    if trigger == 'group':
                        os.killpg(process.pid, signal.SIGINT)
                    else:
                        process.send_signal(trigger)
                    process.wait(timeout=9)
                    self.assertLess(time.monotonic() - started, 8)
                    expects_zero = trigger != signal.SIGTERM
                    expects_success = not expects_zero or zero_mode == 'success'
                    if standalone:
                        self.assertEqual(process.returncode, 0 if expects_success else 1)
                    events = read_events()
                    first_zero = next(i for i, e in enumerate(events)
                                      if e['event'] == 'command' and e['motors'][0][2] == 0)
                    first_signal = next(i for i, e in enumerate(events) if e['event'] == 'signal')
                    self.assertLess(first_zero, first_signal)
                    zeros = [e for e in events[first_zero:] if e['event'] == 'command']
                    self.assertGreaterEqual(len(zeros), 10)
                    for event in zeros:
                        self.assertEqual(event['motors'], [[0.0] * 5] * 28)
                    zero_requests = [i for i, e in enumerate(events)
                                     if e['event'] == 'command' and e['pretighten']]
                    self.assertEqual(len(zero_requests), int(expects_zero))
                    if expects_zero:
                        zero_index = zero_requests[0]
                        self.assertLess(first_zero, zero_index)
                        self.assertEqual(events[zero_index]['zero_torque'], 0)
                        self.assertEqual(events[zero_index]['zero_ms'], 0)
                        finished = next(i for i, e in enumerate(events)
                                        if e['event'] == 'sdk_set_zero_completed')
                        self.assertEqual(events[finished]['motors'], list(range(1, 29)))
                        self.assertLess(finished, first_signal)
                        output.seek(0)
                        logs = output.read()
                        if expects_success:
                            self.assertIn('零点反馈检查通过', logs)
                            self.assertGreaterEqual(sum(e['event'] == 'feedback' and e['zeroed']
                                                        for e in events), 3)
                        else:
                            self.assertIn('零点反馈检查未通过', logs)
                            self.assertNotIn('零点反馈检查通过', logs)
                            if zero_mode in ('bad_motor', 'nan_motor', 'motor_error'):
                                self.assertIn('28:pos=', logs)
                    for pid in (ready['pid'], ready['controller_pid']):
                        with self.assertRaises(ProcessLookupError):
                            os.kill(pid, 0)
                except Exception:
                    output.seek(0)
                    print(output.read(), file=sys.stderr)
                    raise
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    if ready is not None:
                        for pid in (ready['pid'], ready['controller_pid']):
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass

    def test_ctrl_c_sends_zero_before_driver_exit(self):
        self.check_shutdown('group')

    def test_sigint_to_launch_parent_also_stops_current(self):
        self.check_shutdown(signal.SIGINT)

    def test_stuck_driver_receives_zero_then_is_killed(self):
        self.check_shutdown('group', ignore_driver=True)

    def test_sigterm_to_controller_sends_zero(self):
        self.check_shutdown(signal.SIGTERM, standalone=True)

    def test_nonzero_motor_is_reported_and_driver_still_stops(self):
        self.check_shutdown(signal.SIGINT, standalone=True, zero_mode='bad_motor')

    def test_missing_feedback_cannot_confirm_zero(self):
        self.check_shutdown(signal.SIGINT, standalone=True, zero_mode='no_feedback')

    def test_pre_request_zero_feedback_is_not_reused_as_confirmation(self):
        self.check_shutdown(signal.SIGINT, standalone=True, zero_mode='cached_zero')

    def test_nan_feedback_cannot_confirm_zero(self):
        self.check_shutdown(signal.SIGINT, standalone=True, zero_mode='nan_motor')

    def test_zero_position_with_motor_error_is_rejected(self):
        self.check_shutdown(signal.SIGINT, standalone=True, zero_mode='motor_error')

    def test_one_zero_frame_cannot_confirm_zero(self):
        self.check_shutdown(signal.SIGINT, standalone=True, zero_mode='one_frame')


if __name__ == 'fake_driver_fixture':
    fake_driver()
elif __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--launch-fixture':
        raise SystemExit(launch_fixture(sys.argv[2]))
    unittest.main()
