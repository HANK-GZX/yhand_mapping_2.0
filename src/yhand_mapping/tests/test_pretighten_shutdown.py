"""Exercise SDK demo delegation, all-motor feedback checks, and shutdown."""

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
SDK_CONFIG = Path('YAML/Y_Hand_M2/out/TOP.yaml')


def record(event, **values):
    with open(os.environ['PRETIGHTEN_TEST_EVENTS'], 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(event=event, **values)) + '\n')


def fake_demo():
    stopping = False

    def on_signal(signum, _frame):
        nonlocal stopping
        record('signal', signum=signum)
        if os.environ['PRETIGHTEN_TEST_IGNORE'] != '1':
            stopping = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, on_signal)
    record('ready', pid=os.getpid(), parent=os.getppid(), pgid=os.getpgrp(),
           sdk_config=sys.argv[1], dev_config=sys.argv[2], limits=sys.argv[3:])
    feedback_mode = os.environ['PRETIGHTEN_TEST_FEEDBACK']
    frames = 0
    while not stopping:
        if feedback_mode != 'missing':
            motors = [[0.0, 0] for _ in range(28)]
            motors[0][0] = -0.125
            if feedback_mode == 'faulty' and frames >= 2:
                motors[26][0] = 360.0
                motors[27][1] = 7
            print('M2_POSITION_FRAME', *(value for motor in motors for value in motor), flush=True)
            frames += 1
        time.sleep(0.01)
    record('stopped')


def launch_fixture(prefix, hand):
    from launch import LaunchDescription, LaunchService
    from launch.actions import SetLaunchConfiguration

    spec = importlib.util.spec_from_file_location(
        'pretighten_launch', PACKAGE / 'launch' / 'yhand_pretighten_launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.get_package_prefix = lambda _package: prefix
    module.get_package_share_directory = lambda package: str(PACKAGE) if package == 'yhand_mapping' else prefix
    service = LaunchService(noninteractive=False)
    service.include_launch_description(LaunchDescription([
        SetLaunchConfiguration('target_hand', hand),
        *module.generate_launch_description().entities,
    ]))
    return service.run()


class PretightenShutdownTest(unittest.TestCase):
    def check_demo(self, trigger, hand='left', ignore_demo=False, standalone=False, feedback_mode='normal'):
        expect_timeout = feedback_mode != 'normal'
        with tempfile.TemporaryDirectory(prefix='pretighten-test-') as directory:
            root = Path(directory)
            events_path = root / 'events.jsonl'
            demo = root / 'lib/yhand_mapping/Y_Hand_M2_Demo'
            demo.parent.mkdir(parents=True)
            demo.write_text(
                f'#!{sys.executable}\nimport runpy\n'
                f'runpy.run_path({str(Path(__file__).resolve())!r}, run_name="fake_demo_fixture")\n',
                encoding='utf-8',
            )
            demo.chmod(0o755)
            config = root / SDK_CONFIG
            config.parent.mkdir(parents=True)
            config.write_text('Fake M2 SDK configuration\n', encoding='utf-8')

            env = dict(os.environ, PRETIGHTEN_TEST_EVENTS=str(events_path),
                       PRETIGHTEN_TEST_IGNORE=str(int(ignore_demo)),
                       PRETIGHTEN_TEST_FEEDBACK=feedback_mode,
                       ROS_LOG_DIR=str(root / 'ros-log'), PYTHONDONTWRITEBYTECODE='1')
            if standalone:
                command = [sys.executable, str(PACKAGE / 'scripts/yhand_pretighten.py'),
                           '--demo-executable', str(demo), '--sdk-config', str(config),
                           '--hand', hand, '--dev-config', 'fake device config',
                           '--zero-timeout', '0.5' if expect_timeout else '10']
            else:
                command = [sys.executable, str(Path(__file__).resolve()), '--launch-fixture', directory, hand]

            def read_events():
                if not events_path.exists():
                    return []
                return [json.loads(line) for line in events_path.read_text().splitlines()
                        if line.endswith('}')]

            output_path = root / 'output.log'
            with output_path.open('w', encoding='utf-8') as output:
                process = subprocess.Popen(command, env=env, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                ready = None
                try:
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        ready = next((e for e in read_events() if e['event'] == 'ready'), None)
                        confirmed = '绝对零位确认通过' in output_path.read_text(encoding='utf-8')
                        if ready is not None and (expect_timeout or confirmed):
                            break
                        if process.poll() is not None:
                            self.fail('SDK demo exited before the expected state')
                        time.sleep(0.01)
                    else:
                        self.fail('SDK demo did not reach the expected state')
                    self.assertNotEqual(ready['pgid'], process.pid)
                    self.assertEqual(ready['sdk_config'], str(config))
                    self.assertEqual(ready['limits'], ['0.5'] * 28)
                    if not standalone:
                        config_path = PACKAGE / f'config/devices/{hand}_hand_device_config.json'
                        device = json.loads(config_path.read_text())
                        self.assertIn(f"PC_IP: {device['pc_ip']}", ready['dev_config'])
                        self.assertIn(f"SN: {device['sn']}", ready['dev_config'])
                        for board in device['boards']:
                            self.assertIn(f"IP: {board['ip']}", ready['dev_config'])
                            if 'port' in board:
                                self.assertIn(f"Port: {board['port']}", ready['dev_config'])
                    if not expect_timeout:
                        if trigger == 'group':
                            os.killpg(process.pid, signal.SIGINT)
                        else:
                            process.send_signal(trigger)
                    process.wait(timeout=8)
                    self.assertEqual(process.returncode, int(expect_timeout))

                    events = read_events()
                    self.assertIn(signal.SIGINT, [e['signum'] for e in events if e['event'] == 'signal'])
                    if ignore_demo:
                        self.assertIn(signal.SIGTERM, [e['signum'] for e in events if e['event'] == 'signal'])
                    else:
                        self.assertEqual(events[-1]['event'], 'stopped')
                    logs = output_path.read_text(encoding='utf-8')
                    motor_labels = {f'M{index:02d}' for index in range(1, 29)}
                    reported = {}
                    for line in logs.splitlines():
                        columns = line.rsplit('] ', 1)[-1].split()
                        if columns and columns[0] in motor_labels:
                            reported[columns[0]] = columns[1:]
                    self.assertEqual(set(reported), motor_labels)
                    if expect_timeout:
                        self.assertIn('绝对零位确认超时', logs)
                        self.assertNotIn('绝对零位确认通过', logs)
                    if feedback_mode == 'faulty':
                        self.assertEqual(reported['M27'], ['+360.0000', '360.0000', '0', '未到位'])
                        self.assertEqual(reported['M28'], ['+0.0000', '0.0000', '7', '未到位'])
                    elif feedback_mode == 'missing':
                        self.assertEqual(reported['M28'], ['-', '-', '-', '无反馈'])
                    else:
                        self.assertEqual(reported['M01'], ['-0.1250', '0.1250', '0', '到位'])
                    with self.assertRaises(ProcessLookupError):
                        os.kill(ready['pid'], 0)
                except Exception:
                    print(output_path.read_text(encoding='utf-8'), file=sys.stderr)
                    raise
                finally:
                    if process.poll() is None:
                        if ready is not None:
                            try:
                                os.kill(ready['pid'], signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()

    def test_left_launch_starts_demo_and_reports_28_motors(self):
        self.check_demo('group')

    def test_right_launch_passes_device_config_and_honors_parent_sigint(self):
        self.check_demo(signal.SIGINT, hand='right')

    def test_stuck_demo_is_terminated(self):
        self.check_demo('group', ignore_demo=True)

    def test_standalone_releases_demo_on_sigterm(self):
        self.check_demo(signal.SIGTERM, hand='right', standalone=True)

    def test_unwrapped_position_and_motor_fault_prevent_confirmation(self):
        self.check_demo(None, standalone=True, feedback_mode='faulty')

    def test_missing_feedback_reports_no_confirmation(self):
        self.check_demo(None, standalone=True, feedback_mode='missing')


if __name__ == 'fake_demo_fixture':
    fake_demo()
elif __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--launch-fixture':
        raise SystemExit(launch_fixture(sys.argv[2], sys.argv[3]))
    unittest.main()
