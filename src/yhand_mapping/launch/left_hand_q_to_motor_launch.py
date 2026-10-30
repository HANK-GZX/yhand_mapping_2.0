"""Start left-hand single- or multi-finger q-to-motor control with manual/Manus input."""
from pathlib import Path
import shutil
import signal

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            LogInfo, OpaqueFunction, RegisterEventHandler, Shutdown, TimerAction)
from launch.event_handlers import OnShutdown
from launch.events import matches_action
from launch.events.process import SignalProcess
from launch.substitutions import LaunchConfiguration


def _enabled(value):
    if value.lower() not in ('true','false'):
        raise ValueError(f'Expected true/false, got {value!r}')
    return value.lower()=='true'


def _launch_setup(context):
    value = lambda name:LaunchConfiguration(name).perform(context)
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    controller_script = scripts/'q_2_m linear mapping/left_little_q_to_motor_test.py'
    input_script = scripts/'joint_angle_terminal_input.py'
    glove_script = scripts/'minimal_example_dual_stream_commented.py'
    finger = value('finger')
    model = value('model')
    if model not in ('linear','mlp'):
        raise ValueError('model只支持linear或mlp')
    finger_label = {'little':'小指','ring':'无名指','middle':'中指','index':'食指','thumb':'拇指',
                    'both':'无名指和小指','three':'中指、无名指和小指','four':'食指、中指、无名指和小指',
                    'five':'拇指、食指、中指、无名指和小指'}[finger]
    mode = {'1':'manual','2':'manus','manual':'manual','manus':'manus'}[value('input_mode')]
    joint_topic = (value('joint_angle_topic').strip() or f'/left_{finger}_target_q') if mode=='manual' else ''
    if mode=='manus' and not _enabled(value('relative')):
        raise ValueError('Manus模式使用相对角度，请保持relative:=true')
    start_glove = mode=='manus' and _enabled(value('start_glove_node'))
    input_enabled = mode=='manual' and _enabled(value('start_angle_input_terminal'))
    if finger in ('both','three','four','five') and input_enabled:
        count = {'both':8,'three':12,'four':16,'five':20}[finger]
        raise ValueError(f'多指模式请用input_mode:=2；手动{count}角话题输入需start_angle_input_terminal:=false')
    for path in [controller_script,*([input_script] if input_enabled else []),
                 *([glove_script] if start_glove else [])]:
        if not path.is_file():
            raise FileNotFoundError(path)
    terminal = shutil.which('gnome-terminal') if input_enabled else None
    if input_enabled and terminal is None:
        raise RuntimeError('自动输入窗口需要gnome-terminal；无桌面时可用start_angle_input_terminal:=false并向joint_angle_topic发布四个q。')
    command = ['python3','-u',str(controller_script),
               '--finger',finger,
               '--input-mode',mode,'--model',value('model'),
               '--publish-hz',value('publish_hz')]
    mlp_dir = LaunchConfiguration('mlp_model_dir', default='').perform(context).strip()
    if mlp_dir:
        command += ['--mlp-model-dir', mlp_dir]
    combo_path = LaunchConfiguration('thumb_combo_calibration', default='').perform(context).strip()
    if combo_path:
        if finger not in ('thumb', 'five'):
            raise ValueError('thumb_combo_calibration仅适用于thumb/five')
        command += ['--thumb-combo-calibration', combo_path]
    if value('current').strip():
        command += ['--current',value('current')]
    if mode=='manual':
        command += ['--target-topic',joint_topic]
    else:
        command += ['--manus-topic',value('manus_topic'),'--manus-raw-topic',value('manus_raw_topic'),
                    '--manus-zero-seconds',value('manus_zero_seconds'),
                    '--manus-timeout',value('manus_timeout')]
    if _enabled(value('relative')):
        command.append('--relative')
    else:
        command.append('--absolute')
    command.append('--start-driver' if _enabled(value('start_driver_node')) else '--no-start-driver')
    # The controller owns its SDK child in a separate session. Launch only
    # signals the controller, which sends stop frames before closing that SDK.
    controller = ExecuteProcess(cmd=command,name=f'left_{finger}_q_to_motor_test',output='screen',
                                sigterm_timeout='6.0',sigkill_timeout='1.0',
                                on_exit=Shutdown(reason=f'{finger}-finger controller exited'))

    def stop_controller(event,context):
        details = controller.process_details
        if not details or not details.get('pid'):
            return []
        signum = signal.SIGINT if event.due_to_sigint else signal.SIGTERM
        return [EmitEvent(event=SignalProcess(signal_number=signum,
                                             process_matcher=matches_action(controller)))]

    actions = [RegisterEventHandler(OnShutdown(on_shutdown=stop_controller)),
               LogInfo(msg=[f'[left_{finger}] {finger_label}模式：',mode,
                            '；Manus使用启动零位后的相对变化。' if mode=='manus' else '；终端输入四个关节角。'])]
    if start_glove:
        actions.append(ExecuteProcess(cmd=['python3','-u',str(glove_script)],name='glove_publisher',output='screen',
                                      on_exit=Shutdown(reason='Manus publisher exited')))
    actions.append(controller)
    if input_enabled:
        input_process = ExecuteProcess(
            cmd=[terminal,'--wait','--title',f'left_{finger}_q_input','--',
                 'python3','-u',str(input_script),'--hand','left','--finger',finger,
                 '--topic',joint_topic,'--burst-count','5'],
            name=f'left_{finger}_q_input',output='screen',
            on_exit=Shutdown(reason=f'{finger}-finger angle input terminal closed'))
        actions.append(TimerAction(period=1.0,actions=[input_process]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('finger',default_value='little',choices=['little','ring','middle','index','thumb','both','three','four','five'],
                              description='Single finger or both:ring+little / three:middle+ring+little / four:non-thumb / five:all fingers'),
        DeclareLaunchArgument('input_mode',default_value='1',choices=['1','2','manual','manus'],
                              description='1/manual: typed relative q; 2/manus: live left Manus relative changes'),
        DeclareLaunchArgument('relative',default_value='true',choices=['true','false'],
                              description='Input q is relative to the fixed collection reference'),
        DeclareLaunchArgument('model',default_value='linear',choices=['linear','mlp'],
                              description='linear: existing calibration policy; mlp: trained neural networks'),
        DeclareLaunchArgument('mlp_model_dir',default_value='',
                              description='Optional MLP JSON directory; defaults to config/q_to_motor_mlp'),
        DeclareLaunchArgument('thumb_combo_calibration',default_value='',
                              description='Optional measured thumb combo path JSON; thumb/five only; empty keeps existing control'),
        DeclareLaunchArgument('start_driver_node',default_value='true',choices=['true','false']),
        DeclareLaunchArgument('start_angle_input_terminal',default_value='true',choices=['true','false']),
        DeclareLaunchArgument('joint_angle_topic',default_value='',
                              description='Manual target topic; empty selects /left_<finger>_target_q'),
        DeclareLaunchArgument('start_glove_node',default_value='true',choices=['true','false'],
                              description='In Manus mode start the existing dual-stream publisher; false to reuse it'),
        DeclareLaunchArgument('manus_topic',default_value='/glove_joint_data'),
        DeclareLaunchArgument('manus_raw_topic',default_value='/glove_joint_data_raw_skeleton'),
        DeclareLaunchArgument('manus_zero_seconds',default_value='1.0'),
        DeclareLaunchArgument('manus_timeout',default_value='0.5'),
        DeclareLaunchArgument('current',default_value='',
                              description='Empty: thumb/index/middle/ring 0 A, little -0.08 A; set explicitly to override'),
        DeclareLaunchArgument('publish_hz',default_value='50.0'),
        OpaqueFunction(function=_launch_setup),
    ])
