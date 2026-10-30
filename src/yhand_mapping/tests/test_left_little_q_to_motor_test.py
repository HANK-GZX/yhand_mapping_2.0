"""Offline planning and isolated DDS integration tests; SDK is always a fake."""
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE/'scripts/q_2_m linear mapping/left_little_q_to_motor_test.py'
sys.path.insert(0,str(PACKAGE/'scripts'))
sys.path.insert(0,str(PACKAGE/'scripts/q_2_m linear mapping'))
spec = importlib.util.spec_from_file_location('little_test_control',SCRIPT)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def operator_angles(finger):
    if finger=='five':
        neutral,delta = operator_angles('four')
        return [8.,15.,-7.,2.]+neutral,[1.,10.,12.,3.]+delta
    neutral,delta = [14.,22.,-3.,8.],[-1.,10.,20.,5.]
    if finger in ('both','three','four'):
        neutral += [-6.,1.,2.,3.]
        delta += [2.,8.,15.,4.]
    if finger in ('three','four'):
        neutral += [3.,-4.,6.,2.]
        delta += [3.,6.,12.,2.]
    if finger=='four':
        neutral += [-3.,5.,-7.,9.]
        delta += [-2.,12.,18.,6.]
    return neutral,delta


def fake_sdk():
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from devices_pkg.msg import YHandMsg
    from std_msgs.msg import Float64MultiArray
    stopping = False
    events = Path(os.environ['LITTLE_TEST_EVENTS'])

    def record(kind,**values):
        with events.open('a') as stream:
            stream.write(json.dumps({'event':kind,'time':time.monotonic(),**values})+'\n')

    def on_signal(*_):
        nonlocal stopping
        stopping = True

    for signum in (signal.SIGINT,signal.SIGTERM):
        signal.signal(signum,on_signal)
    rclpy.init(args=[],signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node(sys.argv[1])
    pub = node.create_publisher(YHandMsg,sys.argv[2],10)
    glove_pub = node.create_publisher(Float64MultiArray,os.environ['LITTLE_TEST_GLOVE_TOPIC'],10)
    raw_pub = node.create_publisher(Float64MultiArray,os.environ['LITTLE_TEST_RAW_TOPIC'],10)
    target_pub = node.create_publisher(Float64MultiArray,os.environ['LITTLE_TEST_TARGET_TOPIC'],10)
    finger = os.environ.get('LITTLE_TEST_FINGER','little')
    mapper = control.create_mapper(finger)
    motor_ids = mapper.motor_ids
    glove_slice = mapper.glove_slice
    q_reference = list(mapper.q_reference)
    q_count = len(q_reference)
    operator_zero,operator_delta = operator_angles(finger)
    positions = [0.0]*28
    for index,mid in enumerate(motor_ids):
        positions[mid-1] = [20.0,-20.0,-5.0,-30.0][index%4]
    pose_frames = 0
    command_frames = 0

    def command(message):
        nonlocal pose_frames, command_frames
        command_frames += 1
        matrix = [[m.pos,m.vel,m.tor,m.kp,m.kd] for m in message.motors]
        record('command',matrix=matrix,pretighten=message.pretightenforce.flag)
        active = any(matrix[mid-1][3] for mid in motor_ids)
        if active:
            pose_frames += 1
        mode = os.environ.get('LITTLE_TEST_MODE','normal')
        if mode=='dropout' and pose_frames>=3:
            return
        reply = YHandMsg()
        for i,m in enumerate(message.motors):
            if m.kp>0:
                positions[i] = m.pos
            reply.motors[i].pos = positions[i]
            reply.motors[i].tor = m.tor
        if pose_frames>=3 and mode=='nan':
            reply.motors[motor_ids[0]-1].pos = float('nan')
        if pose_frames>=3 and mode=='error':
            reply.motors[motor_ids[0]-1].error = 7
        pub.publish(reply)
        glove = Float64MultiArray()
        glove_values = [0.0]*48
        glove_values[glove_slice] = q_reference
        manus_mode = mode.startswith('manus')
        if manus_mode:
            glove_values[glove_slice] = operator_zero  # Operator zero differs from robot/model zero.
            if mode=='manus_zero_motion' and command_frames<40:
                glove_values[glove_slice.stop-2] += 10*(command_frames%2)
            if pose_frames>=1:
                glove_values[glove_slice] = [a+b for a,b in zip(operator_zero,operator_delta)]
            if mode=='manus_outside_ranges' and pose_frames>=8:
                glove_values[glove_slice.start] = 34.44  # dq0=+20.44, beyond old joint and motor ranges.
                if pose_frames>=12:
                    glove_values[glove_slice] = [a+b for a,b in zip(operator_zero,[20.44,100,100,100]*(q_count//4))]
            if mode=='manus_invalid' and pose_frames>=8:
                glove_values[glove_slice.start+1] = float('nan')
        glove.data = glove_values
        if mode!='manus_dropout' or pose_frames<8:
            if manus_mode:
                raw = Float64MultiArray()
                raw.data = [0.0]*176
                if mode=='manus_wrong_hand' or (mode=='manus_right_only' and pose_frames>=8):
                    raw.data[0] = 1.0
                raw_pub.publish(raw)
            glove_pub.publish(glove)
        if ((mode in ('manus','manus_zero_motion') and pose_frames>=12) or (mode=='manus_outside_ranges' and pose_frames>=20)
                or (mode=='manus_long_run' and pose_frames>=110)):
            os.kill(os.getppid(),signal.SIGTERM)
        if mode=='topic':
            target = Float64MultiArray()
            target.data = [0.0]*q_count
            target_pub.publish(target)
            if pose_frames>=5:
                os.kill(os.getppid(),signal.SIGTERM)

    node.create_subscription(YHandMsg,sys.argv[3],command,10)
    record('ready')
    try:
        while not stopping:
            rclpy.spin_once(node,timeout_sec=.01)
    finally:
        record('driver_stopped')
        node.destroy_node()
        rclpy.shutdown()


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.mapper = control.LittleFingerMapper()
        self.defaults = control.load_defaults(PACKAGE/'scripts/muscle_adjustment_defaults.py')

    def test_calibrated_motor_zero_for_both_models(self):
        reference = [-5.0,-9.0,7.0,-10.0]
        self.assertEqual(list(self.mapper.q_reference),reference)
        self.assertTrue(self.mapper.config['zero_position_constraint'])
        for model in ['linear']:
            for angles,relative in [(reference,False),([0,0,0,0],True)]:
                self.assertEqual(self.mapper.predict(angles,relative=relative,model=model),
                                 {f'M{mid}':0.0 for mid in control.MOTOR_IDS})
                target = control.prepare_target(self.mapper,angles,relative,model)
                self.assertEqual(target['positions'],{mid:0.0 for mid in control.MOTOR_IDS})

    def test_raw_relative_and_side_input_contract(self):
        for model in ['linear']:
            raw = control.prepare_target(self.mapper,self.mapper.q_reference,False,model)
            delta = control.prepare_target(self.mapper,[0,0,0,0],True,model)
            self.assertEqual(raw,delta)
            other = control.prepare_target(self.mapper,[0,10,20,5],True,model)
            for mid in (18,20):
                self.assertEqual(delta['positions'][mid],other['positions'][mid])
            self.assertNotEqual(delta['positions'][11],other['positions'][11])

    def test_tension_and_position_only_affect_little(self):
        idle = control.current_matrix(self.defaults)
        target = control.prepare_target(self.mapper,[0,0,0,0],True)
        active = control.position_matrix(self.defaults,target['positions'],-.08)
        self.assertEqual(len(active),28)
        for mid in range(1,29):
            if mid in control.MOTOR_IDS:
                self.assertEqual(idle[mid-1],[0.0]*5)
                self.assertEqual(active[mid-1],[target['positions'][mid],0.0,-.08,
                                               self.defaults['kp'][mid-1],self.defaults['kd'][mid-1]])
            else:
                self.assertEqual(active[mid-1],idle[mid-1])
                self.assertEqual(active[mid-1],[0.0,0.0,self.defaults['tor'][mid-1],0.0,0.0])

    def test_rejects_invalid_input_and_nonfinite_predictions(self):
        for values in [[0,0,0],[0,0,0,0,0],[math.nan,0,0,0],[0,math.inf,0,0],[1e308]*4]:
            with self.assertRaises(ValueError):
                control.prepare_target(self.mapper,values,True)

    def test_joint_angles_remain_unlimited_and_motor_targets_are_bounded(self):
        cases = [[20.44,10,20,5],[-40,10,20,5],[-25,10,20,5]]
        for j in range(4):
            for value in [-1000,1000]:
                angles = [0.0]*4
                angles[j] = value
                cases.append(angles)
        for model in ['linear']:
            for angles in cases:
                target = control.prepare_target(self.mapper,angles,True,model)
                self.assertEqual(target['q_relative'],angles)
                expected = self.mapper.predict(angles,model=model)
                matrix = control.position_matrix(self.defaults,target['positions'],-.08)
                for mid in control.MOTOR_IDS:
                    self.assertEqual(target['predicted_positions'][mid],expected[f'M{mid}'])
                    lo,hi = control.MOTOR_POSITION_LIMITS_DEG[mid]
                    self.assertLessEqual(lo,target['positions'][mid])
                    self.assertLessEqual(target['positions'][mid],hi)
                    self.assertEqual(matrix[mid-1][0],target['positions'][mid])

    def test_motor_clamping_endpoints_and_unclipped_prediction_logs(self):
        cases = [([-6,-401,-16,-81],[-5,-400,-15,-80],list(control.MOTOR_IDS)),
                 ([451,21,51,31],[450,20,50,30],list(control.MOTOR_IDS)),
                 ([-5,-400,-15,-80],[-5,-400,-15,-80],[]),
                 ([450,20,50,30],[450,20,50,30],[]),
                 ([10,-10,5,-5],[10,-10,5,-5],[])]
        for prediction,expected,clipped in cases:
            raw = dict(zip(control.MOTOR_IDS,prediction))
            with patch.object(self.mapper,'predict',return_value={f'M{mid}':v for mid,v in raw.items()}):
                target = control.prepare_target(self.mapper,[0]*4)
            self.assertEqual(target['predicted_positions'],raw)
            self.assertEqual(target['positions'],dict(zip(control.MOTOR_IDS,expected)))
            self.assertEqual(target['clipped_motors'],clipped)
            # The final SDK frame also clamps an unclipped caller's values.
            matrix = control.position_matrix(self.defaults,raw,-.08)
            self.assertEqual([matrix[mid-1][0] for mid in control.MOTOR_IDS],expected)

    def test_parser_and_dry_run_need_no_ros(self):
        self.assertEqual(control.parse_angles('【0；10，20 5】'),[0,10,20,5])
        run = subprocess.run([sys.executable,str(SCRIPT),'--dry-run','--q','0','0','0','0'],
                             text=True,capture_output=True,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr)
        data = json.loads(run.stdout)
        self.assertEqual(len(data['mit_matrix']),28)
        self.assertEqual(list(data['motor_positions']),['M11','M13','M18','M20'])
        self.assertEqual(data['q_relative'],[0.0]*4)
        self.assertEqual(data['q_glove'],[-5.0,-9.0,7.0,-10.0])
        self.assertEqual(data['motor_positions'],{f'M{mid}':0.0 for mid in control.MOTOR_IDS})
        self.assertEqual(self.mapper.predict([0,0,0,0]),data['motor_positions'])
        self.assertEqual(control.prepare_target(self.mapper,[0,0,0,0])['positions'],
                         {mid:0.0 for mid in control.MOTOR_IDS})
        inference = subprocess.run([sys.executable,str(PACKAGE/'scripts/q_2_m linear mapping/left_little_q_to_motor.py'),
                                    '--q','0','0','0','0','--json'],
                                   text=True,capture_output=True,timeout=10)
        self.assertEqual(inference.returncode,0,inference.stderr)
        self.assertEqual(json.loads(inference.stdout),data['motor_positions'])

    def test_manus_relative_zero_is_latched_once(self):
        source = control.ManusRelativeInput(1.0)
        self.assertIsNone(source.update([10,20,30,40],0))
        self.assertIsNone(source.update([12,22,32,42],.5))
        self.assertEqual(source.update([14,24,34,44],1),[2.0]*4)
        self.assertEqual(source.reference,[12,22,32,42])
        self.assertEqual(source.update([11,32,52,47],2),[-1,10,20,5])
        self.assertEqual(source.reference,[12,22,32,42])
        source.invalidate()
        self.assertIsNone(source.latest_q)
        self.assertEqual(source.reference,[12,22,32,42])
        waiting = control.ManusRelativeInput(1.0)
        waiting.update([0]*4,0)
        waiting.invalidate()
        self.assertIsNone(waiting.update([10]*4,2))
        self.assertEqual(waiting.samples,[(2,[10.0]*4)])

    def test_manus_zero_waits_until_every_joint_is_stable(self):
        source = control.ManusRelativeInput(1,max_range_deg=3)
        self.assertIsNone(source.update([0,0,0,0],0))
        self.assertIsNone(source.update([0,0,90,0],.5))
        self.assertIsNone(source.update([0,0,0,0],1))
        self.assertIsNone(source.reference)
        self.assertEqual(source.rejected_windows,2)
        self.assertIsNone(source.update([1,1,1,1],1.5))
        self.assertEqual(source.update([0,0,0,0],2),[-1/3]*4)
        self.assertEqual(source.reference,[1/3]*4)
        self.assertEqual(source.reference_range,[1]*4)
        self.assertEqual(source.update([10,20,30,40],3),[10-1/3,20-1/3,30-1/3,40-1/3])
        self.assertEqual(source.reference,[1/3]*4)

    def test_manus_zero_stability_boundary_and_stale_sample_reset(self):
        source = control.ManusRelativeInput(1,max_range_deg=3)
        source.update([0]*4,0)
        source.update([3]*4,.5)
        self.assertEqual(source.update([3]*4,1),[1]*4)
        self.assertEqual(source.reference_range,[3]*4)
        waiting = control.ManusRelativeInput(1,max_range_deg=3)
        waiting.update([0]*4,0)
        waiting.invalidate()
        self.assertIsNone(waiting.update([0]*4,2))
        self.assertIsNone(waiting.reference)
        self.assertEqual(waiting.update([0]*4,3),[0]*4)

    def test_manus_packet_selection_matches_training_angles(self):
        values = list(range(48))
        self.assertEqual(control.left_manus_q(values),[24,25,26,27])
        with self.assertRaises(ValueError):
            control.left_manus_q(values[:47])
        values[25] = math.nan
        with self.assertRaises(ValueError):
            control.left_manus_q(values)
        left = [0.0]*176
        right = [1.0]+[0.0]*175
        self.assertTrue(control.has_left_manus_skeleton(left))
        self.assertTrue(control.has_left_manus_skeleton(right+left))
        self.assertFalse(control.has_left_manus_skeleton(right))
        self.assertFalse(control.has_left_manus_skeleton(left[:-1]))
        left[4] = math.nan
        self.assertFalse(control.has_left_manus_skeleton(left))


class RingPlanningTests(unittest.TestCase):
    def setUp(self):
        self.mapper = control.FingerMapper(finger='ring')
        self.defaults = control.load_defaults(PACKAGE/'scripts/muscle_adjustment_defaults.py')

    def test_ring_models_reference_and_side_feature_contract(self):
        self.assertEqual(self.mapper.motor_ids,(10,17,22,21))
        self.assertEqual(self.mapper.q_reference,(-2,-6,0,-12))
        self.assertEqual(control.left_manus_q(list(range(48)),'ring'),[20,21,22,23])
        for model in ['linear']:
            zero = self.mapper.predict([0]*4,model=model)
            self.assertEqual(zero,dict.fromkeys(['M10','M17','M22','M21'],0.0))
            self.assertEqual(zero,self.mapper.predict(self.mapper.q_reference,relative=False,model=model))
            moved = self.mapper.predict([0,10,20,5],model=model)
            self.assertNotEqual(moved['M22'],0)
            self.assertNotEqual(moved['M21'],0)
            self.assertNotEqual(moved['M10'],0)
            self.assertNotEqual(moved['M17'],0)
            raw = [v+r for v,r in zip([0,10,20,5],self.mapper.q_reference)]
            self.assertEqual(moved,self.mapper.predict(raw,relative=False,model=model))

    def test_ring_four_q_side_mapping_is_linear_and_neutral_aligned(self):
        # Independently recorded slopes from the 2170-point affine analysis.
        slopes={22:[-1.2730498465586373,.6582796919042382,-.07069193848395074,-.08362085406108608],
                21:[1.3639031607010055,-.347486996429509,.04464212963110632,-.008604078486167075]}
        for angles in [[-4,10,40,15],[0,10,0,0],[0,0,20,0],[0,0,0,10]]:
            for model in ['linear']:
                prediction = self.mapper.predict(angles,model=model)
                for mid,coefficients in slopes.items():
                    self.assertAlmostEqual(prediction[f'M{mid}'],sum(a*b for a,b in zip(angles,coefficients)),places=12)
        self.assertEqual(self.mapper.config['training_points_by_motor']['22'],2170)

    def test_ring_clamping_and_other24_defaults_include_little(self):
        raw = {10:500,17:-500,22:60,21:-100}
        matrix = control.position_matrix(self.defaults,raw,-.08,'ring')
        expected = {10:450,17:-400,22:25,21:-40}
        idle = control.current_matrix(self.defaults,'ring')
        for mid in range(1,29):
            if mid in expected:
                self.assertEqual(matrix[mid-1],[expected[mid],0,-.08,
                                               self.defaults['kp'][mid-1],self.defaults['kd'][mid-1]])
                self.assertEqual(idle[mid-1],[0]*5)
            else:
                self.assertEqual(matrix[mid-1],[0,0,self.defaults['tor'][mid-1],0,0])
        low = control.position_matrix(self.defaults,{10:-6,17:21,22:-16,21:31},-.08,'ring')
        self.assertEqual([low[mid-1][0] for mid in expected],[-5,20,-15,25])

    def test_ring_model_selection_and_cli_dry_run(self):
        with self.assertRaises(ValueError):
            control.FingerMapper(PACKAGE/'config/left_little_q_to_motor_models.json',finger='ring')
        result = subprocess.run([sys.executable,str(SCRIPT),'--finger','ring','--dry-run','--q','0','0','0','0'],
                                text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['finger'],'ring')
        self.assertEqual(data['q_glove'],[-2,-6,0,-12])
        self.assertEqual(data['motor_positions'],{'M10':0,'M17':0,'M22':0,'M21':0})
        self.assertEqual(data['release_currents_A'],{})
        self.assertEqual(data['side_release_duration_sec'],0)
        for mid in [22,21]:
            self.assertEqual(data['mit_matrix'][mid-1],[0,0,0,self.defaults['kp'][mid-1],self.defaults['kd'][mid-1]])
        for mid in [11,13,18,20]:
            self.assertEqual(data['mit_matrix'][mid-1],[0,0,self.defaults['tor'][mid-1],0,0])

    def test_ring_four_position_targets_continue_with_unlimited_joint_input(self):
        for model in ['linear']:
            results = []
            for angles in [[20,10,20,5],[-20,20,30,10],[0,0,0,0],[1000]*4,[-1000]*4]:
                target = control.prepare_target(self.mapper,angles,True,model)
                self.assertEqual(target['q_relative'],angles)
                self.assertEqual(set(target['positions']),{10,17,22,21})
                self.assertEqual(target['release_currents_A'],{})
                matrix = control.position_matrix(self.defaults,target['positions'],-.08,'ring')
                for mid,(lo,hi) in control.FINGER_MOTOR_POSITION_LIMITS_DEG['ring'].items():
                    self.assertGreaterEqual(matrix[mid-1][0],lo)
                    self.assertLessEqual(matrix[mid-1][0],hi)
                    self.assertEqual(matrix[mid-1][2:],[-.08,self.defaults['kp'][mid-1],self.defaults['kd'][mid-1]])
                results.append(target['positions'])
            self.assertNotEqual(results[0],results[1])
            self.assertEqual(results[2],dict.fromkeys(self.mapper.motor_ids,0))

    def test_cli_feedforward_defaults_and_explicit_override(self):
        for finger,extra,current in [('ring',[],0),('little',[],-.08),('ring',['--current','-0.03'],-.03)]:
            result = subprocess.run([sys.executable,str(SCRIPT),'--finger',finger,'--dry-run',
                                     '--q','-5','10','20','5',*extra],text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            data = json.loads(result.stdout)
            active = control.FingerMapper(finger=finger).motor_ids
            for mid,row in enumerate(data['mit_matrix'],1):
                if mid in active:
                    expected = (control.directional_side_current(row[0],0)
                                if finger=='ring' and mid in [22,21] else current)
                    self.assertEqual(row[2],expected)
                    self.assertEqual(row[3:], [self.defaults['kp'][mid-1],self.defaults['kd'][mid-1]])
                else:
                    self.assertEqual(row,[0,0,self.defaults['tor'][mid-1],0,0])

    def test_side_bias_uses_error_direction_instead_of_target_sign(self):
        # Negative targets can still require paying out; positive targets can
        # require winding in. Direction is relative to measured position.
        cases=[(-20,-30,.05),(-30,-20,-.05),(10,20,-.05),(20,10,.05),
               (0,0,0),(.5,0,0),(-.5,0,0),(.501,0,.05),(-.501,0,-.05)]
        for target,actual,expected in cases:
            self.assertEqual(control.directional_side_current(target,actual),expected)
        for actual in [math.nan,math.inf]:
            with self.assertRaises(ValueError):
                control.directional_side_current(0,actual)

    def test_side_bias_is_recomputed_from_feedback_after_position_clipping(self):
        positions={10:100,17:-100,22:-20,21:40}
        first=control.position_matrix(self.defaults,positions,-.03,'ring',
                                      feedback_positions={22:-30,21:30})
        # Clipped M22=-15 > -30: release, M21=25 < 30: tighten.
        self.assertEqual(first[21],[ -15,0,.05,800,100])
        self.assertEqual(first[20],[ 25,0,-.05,800,100])
        # Same target, opposite measured side of it: the bias must reverse.
        second=control.position_matrix(self.defaults,positions,-.03,'ring',
                                       feedback_positions={22:-10,21:5})
        self.assertEqual(second[21][2],-.05)
        self.assertEqual(second[20][2],.05)
        arrived=control.position_matrix(self.defaults,positions,-.03,'ring',
                                        feedback_positions={22:-15,21:25})
        for mid in [22,21]:self.assertEqual(arrived[mid-1][2],0)
        for matrix in [first,second,arrived]:
            for mid in range(1,29):
                if mid in [10,17]:self.assertEqual(matrix[mid-1][2],-.03)
                elif mid not in [22,21]:self.assertEqual(matrix[mid-1],[0,0,self.defaults['tor'][mid-1],0,0])

    def test_ring_stop_zeros_four_channels_and_keeps_other24_defaults(self):
        matrix = control.current_matrix(self.defaults,'ring',release_sway=False)
        for mid in range(1,29):
            if mid in self.mapper.motor_ids:
                self.assertEqual(matrix[mid-1],[0]*5)
            else:
                self.assertEqual(matrix[mid-1],[0,0,self.defaults['tor'][mid-1],0,0])

    def test_ring_has_no_release_window_overwriting_side_position_targets(self):
        window = control.InitialSideRelease('ring')
        target = control.prepare_target(self.mapper,[-4,10,40,15])
        for now in [0,1,1.999,2,2.001,3,30]:
            matrix = control.position_matrix(self.defaults,target['positions'],-.08,'ring')
            self.assertEqual(window.apply(matrix,now),matrix)
            for mid in [21,22]:
                self.assertEqual(matrix[mid-1][0],target['positions'][mid])
                self.assertGreater(matrix[mid-1][3],0)
        self.assertIsNone(window.started_at)
        self.assertFalse(window.completed)


class MiddlePlanningTests(unittest.TestCase):
    def setUp(self):
        self.mapper = control.create_mapper('middle')
        self.defaults = control.load_defaults(PACKAGE/'scripts/muscle_adjustment_defaults.py')

    def test_five_motor_roles_manus_columns_and_neutral(self):
        self.assertEqual(self.mapper.motor_ids,(23,16,2,9,24))
        self.assertEqual(list(self.mapper.motor_roles.values()),['伸肌','浅屈肌','深屈肌','内收','外展'])
        self.assertEqual(control.left_manus_q(list(range(48)),'middle'),[16,17,18,19])
        self.assertFalse(self.mapper.config['reference_is_exact_zero_calibration'])
        self.assertEqual(self.mapper.config['training_points'],5889)
        for model in ['linear']:
            self.assertEqual(self.mapper.predict([0]*4,model=model),
                             {f'M{mid}':0.0 for mid in self.mapper.motor_ids})
            self.assertEqual(self.mapper.predict(self.mapper.q_reference,relative=False,model=model),
                             self.mapper.predict([0]*4,model=model))

    def test_reviewed_linear_slopes_used_as_incremental_positions(self):
        # Independent source values from the reviewed 5889-point analyses.
        slopes = {
            23:[-.240285827950276,1.736833748379735,1.470990540060083,1.244149286558528],
            16:[1.020289262325872,-1.586283636939883,-1.525096966360787,.064612310526444],
            2:[-.279927034843161,-1.478943781767162,-1.816393181481213,-1.190385960468595],
            9:[2.454986483641826,-.233850309626944,-.055759318247319,.037899134617371],
            24:[-2.239807650193766,-.234990832988493,.044941243611357,.013148191315889],
        }
        for q in [[10,0,0,0],[0,10,0,0],[0,0,10,0],[0,0,0,10],[-5,10,20,5]]:
            linear = self.mapper.predict(q)
            self.assertEqual(linear,self.mapper.predict(q,model='quadratic'))
            absolute = self.mapper.predict([a+b for a,b in zip(q,self.mapper.q_reference)],relative=False)
            for mid,b in slopes.items():
                expected = math.fsum(a*v for a,v in zip(b,q))
                self.assertAlmostEqual(linear[f'M{mid}'],expected,places=10)
                self.assertAlmostEqual(absolute[f'M{mid}'],expected,places=10)
            # The two side motors also compensate flexion, not just q0.
        self.assertNotEqual(self.mapper.predict([0,10,0,0])['M9'],0)
        self.assertNotEqual(self.mapper.predict([0,10,0,0])['M24'],0)

    def test_no_middle_limits_all_five_position_channels_and_other23_tension(self):
        self.assertEqual(control.FINGER_MOTOR_POSITION_LIMITS_DEG['middle'],{})
        for q in [[0]*4,[1000]*4,[-1000]*4]:
            target = control.prepare_target(self.mapper,q)
            self.assertEqual(target['q_relative'],q)
            self.assertEqual(target['positions'],target['predicted_positions'])
            self.assertEqual(target['clipped_motors'],[])
            frame = control.position_matrix(self.defaults,target['positions'],
                                            control.default_feedforward('middle'),'middle')
            stopped = control.current_matrix(self.defaults,'middle')
            self.assertEqual(len(frame),28)
            for mid in range(1,29):
                if mid in self.mapper.motor_ids:
                    self.assertEqual(frame[mid-1],[target['positions'][mid],0,0,
                                                  self.defaults['kp'][mid-1],self.defaults['kd'][mid-1]])
                    self.assertEqual(stopped[mid-1],[0]*5)
                else:
                    expected = [0,0,self.defaults['tor'][mid-1],0,0]
                    self.assertEqual(frame[mid-1],expected)
                    self.assertEqual(stopped[mid-1],expected)

    def test_cli_loads_middle_model_and_keeps_unlimited_predictions(self):
        command = [sys.executable,str(SCRIPT),'--finger','middle','--dry-run','--q','1000','1000','1000','1000']
        result = subprocess.run(command,text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['motor_position_limits_deg'],{})
        self.assertEqual(data['motor_positions'],data['motor_predictions'])
        self.assertEqual(data['clipped_motors'],[])
        self.assertEqual(set(data['motor_positions']),{f'M{mid}' for mid in self.mapper.motor_ids})
        for mid in self.mapper.motor_ids:
            self.assertEqual(data['mit_matrix'][mid-1][0],data['motor_predictions'][f'M{mid}'])
            self.assertEqual(data['mit_matrix'][mid-1][2],0)
        pure = subprocess.run([sys.executable,str(PACKAGE/'scripts/q_2_m linear mapping/left_little_q_to_motor.py'),
                               '--finger','middle','--json','--q','1000','1000','1000','1000'],
                              text=True,capture_output=True,timeout=10)
        self.assertEqual(pure.returncode,0,pure.stderr)
        self.assertEqual(json.loads(pure.stdout),data['motor_positions'])

    def test_unlimited_middle_still_rejects_nonfinite_values_and_incomplete_targets(self):
        for q in [[0]*3,[0]*5,[math.nan,0,0,0],[0,math.inf,0,0]]:
            with self.assertRaises(ValueError):
                control.prepare_target(self.mapper,q)
        bad = dict.fromkeys(self.mapper.motor_ids,0.0)
        bad[24] = math.nan
        with self.assertRaises(ValueError):
            control.clamp_motor_positions(bad,'middle')
        del bad[24]
        with self.assertRaises(KeyError):
            control.clamp_motor_positions(bad,'middle')


class IndexPlanningTests(unittest.TestCase):
    def setUp(self):
        self.mapper = control.create_mapper('index')
        self.defaults = control.load_defaults(PACKAGE/'scripts/muscle_adjustment_defaults.py')

    def test_motor_roles_manus_columns_and_zero_reference(self):
        self.assertEqual(self.mapper.motor_ids,(25,3,15,1,28))
        self.assertEqual(list(self.mapper.motor_roles.values()),['伸肌','浅屈肌','深屈肌','内收','外展'])
        self.assertEqual(control.left_manus_q(list(range(48)),'index'),[12,13,14,15])
        self.assertEqual(self.mapper.config['training_points'],5606)
        self.assertFalse(self.mapper.config['reference_is_exact_zero_calibration'])
        for model in ['linear']:
            self.assertEqual(self.mapper.predict([0]*4,model=model),dict.fromkeys(['M25','M3','M15','M1','M28'],0))
            self.assertEqual(self.mapper.predict(self.mapper.q_reference,relative=False,model=model),self.mapper.predict([0]*4,model=model))

    def test_five_reviewed_four_q_slopes_and_incremental_coordinate_contract(self):
        coefficients = {
            25:[-.5122106284143217,1.817614305304914,1.467752913953902,.9282331488603762],
            3:[1.1272488209145868,-2.0787199572568613,-1.4627658022626469,.1261715228834114],
            15:[.16910393305268956,-2.0664404406540777,-1.7128276331060386,-1.1004042223038264],
            1:[2.065612664761241,-.1352024493728004,-.09967583403088713,-.21285099641829958],
            28:[-2.148852040698864,-.7555797182544404,.1375418966114658,.30823654283247837],
        }
        for q in [[10,0,0,0],[0,10,0,0],[0,0,10,0],[0,0,0,10],[-5,10,20,5]]:
            prediction = self.mapper.predict(q)
            self.assertEqual(prediction,self.mapper.predict(q,model='quadratic'))
            raw = self.mapper.predict([a+b for a,b in zip(q,self.mapper.q_reference)],relative=False)
            for mid,b in coefficients.items():
                self.assertAlmostEqual(prediction[f'M{mid}'],sum(a*v for a,v in zip(q,b)),places=11)
                self.assertAlmostEqual(raw[f'M{mid}'],prediction[f'M{mid}'],places=11)

    def test_all_five_unlimited_channels_current_policy_and_other23_defaults(self):
        for q in [[0]*4,[1000]*4,[-1000]*4]:
            plan = control.prepare_target(self.mapper,q)
            self.assertEqual(plan['positions'],plan['predicted_positions'])
            self.assertEqual(plan['clipped_motors'],[])
            frame = control.position_matrix(self.defaults,plan['positions'],control.default_feedforward('index'),'index')
            stop = control.current_matrix(self.defaults,'index')
            for mid in range(1,29):
                if mid in self.mapper.motor_ids:
                    self.assertEqual(frame[mid-1],[plan['positions'][mid],0,0,self.defaults['kp'][mid-1],self.defaults['kd'][mid-1]])
                    self.assertEqual(stop[mid-1],[0]*5)
                else:
                    self.assertEqual(frame[mid-1],[0,0,self.defaults['tor'][mid-1],0,0])
                    self.assertEqual(stop[mid-1],frame[mid-1])

    def test_both_cli_entry_points_load_index_without_ros(self):
        for script,options in [(SCRIPT,['--dry-run']),(PACKAGE/'scripts/q_2_m linear mapping/left_little_q_to_motor.py',['--json'])]:
            result = subprocess.run([sys.executable,str(script),'--finger','index',*options,'--q','-5','10','20','5'],text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            obj = json.loads(result.stdout)
            self.assertEqual(obj['motor_positions'] if script==SCRIPT else obj,self.mapper.predict([-5,10,20,5]))

    def test_invalid_inputs_and_wrong_finger_model_rejected(self):
        for q in [[0]*3,[0]*5,[0,math.nan,0,0],[math.inf,0,0,0]]:
            with self.assertRaises(ValueError):control.prepare_target(self.mapper,q)
        with self.assertRaises(ValueError):
            control.FingerMapper(PACKAGE/'config/left_middle_q_to_motor_models.json',finger='index')


class BothPlanningTests(unittest.TestCase):
    def setUp(self):
        self.mapper=control.create_mapper('both')
        self.defaults=control.load_defaults(PACKAGE/'scripts/muscle_adjustment_defaults.py')

    def test_models_are_independent_and_absolute_coordinates_match(self):
        ring=control.FingerMapper(finger='ring');little=control.FingerMapper(finger='little')
        angles=[-5,10,20,5,2,8,15,4]
        for model in ['linear']:
            result=self.mapper.predict(angles,model=model)
            self.assertEqual(result,{**ring.predict(angles[:4],model=model),**little.predict(angles[4:],model=model)})
            raw=[a+b for a,b in zip(angles,self.mapper.q_reference)]
            self.assertEqual(result,self.mapper.predict(raw,relative=False,model=model))
            changed=self.mapper.predict([0]*4+angles[4:],model=model)
            for mid in little.motor_ids:self.assertEqual(result[f'M{mid}'],changed[f'M{mid}'])
        self.assertEqual(control.left_manus_q(list(range(48)),'both'),list(range(20,28)))
        self.assertEqual(self.mapper.motor_ids,(10,17,22,21,11,13,18,20))
        self.assertEqual(set(self.mapper.predict([0]*8).values()),{0})

    def test_one_frame_keeps_each_finger_policy_and_other20_tension(self):
        plan=control.prepare_target(self.mapper,[-5,10,20,5,2,8,15,4])
        feedback=dict.fromkeys(self.mapper.motor_ids,0.)
        frame=control.position_matrix(self.defaults,plan['positions'],control.default_feedforward('both'),
                                       'both',feedback_positions=feedback)
        for mid in range(1,29):
            row=frame[mid-1]
            if mid in self.mapper.motor_ids:
                expected=(control.directional_side_current(row[0],0) if mid in [22,21] else 0 if mid in [10,17] else -.08)
                self.assertEqual(row[2],expected)
                self.assertEqual(row[3:],[self.defaults['kp'][mid-1],self.defaults['kd'][mid-1]])
            else:self.assertEqual(row,[0,0,self.defaults['tor'][mid-1],0,0])
        stop=control.current_matrix(self.defaults,'both')
        for mid in self.mapper.motor_ids:self.assertEqual(stop[mid-1],[0]*5)

    def test_both_cli_dry_run_and_wrong_angle_count(self):
        args=[sys.executable,str(SCRIPT),'--finger','both','--dry-run','--q']
        run=subprocess.run(args+['0']*8,text=True,capture_output=True,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr);data=json.loads(run.stdout)
        self.assertEqual(len(data['motor_positions']),8)
        self.assertEqual(set(data['motor_positions'].values()),{0})
        self.assertEqual(len(data['mit_matrix']),28)
        for mid in self.mapper.motor_ids:
            self.assertEqual(data['mit_matrix'][mid-1][2],0 if mid in [10,17,22,21] else -.08)
        invalid=subprocess.run(args+['0']*4,text=True,capture_output=True,timeout=10)
        self.assertNotEqual(invalid.returncode,0)

    def test_both_reference_requires_all_eight_angles_to_stabilize(self):
        source=control.ManusRelativeInput(1,max_range_deg=3,size=8)
        self.assertIsNone(source.update([0]*8,0))
        moving=[0]*7+[10]
        self.assertIsNone(source.update(moving,.5))
        self.assertIsNone(source.update([0]*8,1))
        self.assertIsNone(source.reference)
        self.assertEqual(source.update([0]*8,2),[0]*8)
        self.assertEqual(source.reference_range,[0]*8)


class ThreePlanningTests(unittest.TestCase):
    def setUp(self):
        self.mapper = control.create_mapper('three')
        self.defaults = control.load_defaults(PACKAGE/'scripts/muscle_adjustment_defaults.py')

    def test_independent_twelve_angles_and_all_thirteen_motor_outputs(self):
        self.assertEqual(self.mapper.motor_ids,(23,16,2,9,24,10,17,22,21,11,13,18,20))
        self.assertEqual(control.left_manus_q(list(range(48)),'three'),list(range(16,28)))
        angles = [-5,10,20,5]+[-4,15,30,8]+[2,8,15,4]
        for model in ['linear']:
            result = self.mapper.predict(angles,model=model)
            expected = {}
            for i,finger in enumerate(['middle','ring','little']):
                expected.update(control.create_mapper(finger).predict(angles[4*i:4*i+4],model=model))
            self.assertEqual(result,expected)
            raw = [a+b for a,b in zip(angles,self.mapper.q_reference)]
            actual = self.mapper.predict(raw,relative=False,model=model)
            for mid in result:self.assertAlmostEqual(actual[mid],result[mid],places=10)
        self.assertEqual(self.mapper.motor_inputs[23],[0,1,2,3])
        self.assertEqual(self.mapper.motor_inputs[22],[4,5,6,7])
        self.assertEqual(self.mapper.motor_inputs[18],[8])
        with self.assertRaises(ValueError):self.mapper.predict([0]*8)

    def test_one_frame_preserves_partial_limits_currents_and_other15_tension(self):
        q = [1000]*12
        target = control.prepare_target(self.mapper,q)
        self.assertTrue(target['clipped_motors'])
        self.assertFalse(set(target['clipped_motors']) & {23,16,2,9,24})
        frame = control.position_matrix(self.defaults,target['positions'],control.default_feedforward('three'),
                                        'three',feedback_positions=dict.fromkeys(self.mapper.motor_ids,0.0))
        stop = control.current_matrix(self.defaults,'three')
        limits = control.FINGER_MOTOR_POSITION_LIMITS_DEG['three']
        self.assertEqual(len(limits),8)
        for mid in range(1,29):
            if mid not in self.mapper.motor_ids:
                self.assertEqual(frame[mid-1],[0,0,self.defaults['tor'][mid-1],0,0])
                self.assertEqual(stop[mid-1],frame[mid-1])
                continue
            self.assertEqual(stop[mid-1],[0]*5)
            self.assertEqual(frame[mid-1][0],target['positions'][mid])
            self.assertEqual(frame[mid-1][3:],[self.defaults['kp'][mid-1],self.defaults['kd'][mid-1]])
            if mid in [22,21]:
                self.assertEqual(frame[mid-1][2],control.directional_side_current(target['positions'][mid],0))
            else:self.assertEqual(frame[mid-1][2],-.08 if mid in [11,13,18,20] else 0)
            if mid in limits:
                lo,hi = limits[mid]
                self.assertLessEqual(lo,frame[mid-1][0]);self.assertLessEqual(frame[mid-1][0],hi)
            else:self.assertEqual(frame[mid-1][0],target['predicted_positions'][mid])

    def test_three_cli_and_all_twelve_zero_angles_must_stabilize(self):
        command = [sys.executable,str(SCRIPT),'--finger','three','--dry-run','--q']
        run = subprocess.run(command+['0']*12,text=True,capture_output=True,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr)
        data = json.loads(run.stdout)
        self.assertEqual(len(data['motor_positions']),13)
        self.assertEqual(set(data['model_files']),{'middle','ring','little'})
        self.assertEqual(set(data['motor_positions'].values()),{0})
        self.assertEqual(len(data['mit_matrix']),28)
        invalid = subprocess.run(command+['0']*8,text=True,capture_output=True,timeout=10)
        self.assertNotEqual(invalid.returncode,0)
        source = control.ManusRelativeInput(1,max_range_deg=3,size=12)
        for moving_index in [0,6,11]:
            source.invalidate()
            source.update([0]*12,0)
            moving = [0]*12;moving[moving_index] = 10
            self.assertIsNone(source.update(moving,.5))
            self.assertIsNone(source.update([0]*12,1))
            self.assertIsNone(source.reference)
        self.assertEqual(source.update([0]*12,2),[0]*12)


class FourPlanningTests(unittest.TestCase):
    def setUp(self):
        self.mapper = control.create_mapper('four')
        self.defaults = control.load_defaults(PACKAGE/'scripts/muscle_adjustment_defaults.py')

    def test_four_independent_groups_correct_columns_and_eighteen_unique_motors(self):
        self.assertEqual(self.mapper.motor_ids,(25,3,15,1,28,23,16,2,9,24,10,17,22,21,11,13,18,20))
        self.assertEqual(control.left_manus_q(list(range(48)),'four'),list(range(12,28)))
        q = [-5,10,20,5]+[2,15,25,8]+[-3,10,15,4]+[1,8,12,2]
        for model in ['linear']:
            expected = {}
            for i,f in enumerate(['index','middle','ring','little']):
                expected.update(control.create_mapper(f).predict(q[i*4:i*4+4],model=model))
            self.assertEqual(self.mapper.predict(q,model=model),expected)
        self.assertEqual(self.mapper.motor_inputs[25],[0,1,2,3])
        self.assertEqual(self.mapper.motor_inputs[23],[4,5,6,7])
        self.assertEqual(self.mapper.motor_inputs[22],[8,9,10,11])
        self.assertEqual(self.mapper.motor_inputs[18],[12])
        with self.assertRaises(ValueError):self.mapper.predict([0]*12)

    def test_four_shared_frame_preserves_each_current_limit_and_other10_tension(self):
        plan = control.prepare_target(self.mapper,[1000]*16)
        limits = control.FINGER_MOTOR_POSITION_LIMITS_DEG['four']
        self.assertEqual(len(limits),8)
        frame = control.position_matrix(self.defaults,plan['positions'],control.default_feedforward('four'),
                                        'four',feedback_positions=dict.fromkeys(self.mapper.motor_ids,0))
        stop = control.current_matrix(self.defaults,'four')
        for mid in range(1,29):
            if mid not in self.mapper.motor_ids:
                self.assertEqual(frame[mid-1],[0,0,self.defaults['tor'][mid-1],0,0])
                self.assertEqual(stop[mid-1],frame[mid-1]);continue
            self.assertEqual(frame[mid-1][0],plan['positions'][mid])
            self.assertEqual(frame[mid-1][3:],[self.defaults['kp'][mid-1],self.defaults['kd'][mid-1]])
            self.assertEqual(stop[mid-1],[0]*5)
            if mid in limits:
                lo,hi = limits[mid];self.assertLessEqual(lo,frame[mid-1][0]);self.assertLessEqual(frame[mid-1][0],hi)
            else:self.assertEqual(plan['positions'][mid],plan['predicted_positions'][mid])
            ff = control.directional_side_current(plan['positions'][mid],0) if mid in [22,21] else -.08 if mid in [11,13,18,20] else 0
            self.assertEqual(frame[mid-1][2],ff)
        self.assertEqual(frame[1][3:],[500,50])  # Preserve the requested middle M2 tuning.
        self.assertEqual(frame[14][3:],[200,30])

    def test_cli_and_reference_wait_for_all_sixteen_angles(self):
        result = subprocess.run([sys.executable,str(SCRIPT),'--finger','four','--dry-run','--q',*(['0']*16)],text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        obj = json.loads(result.stdout)
        self.assertEqual(len(obj['motor_positions']),18)
        self.assertEqual(set(obj['model_files']),{'index','middle','ring','little'})
        self.assertEqual(set(obj['motor_positions'].values()),{0})
        for j in [0,6,10,15]:
            reference = control.ManusRelativeInput(1,max_range_deg=3,size=16)
            reference.update([0]*16,0)
            moving=[0]*16;moving[j]=10
            self.assertIsNone(reference.update(moving,.5))
            self.assertIsNone(reference.update([0]*16,1))
            self.assertIsNone(reference.reference)
            self.assertEqual(reference.update([0]*16,2),[0]*16)


class LaunchWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory(prefix='little-launch-log-')
        cls.addClassCleanup(directory.cleanup)
        environment = patch.dict(os.environ,ROS_LOG_DIR=directory.name)
        environment.start()
        cls.addClassCleanup(environment.stop)

    def test_one_launch_starts_headless_controller_and_input_window(self):
        from launch import LaunchContext
        from launch.actions import ExecuteProcess,RegisterEventHandler,TimerAction
        from launch.utilities import perform_substitutions
        spec = importlib.util.spec_from_file_location('little_launch',PACKAGE/'launch/left_hand_q_to_motor_launch.py')
        launch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch_module)
        context = LaunchContext()
        context.launch_configurations.update(finger='little',input_mode='1',relative='true',model='linear',start_driver_node='true',
                                             start_angle_input_terminal='true',joint_angle_topic='/little_target',
                                             current='-0.08',publish_hz='50.0')
        with patch.object(launch_module.shutil,'which',return_value='/fake/gnome-terminal'):
            actions = launch_module._launch_setup(context)
        processes = [a for a in actions if isinstance(a,ExecuteProcess)]
        self.assertEqual(len(processes),1)  # Only the controller owns/starts SDK.
        command = [perform_substitutions(context,part) for part in processes[0].cmd]
        self.assertIn(str(SCRIPT),command)
        self.assertIn('--relative',command)
        self.assertIn('--start-driver',command)
        self.assertEqual(command[command.index('--target-topic')+1],'/little_target')
        self.assertTrue(any(isinstance(a,RegisterEventHandler) for a in actions))
        timer = next(a for a in actions if isinstance(a,TimerAction))
        terminal = timer.actions[0]
        command = [perform_substitutions(context,part) for part in terminal.cmd]
        self.assertEqual(command[0],'/fake/gnome-terminal')
        self.assertEqual(command[command.index('--topic')+1],'/little_target')

    def test_headless_launch_reuses_driver_and_raw_q(self):
        from launch import LaunchContext
        from launch.actions import ExecuteProcess,TimerAction
        from launch.utilities import perform_substitutions
        spec = importlib.util.spec_from_file_location('little_launch',PACKAGE/'launch/left_hand_q_to_motor_launch.py')
        launch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch_module)
        context = LaunchContext()
        context.launch_configurations.update(finger='little',input_mode='1',relative='false',model='linear',start_driver_node='false',
                                             start_angle_input_terminal='false',joint_angle_topic='/little_target',
                                             current='-0.08',publish_hz='50.0')
        actions = launch_module._launch_setup(context)
        process = next(a for a in actions if isinstance(a,ExecuteProcess))
        command = [perform_substitutions(context,part) for part in process.cmd]
        self.assertNotIn('--relative',command)
        self.assertIn('--absolute',command)
        self.assertIn('--no-start-driver',command)
        self.assertIn('linear',command)
        self.assertFalse(any(isinstance(a,TimerAction) for a in actions))

    def test_manus_launch_starts_publisher_without_input_terminal(self):
        from launch import LaunchContext
        from launch.actions import ExecuteProcess,TimerAction
        from launch.utilities import perform_substitutions
        spec = importlib.util.spec_from_file_location('little_launch',PACKAGE/'launch/left_hand_q_to_motor_launch.py')
        launch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch_module)
        context = LaunchContext()
        context.launch_configurations.update(finger='little',input_mode='2',relative='true',model='linear',start_driver_node='true',
                                             start_angle_input_terminal='true',start_glove_node='true',
                                             current='-0.08',publish_hz='50.0',manus_topic='/manus_angles',
                                             manus_raw_topic='/manus_raw',manus_zero_seconds='1.0',manus_timeout='0.5')
        with patch.object(launch_module.shutil,'which') as which:
            actions = launch_module._launch_setup(context)
            which.assert_not_called()
        processes = [a for a in actions if isinstance(a,ExecuteProcess)]
        commands = [[perform_substitutions(context,part) for part in p.cmd] for p in processes]
        self.assertEqual(len(commands),2)
        self.assertTrue(any(str(PACKAGE/'scripts/minimal_example_dual_stream_commented.py') in c for c in commands))
        controller = next(c for c in commands if str(SCRIPT) in c)
        self.assertEqual(controller[controller.index('--input-mode')+1],'manus')
        self.assertEqual(controller[controller.index('--manus-topic')+1],'/manus_angles')
        self.assertIn('--relative',controller)
        self.assertNotIn('--target-topic',controller)
        self.assertFalse(any(isinstance(a,TimerAction) for a in actions))
        context.launch_configurations['start_glove_node'] = 'false'
        self.assertEqual(len([a for a in launch_module._launch_setup(context) if isinstance(a,ExecuteProcess)]),1)
        context.launch_configurations['relative'] = 'false'
        with self.assertRaises(ValueError):
            launch_module._launch_setup(context)

    def test_ring_flag_selects_controller_terminal_and_default_topic(self):
        from launch import LaunchContext
        from launch.actions import ExecuteProcess,TimerAction
        from launch.utilities import perform_substitutions
        spec = importlib.util.spec_from_file_location('ring_launch',PACKAGE/'launch/left_hand_q_to_motor_launch.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        context = LaunchContext()
        context.launch_configurations.update(finger='ring',input_mode='1',relative='true',model='linear',
                                             start_driver_node='true',start_angle_input_terminal='true',
                                             joint_angle_topic='',current='',publish_hz='50.0')
        with patch.object(module.shutil,'which',return_value='/fake/gnome-terminal'):
            actions = module._launch_setup(context)
        process = next(a for a in actions if isinstance(a,ExecuteProcess))
        command = [perform_substitutions(context,part) for part in process.cmd]
        self.assertEqual(command[command.index('--finger')+1],'ring')
        self.assertEqual(command[command.index('--target-topic')+1],'/left_ring_target_q')
        self.assertNotIn('--current',command)  # Controller selects ring's 0 A default.
        terminal = next(a for a in actions if isinstance(a,TimerAction)).actions[0]
        command = [perform_substitutions(context,part) for part in terminal.cmd]
        self.assertEqual(command[command.index('--finger')+1],'ring')
        self.assertEqual(command[command.index('--topic')+1],'/left_ring_target_q')
        context.launch_configurations['current'] = '-0.03'
        with patch.object(module.shutil,'which',return_value='/fake/gnome-terminal'):
            explicit = next(a for a in module._launch_setup(context) if isinstance(a,ExecuteProcess))
        command = [perform_substitutions(context,part) for part in explicit.cmd]
        self.assertEqual(command[command.index('--current')+1],'-0.03')


    def test_both_manus_launch_uses_one_controller_and_glove_publisher(self):
        from launch import LaunchContext
        from launch.actions import ExecuteProcess,TimerAction
        from launch.utilities import perform_substitutions
        spec=importlib.util.spec_from_file_location('both_launch',PACKAGE/'launch/left_hand_q_to_motor_launch.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        context=LaunchContext()
        context.launch_configurations.update(finger='both',input_mode='2',relative='true',model='linear',
            start_driver_node='true',start_angle_input_terminal='true',start_glove_node='true',
            current='',publish_hz='50',manus_topic='/test_manus',manus_raw_topic='/test_raw',
            manus_zero_seconds='1',manus_timeout='.5')
        with patch.object(module.shutil,'which') as terminal:
            actions=module._launch_setup(context);terminal.assert_not_called()
        processes=[a for a in actions if isinstance(a,ExecuteProcess)]
        commands=[[perform_substitutions(context,part) for part in a.cmd] for a in processes]
        self.assertEqual(len(commands),2)
        controllers=[c for c in commands if str(SCRIPT) in c];self.assertEqual(len(controllers),1)
        c=controllers[0];self.assertEqual(c[c.index('--finger')+1],'both')
        self.assertIn('--start-driver',c);self.assertNotIn('--current',c)
        self.assertFalse(any(isinstance(a,TimerAction) for a in actions))


    def test_middle_launch_routes_manual_and_manus_to_same_five_motor_controller(self):
        from launch import LaunchContext
        from launch.actions import ExecuteProcess,TimerAction
        from launch.utilities import perform_substitutions
        spec = importlib.util.spec_from_file_location('middle_launch',PACKAGE/'launch/left_hand_q_to_motor_launch.py')
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        context = LaunchContext()
        context.launch_configurations.update(finger='middle',input_mode='1',relative='true',model='linear',
            start_driver_node='true',start_angle_input_terminal='true',start_glove_node='true',joint_angle_topic='',
            current='',publish_hz='50',manus_topic='/test_manus',manus_raw_topic='/test_raw',
            manus_zero_seconds='1',manus_timeout='.5')
        with patch.object(module.shutil,'which',return_value='/fake/gnome-terminal'):
            actions = module._launch_setup(context)
        controller = next(a for a in actions if isinstance(a,ExecuteProcess))
        command = [perform_substitutions(context,p) for p in controller.cmd]
        self.assertEqual(command[command.index('--finger')+1],'middle')
        self.assertEqual(command[command.index('--target-topic')+1],'/left_middle_target_q')
        self.assertNotIn('--current',command)
        terminal = next(a for a in actions if isinstance(a,TimerAction)).actions[0]
        command = [perform_substitutions(context,p) for p in terminal.cmd]
        self.assertEqual(command[command.index('--finger')+1],'middle')
        context.launch_configurations['input_mode'] = '2'
        with patch.object(module.shutil,'which') as terminal_lookup:
            actions = module._launch_setup(context);terminal_lookup.assert_not_called()
        processes = [a for a in actions if isinstance(a,ExecuteProcess)]
        self.assertEqual(len(processes),2)
        commands = [[perform_substitutions(context,p) for p in a.cmd] for a in processes]
        controller = next(c for c in commands if str(SCRIPT) in c)
        self.assertEqual(controller[controller.index('--finger')+1],'middle')
        self.assertEqual(controller[controller.index('--input-mode')+1],'manus')
        self.assertNotIn('--target-topic',controller)
        self.assertFalse(any(isinstance(a,TimerAction) for a in actions))


    def test_three_manus_has_one_controller_and_manual_requires_twelve_angle_topic(self):
        from launch import LaunchContext
        from launch.actions import ExecuteProcess,TimerAction
        from launch.utilities import perform_substitutions
        spec = importlib.util.spec_from_file_location('three_launch',PACKAGE/'launch/left_hand_q_to_motor_launch.py')
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        context = LaunchContext()
        context.launch_configurations.update(finger='three',input_mode='2',relative='true',model='linear',
            start_driver_node='true',start_angle_input_terminal='true',start_glove_node='true',joint_angle_topic='',
            current='',publish_hz='50',manus_topic='/test_manus',manus_raw_topic='/test_raw',
            manus_zero_seconds='1',manus_timeout='.5')
        with patch.object(module.shutil,'which') as terminal:
            actions = module._launch_setup(context);terminal.assert_not_called()
        commands = [[perform_substitutions(context,p) for p in a.cmd] for a in actions if isinstance(a,ExecuteProcess)]
        self.assertEqual(len(commands),2)
        controllers = [c for c in commands if str(SCRIPT) in c]
        self.assertEqual(len(controllers),1)
        self.assertEqual(controllers[0][controllers[0].index('--finger')+1],'three')
        self.assertFalse(any(isinstance(a,TimerAction) for a in actions))
        context.launch_configurations['input_mode'] = '1'
        with self.assertRaisesRegex(ValueError,'12角'):
            module._launch_setup(context)
        context.launch_configurations['start_angle_input_terminal'] = 'false'
        actions = module._launch_setup(context)
        controller = next(a for a in actions if isinstance(a,ExecuteProcess))
        command = [perform_substitutions(context,p) for p in controller.cmd]
        self.assertEqual(command[command.index('--target-topic')+1],'/left_three_target_q')


    def test_index_and_four_launch_select_the_matching_input_and_controller(self):
        from launch import LaunchContext
        from launch.actions import ExecuteProcess,TimerAction
        from launch.utilities import perform_substitutions
        spec=importlib.util.spec_from_file_location('index_four_launch',PACKAGE/'launch/left_hand_q_to_motor_launch.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for finger in ['index','four','thumb','five']:
            ctx=LaunchContext()
            ctx.launch_configurations.update(finger=finger,input_mode='2',relative='true',model='linear',
                start_driver_node='true',start_angle_input_terminal='true',start_glove_node='true',joint_angle_topic='',
                current='',publish_hz='50',manus_topic='/test_manus',manus_raw_topic='/test_raw',manus_zero_seconds='1',manus_timeout='.5')
            with patch.object(module.shutil,'which') as terminal:
                actions=module._launch_setup(ctx);terminal.assert_not_called()
            commands=[[perform_substitutions(ctx,p) for p in a.cmd] for a in actions if isinstance(a,ExecuteProcess)]
            self.assertEqual(len(commands),2)
            ctrl=next(c for c in commands if str(SCRIPT) in c)
            self.assertEqual(ctrl[ctrl.index('--finger')+1],finger)
            self.assertNotIn('--current',ctrl)
            ctx.launch_configurations['input_mode']='1'
            if finger in ('four','five'):
                with self.assertRaisesRegex(ValueError,'16角' if finger=='four' else '20角'):module._launch_setup(ctx)
                ctx.launch_configurations['start_angle_input_terminal']='false'
            with patch.object(module.shutil,'which',return_value='/fake/terminal'):
                actions=module._launch_setup(ctx)
            ctrl=next(a for a in actions if isinstance(a,ExecuteProcess))
            command=[perform_substitutions(ctx,p) for p in ctrl.cmd]
            self.assertEqual(command[command.index('--target-topic')+1],f'/left_{finger}_target_q')
            if finger in ('index','thumb'):
                term=next(a for a in actions if isinstance(a,TimerAction)).actions[0]
                command=[perform_substitutions(ctx,p) for p in term.cmd]
                self.assertEqual(command[command.index('--finger')+1],finger)


class FakeSDKIntegrationTests(unittest.TestCase):
    def exercise(self,mode='normal',interrupt=False,signum=signal.SIGINT,finger='little',model='linear'):
        mapper = control.create_mapper(finger, model=model)
        motor_ids = mapper.motor_ids
        motor_limits = control.FINGER_MOTOR_POSITION_LIMITS_DEG[finger]
        q_count = len(mapper.q_reference)
        expected_current = ({str(mid):(-.08 if mid in [11,13,18,20] else 0) for mid in motor_ids}
                            if finger in ('both','three','four','five') else 0 if finger in ('ring','middle','index','thumb') else -.08)
        expected_for_mid = lambda mid:expected_current[str(mid)] if isinstance(expected_current,dict) else expected_current
        operator_zero,operator_delta = operator_angles(finger)
        try:
            from devices_pkg.msg import YHandMsg
        except ImportError:
            self.skipTest('source install/setup.bash to run isolated DDS tests')
        with tempfile.TemporaryDirectory(prefix='little-sdk-test-') as directory:
            root = Path(directory)
            fake = root/'fake_sdk'
            fake.write_text(f'#!{sys.executable}\nimport runpy\nrunpy.run_path({str(Path(__file__).resolve())!r},run_name="fake_little_sdk")\n')
            fake.chmod(0o755)
            topic = f'/little_test_{os.getpid()}'
            env = dict(os.environ,ROS_LOCALHOST_ONLY='1',ROS_DOMAIN_ID=str(100+os.getpid()%100),
                       ROS_LOG_DIR=str(root/'roslog'),LITTLE_TEST_EVENTS=str(root/'events.jsonl'),
                       LITTLE_TEST_GLOVE_TOPIC=topic+'/glove',LITTLE_TEST_TARGET_TOPIC=topic+'/target',
                       LITTLE_TEST_RAW_TOPIC=topic+'/raw',
                       LITTLE_TEST_MODE=mode,LITTLE_TEST_FINGER=finger,PYTHONDONTWRITEBYTECODE='1')
            command = [sys.executable,str(SCRIPT),'--finger',finger,'--relative','--q',*(['0']*q_count),
                       '--model',model,
                       '--driver-executable',str(fake),'--cmd-topic',topic+'/cmd','--feedback-topic',topic+'/feedback',
                       '--glove-topic',topic+'/glove','--output-dir',str(root/'output'),
                       '--hold-seconds','30' if interrupt else '1' if mode=='dropout' else '.2',
                       '--feedback-timeout','.25','--ready-timeout','8']
            if mode=='topic':
                index = command.index('--q')
                del command[index:index+1+q_count]
                command += ['--target-topic',topic+'/target']
            if mode.startswith('manus'):
                index = command.index('--q')
                del command[index:index+1+q_count]
                command += ['--input-mode','manus','--manus-topic',topic+'/glove',
                            '--manus-raw-topic',topic+'/raw','--manus-zero-seconds','.1','--manus-timeout','.2']
                command[command.index('--ready-timeout')+1] = '3'
            process = subprocess.Popen(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
            try:
                if interrupt:
                    deadline = time.monotonic()+12
                    while time.monotonic()<deadline:
                        events = root/'events.jsonl'
                        if events.exists():
                            entries = [json.loads(line) for line in events.read_text().splitlines()]
                            if any(e['event']=='command' and e['matrix'][motor_ids[0]-1][3]>0 for e in entries):
                                break
                        if process.poll() is not None:
                            break
                        time.sleep(.02)
                    process.send_signal(signum)
                stdout,stderr = process.communicate(timeout=20)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            expected = 0 if mode in ('normal','topic','manus','manus_outside_ranges','manus_long_run','manus_zero_motion') else 1
            self.assertEqual(process.returncode,expected,stdout+'\n'+stderr)
            entries = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
            commands = [e for e in entries if e['event']=='command']
            self.assertTrue(commands,stdout+'\n'+stderr)
            defaults = control.load_defaults(PACKAGE/'scripts/muscle_adjustment_defaults.py')
            for entry in commands:
                self.assertFalse(entry['pretighten'])
                self.assertEqual(len(entry['matrix']),28)
                for mid in range(1,29):
                    if mid not in motor_ids:
                        actual = entry['matrix'][mid-1]
                        self.assertEqual(actual[:2],[0,0])
                        self.assertEqual(actual[3:],[0,0])
                        self.assertAlmostEqual(actual[2],defaults['tor'][mid-1],places=6)
                if finger in ('ring','both','three','four','five'):
                    for mid in [22,21]:
                        row = entry['matrix'][mid-1]
                        if row[3]>0:
                            self.assertLess(min(abs(row[2]-v) for v in [-.05,0,.05]),1e-6)
                        else:
                            self.assertEqual(row,[0]*5)
            active = [e for e in commands if e['matrix'][motor_ids[0]-1][3]>0]
            if finger in ('ring','both','three','four','five') and mode=='manus_long_run':
                self.assertGreater(active[-1]['time']-active[0]['time'],2)
                self.assertTrue(all(e['matrix'][20][3]>0 and e['matrix'][21][3]>0 for e in active))
            if mode=='manus_wrong_hand':
                self.assertFalse(active)
            else:
                self.assertTrue(active)
            def expected_positions(angles):
                planned = control.prepare_target(mapper,angles,model=model)
                return {f'M{mid}':value for mid,value in planned['positions'].items()}
            predicted = expected_positions([0]*q_count)
            moved = expected_positions(operator_delta)
            outside = expected_positions([20.44,10,20,5]+operator_delta[4:])
            saturated = expected_positions([20.44,100,100,100]*(q_count//4))
            if mode.startswith('manus') and mode!='manus_wrong_hand':
                self.assertTrue(any(abs(e['matrix'][motor_ids[0]-1][0]-moved[f'M{motor_ids[0]}'])<1e-4 for e in active))
            if mode=='manus_outside_ranges':
                self.assertTrue(any(abs(e['matrix'][motor_ids[3]-1][0]-outside[f'M{motor_ids[3]}'])<1e-4 for e in active))
                for mid in motor_ids:
                    # SDK message positions are float32; raw slopes are float64.
                    self.assertAlmostEqual(active[-1]['matrix'][mid-1][0],saturated[f'M{mid}'],places=4)
            for entry in active:
                for mid in motor_ids:
                    if finger in ('ring','both','three','four','five') and mid in [22,21]:
                        self.assertLess(min(abs(entry['matrix'][mid-1][2]-v) for v in [-.05,0,.05]),1e-6)
                    else:
                        self.assertAlmostEqual(entry['matrix'][mid-1][2],expected_for_mid(mid),places=6)
                    # First and later position frames equal the mapped target.
                    candidates = [predicted[f'M{mid}'],moved[f'M{mid}']] if mode.startswith('manus') else [predicted[f'M{mid}']]
                    if mode=='manus_outside_ranges':
                        candidates += [outside[f'M{mid}'],saturated[f'M{mid}']]
                    self.assertLess(min(abs(entry['matrix'][mid-1][0]-v) for v in candidates),1e-4)
                    if mid in motor_limits:
                        lo,hi = motor_limits[mid]
                        self.assertLessEqual(lo,entry['matrix'][mid-1][0])
                        self.assertLessEqual(entry['matrix'][mid-1][0],hi)
            # Final stop frames must reach the fake before its child process exits.
            for mid in motor_ids:
                self.assertEqual(commands[-1]['matrix'][mid-1],[0]*5)
            self.assertEqual(entries[-1]['event'],'driver_stopped')
            if finger in ('both','three','four','five'):
                self.assertEqual(sum(e['event']=='ready' for e in entries),1)
                if mode.startswith('manus') and mode not in ('manus_wrong_hand','manus_outside_ranges'):
                    for mid in motor_ids:
                        self.assertAlmostEqual(active[-1]['matrix'][mid-1][0],moved[f'M{mid}'],places=4)
            metadata = json.loads((root/'output/metadata.json').read_text())
            self.assertFalse(metadata['zero_coordinate_changed'])
            self.assertEqual(metadata['current_A'],expected_current)
            self.assertEqual(metadata['position_target_mode'],'direct')
            self.assertFalse(metadata['joint_range_checks'])
            self.assertEqual(metadata['motor_position_range_checks'],bool(motor_limits))
            self.assertEqual(metadata['motor_position_limit_action'],'clip' if motor_limits else 'none')
            self.assertEqual(metadata['motor_position_limits_deg'],
                             {str(mid):list(pair) for mid,pair in motor_limits.items()})
            self.assertEqual(metadata['finger'],finger)
            self.assertEqual(metadata['motor_ids'],list(motor_ids))
            self.assertEqual(metadata['position_motor_ids'],list(motor_ids))
            self.assertEqual(metadata['model'], model)
            if model == 'mlp':
                self.assertEqual(set(metadata['model_effective_models'].values()), {'mlp'})
                self.assertEqual(set(metadata['model_effective_degrees'].values()), {None})
            if finger in ('middle','index') and model != 'mlp':
                self.assertEqual(metadata['model_effective_degrees'],dict.fromkeys(map(str,motor_ids),1))
            if finger in ('thumb','five'):
                if model != 'mlp':
                    self.assertEqual({mid:metadata['model_effective_degrees'][mid] for mid in ['4','6','7','8','14','27']},
                                     {'4':1,'6':2,'7':1,'8':2,'14':2,'27':2})
                self.assertTrue(all(mid in metadata['unlimited_position_motor_ids'] for mid in [4,6,7,8,14,27]))
            if finger=='three':
                self.assertEqual(metadata['unlimited_position_motor_ids'],[23,16,2,9,24])
                self.assertEqual(len(metadata['motor_position_limits_deg']),8)
            if finger=='four':
                self.assertEqual(metadata['unlimited_position_motor_ids'],[25,3,15,1,28,23,16,2,9,24])
                self.assertEqual(len(metadata['motor_position_limits_deg']),8)
            if finger in ('ring','both','three','four','five'):
                self.assertEqual(metadata['release_currents_A'],{})
                self.assertEqual(metadata['directional_side_motor_ids'],[22,21])
                self.assertEqual(metadata['side_direction_current_A'],.05)
                self.assertEqual(metadata['side_direction_deadband_deg'],.5)
                self.assertEqual(metadata['side_release_duration_sec'],0)
                self.assertIsNone(metadata['side_release_started_elapsed_sec'])
                for mid in [22,21]:
                    self.assertEqual(metadata['position_kp'][str(mid)],defaults['kp'][mid-1])
                    self.assertEqual(metadata['position_kd'][str(mid)],defaults['kd'][mid-1])
            self.assertEqual(metadata['exit_code'],expected)
            import csv
            records = list(csv.DictReader((root/'output/samples.csv').open()))
            self.assertTrue(records)
            if finger=='both' and mode.startswith('manus'):
                self.assertEqual(metadata['finger_q_slices'],{'ring':[0,4],'little':[4,8]})
                self.assertEqual(set(metadata['model_files']),{'ring','little'})
                self.assertTrue(any(row['target_relative_q4']=='2.0' and row['target_relative_q5']=='8.0' for row in records))
            if finger=='three' and mode.startswith('manus'):
                self.assertEqual(metadata['finger_q_slices'],{'middle':[0,4],'ring':[4,8],'little':[8,12]})
                self.assertEqual(set(metadata['model_files']),{'middle','ring','little'})
                self.assertTrue(any(row['target_relative_q4']=='2.0' and row['target_relative_q8']=='3.0' for row in records))
            if finger=='four' and mode.startswith('manus'):
                self.assertEqual(metadata['finger_q_slices'],{'index':[0,4],'middle':[4,8],'ring':[8,12],'little':[12,16]})
                self.assertEqual(set(metadata['model_files']),{'index','middle','ring','little'})
                self.assertTrue(any(row['target_relative_q4']=='2.0' and row['target_relative_q12']=='-2.0' for row in records))
            if finger=='five' and mode.startswith('manus'):
                self.assertEqual(metadata['finger_q_slices'],{'thumb':[0,4],'index':[4,8],'middle':[8,12],
                                                            'ring':[12,16],'little':[16,20]})
                self.assertEqual(set(metadata['model_files']),{'thumb','index','middle','ring','little'})
                self.assertTrue(any(row['target_relative_q0']=='1.0' and row['target_relative_q16']=='-2.0' for row in records))
            if mode in ('normal','topic') and not interrupt:
                self.assertTrue(any(row['measured_q0'] for row in records))
                for mid in motor_ids:
                    self.assertAlmostEqual(active[-1]['matrix'][mid-1][0],predicted[f'M{mid}'],places=4)
            if finger in ('ring','both','three','four','five'):
                for mid in [22,21]:
                    position_records = [row for row in records if row[f'm{mid}_control_mode']=='position']
                    if mode!='manus_wrong_hand':
                        self.assertTrue(position_records)
                    for row in position_records:
                        self.assertTrue(row[f'm{mid}_target'] and row[f'm{mid}_command'])
                        error=float(row[f'm{mid}_command'])-float(row[f'm{mid}_position'])
                        bias=0 if abs(error)<=.5 else .05 if error>0 else -.05
                        self.assertAlmostEqual(float(row[f'm{mid}_command_current']),bias)
                    self.assertFalse(any(row[f'm{mid}_control_mode']=='release' for row in records))
            if mode.startswith('manus'):
                self.assertFalse(metadata['robot_glove_measurement_enabled'])
                self.assertTrue(all(not row['measured_q0'] and not row['error_q0'] for row in records))
                if mode=='manus_wrong_hand':
                    self.assertIsNone(metadata['manus_zero_q'])
                    self.assertIn('标定超时',metadata['failure'])
                else:
                    self.assertEqual(metadata['manus_zero_q'],operator_zero)
                    self.assertGreater(metadata['manus_target_updates'],1)
                    self.assertTrue(any(row['target_relative_q1']=='10.0' for row in records))
                if finger in ('ring','middle','index','thumb','both','three','four','five'):
                    self.assertEqual(metadata['manus_zero_max_range_deg'],3)
                    if mode!='manus_wrong_hand':
                        self.assertTrue(all(value<=3 for value in metadata['manus_zero_observed_range_deg']))
                if mode=='manus_zero_motion':
                    self.assertGreater(metadata['manus_zero_rejected_windows'],5)
                    self.assertEqual(metadata['manus_zero_observed_range_deg'],[0]*q_count)
                if mode=='manus_outside_ranges':
                    self.assertEqual(metadata['manus_rejected_targets'],0)
                    self.assertTrue(any(float(row['target_relative_q0'])>20 for row in records if row['target_relative_q0']))
                    if finger in ('middle','index'):
                        self.assertEqual(metadata['motor_position_clipped_targets'],0)
                        for row in records:
                            for mid in motor_ids:
                                if row[f'm{mid}_control_mode']=='position':
                                    self.assertEqual(row[f'm{mid}_prediction'],row[f'm{mid}_command'])
                    else:
                        self.assertGreater(metadata['motor_position_clipped_targets'],0)
                        saturated_rows = [row for row in records if row[f'm{motor_ids[0]}_clipped']=='1' and row[f'm{motor_ids[1]}_clipped']=='1']
                        self.assertTrue(saturated_rows)
                        self.assertTrue(all(float(row[f'm{motor_ids[0]}_prediction'])>450 and float(row[f'm{motor_ids[0]}_command'])==450
                                            and float(row[f'm{motor_ids[1]}_prediction'])<-400 and float(row[f'm{motor_ids[1]}_command'])==-400
                                            for row in saturated_rows))
                if mode in ('manus_dropout','manus_invalid','manus_right_only'):
                    self.assertIn('Manus左手数据',metadata['failure'])
            if expected:
                self.assertIn('failure',metadata)

    def test_index_single_point_controls_five_channels(self):
        self.exercise(finger='index')

    def test_thumb_single_point_controls_six_channels(self):
        self.exercise(finger='thumb')

    def test_thumb_manus_uses_mixed_polynomial_model(self):
        self.exercise('manus',finger='thumb')

    def test_thumb_manus_dropout_stops_all_six(self):
        self.exercise('manus_dropout',finger='thumb')

    def test_five_manus_updates_24_channels_with_one_driver(self):
        self.exercise('manus',finger='five')

    def test_mlp_five_manus_updates24_and_logs_model(self):
        self.exercise('manus',finger='five',model='mlp')

    def test_mlp_five_feedback_dropout_stops24(self):
        self.exercise('dropout',finger='five',model='mlp')

    def test_mlp_five_manus_dropout_stops24(self):
        self.exercise('manus_dropout',finger='five',model='mlp')

    def test_five_feedback_dropout_stops_all24_before_driver_exit(self):
        self.exercise('dropout',finger='five')

    def test_five_manus_dropout_stops_all24(self):
        self.exercise('manus_dropout',finger='five')

    def test_five_zero_waits_for_little_finger_to_stop(self):
        self.exercise('manus_zero_motion',finger='five')

    def test_index_manus_relative_four_q_controls_all_five(self):
        self.exercise('manus',finger='index')

    def test_index_manus_dropout_stops_all_five(self):
        self.exercise('manus_dropout',finger='index')

    def test_index_zero_requires_four_stable_angles(self):
        self.exercise('manus_zero_motion',finger='index')

    def test_index_unlimited_joint_and_motor_targets_continue_updating(self):
        self.exercise('manus_outside_ranges',finger='index')

    def test_four_single_point_controls_eighteen_with_one_driver(self):
        self.exercise(finger='four')

    def test_four_manus_uses_four_independent_models(self):
        self.exercise('manus',finger='four')

    def test_four_manus_dropout_stops_all_eighteen(self):
        self.exercise('manus_dropout',finger='four')

    def test_four_zero_waits_for_little_finger_to_stop(self):
        self.exercise('manus_zero_motion',finger='four')

    def test_four_feedback_dropout_stops_all_eighteen(self):
        self.exercise('dropout',finger='four')

    def test_three_single_point_controls_thirteen_motors_with_one_driver(self):
        self.exercise(finger='three')

    def test_three_manus_updates_three_independent_models(self):
        self.exercise('manus',finger='three')

    def test_three_manus_dropout_stops_all_thirteen_before_driver_exit(self):
        self.exercise('manus_dropout',finger='three')

    def test_three_zero_waits_for_little_finger_motion_to_stop(self):
        self.exercise('manus_zero_motion',finger='three')

    def test_middle_single_point_controls_five_motors(self):
        self.exercise(finger='middle')

    def test_middle_manus_uses_relative_four_q_for_five_targets(self):
        self.exercise('manus',finger='middle')

    def test_middle_manus_dropout_stops_all_five(self):
        self.exercise('manus_dropout',finger='middle')

    def test_middle_zero_waits_until_all_four_angles_are_stable(self):
        self.exercise('manus_zero_motion',finger='middle')

    def test_middle_large_angles_continue_without_motor_limits(self):
        self.exercise('manus_outside_ranges',finger='middle')

    def test_both_single_point_controls_eight_motors_with_one_driver(self):
        self.exercise(finger='both')

    def test_both_manus_updates_two_independent_models(self):
        self.exercise('manus',finger='both')

    def test_both_manus_dropout_stops_all_eight_before_driver_exit(self):
        self.exercise('manus_dropout',finger='both')

    def test_both_zero_waits_for_little_finger_motion_to_stop(self):
        self.exercise('manus_zero_motion',finger='both')

    def test_single_point_and_stop(self):
        self.exercise()

    def test_ring_single_point_only_controls_ring(self):
        self.exercise(finger='ring')

    def test_ring_manus_relative_angles_use_ring_glove_columns(self):
        self.exercise('manus',finger='ring')

    def test_ring_manus_motion_cannot_be_latched_as_zero(self):
        self.exercise('manus_zero_motion',finger='ring')

    def test_ring_manus_clips_targets_and_continues_updating(self):
        self.exercise('manus_outside_ranges',finger='ring')

    def test_ring_manus_dropout_stops_ring_before_driver_exit(self):
        self.exercise('manus_dropout',finger='ring')

    def test_ring_missing_left_glove_keeps_four_channels_idle(self):
        self.exercise('manus_wrong_hand',finger='ring')

    def test_ring_side_position_control_continues_past2sec(self):
        self.exercise('manus_long_run',finger='ring')

    def test_target_topic_runs_without_stdin(self):
        self.exercise('topic')

    def test_manus_uses_operator_relative_changes(self):
        self.exercise('manus')

    def test_manus_dropout_stops_with_live_motor_feedback(self):
        self.exercise('manus_dropout')

    def test_manus_invalid_angles_stop(self):
        self.exercise('manus_invalid')

    def test_manus_right_glove_cannot_calibrate_left_control(self):
        self.exercise('manus_wrong_hand')

    def test_manus_right_only_packets_cannot_refresh_left(self):
        self.exercise('manus_right_only')

    def test_manus_unlimited_angles_continue_updating_clipped_motor_targets(self):
        self.exercise('manus_outside_ranges')

    def test_feedback_dropout_stops_without_resuming(self):
        self.exercise('dropout')

    def test_invalid_feedback_stops(self):
        self.exercise('nan')

    def test_motor_error_stops(self):
        self.exercise('error')

    def test_ctrl_c_stops_before_driver_exit(self):
        self.exercise(interrupt=True)

    def test_sigterm_stops_before_driver_exit(self):
        self.exercise(interrupt=True,signum=signal.SIGTERM)

    def test_terminal_hangup_stops_before_driver_exit(self):
        self.exercise(interrupt=True,signum=signal.SIGHUP)


if __name__=='fake_little_sdk':
    fake_sdk()
elif __name__=='__main__':
    unittest.main()
