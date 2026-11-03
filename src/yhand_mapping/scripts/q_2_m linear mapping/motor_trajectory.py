"""Per-motor position-command velocity/acceleration limits, without ROS.

Limits apply to commanded positions, not a guarantee of measured motor speed.
Initialize from fresh feedback; inactive position rows reset immediately.
Startup-only profiles wait for meaningful motion, limit the initial movement,
then smoothly hand over to the live target. Continuous profiles are unchanged.
"""
import math


def mapper_motion_limits(mapper):
    result = {}
    for child in getattr(mapper, 'mappers', (mapper,)):
        limits = child.config.get('teleop_motion_limits', {})
        if not isinstance(limits, dict):
            raise ValueError('teleop_motion_limits必须为字典')
        for key, value in limits.items():
            try:
                mid = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError('轨迹约束电机编号无效') from exc
            if str(mid) != str(key) or mid not in child.motor_ids or mid in result:
                raise ValueError('轨迹约束电机不属于当前模型或重复')
            result[mid] = value
    return result


class MotorTrajectoryLimiter:
    def __init__(self, limits):
        self.limits = {}
        self.startup_profiles = {}
        for mid, config in limits.items():
            try:
                values = [config[key] for key in ('max_velocity_deg_sec', 'max_acceleration_deg_sec2')]
                speed, acceleration = map(float, values)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('电机轨迹速度/加速度配置不完整') from exc
            if any(isinstance(v, bool) for v in values) or not all(
                    math.isfinite(v) and v > 0 for v in (speed, acceleration)):
                raise ValueError('电机轨迹速度/加速度必须为正的有限数值')
            self.limits[mid] = (speed, acceleration)
            mode = config.get('mode', 'continuous')
            if mode not in ('continuous', 'startup_only'):
                raise ValueError('轨迹模式必须为continuous或startup_only')
            if mode == 'startup_only':
                try:
                    raw = [config[key] for key in ('startup_duration_sec',
                           'handoff_duration_sec', 'activation_threshold_deg')]
                    startup, handoff, threshold = map(float, raw)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError('启动轨迹配置不完整') from exc
                if any(isinstance(v, bool) for v in raw) or not all(
                        math.isfinite(v) and v > 0 for v in (startup, handoff, threshold)):
                    raise ValueError('启动时长、交接时长和触发量必须为正的有限数值')
                self.startup_profiles[mid] = (startup, handoff, threshold)
        self.states = {}
        self.startup_states = {}
        self.last = {}
        self.last_time = None

    def reset(self):
        self.states.clear()
        self.startup_states.clear()
        self.last.clear()
        self.last_time = None

    @staticmethod
    def limited_step(position, velocity, target, dt, speed, acceleration):
        if dt > 0:
            error = target-position
            goal = math.copysign(min(speed, abs(error)/dt,
                                     math.sqrt(2.*acceleration*abs(error))), error)
            velocity = max(velocity-acceleration*dt, min(velocity+acceleration*dt, goal))
            position += velocity*dt
        return position, velocity

    def startup_step(self, mid, target, dt, real_dt, speed, acceleration):
        state = self.startup_states[mid]
        startup, handoff, threshold = self.startup_profiles[mid]
        elapsed = state['elapsed']
        if elapsed is None:
            if abs(target-state['initial_position']) >= threshold:
                elapsed = 0.
        else:
            elapsed += dt  # Do not skip the ramp after a delayed control tick.
        state['elapsed'] = elapsed
        if elapsed is not None and elapsed >= startup+handoff:
            position, phase, weight = target, 'follow', 1.
        else:
            limited = self.limited_step(*state['limited'], target, dt, speed, acceleration)
            state['limited'] = limited
            position, phase, weight = limited[0], 'waiting', 0.
            if elapsed is not None:
                phase = 'startup'
                if elapsed >= startup:
                    phase = 'handoff'
                    u = min(1., (elapsed-startup)/handoff)
                    # Quintic endpoints have zero first/second derivative:
                    # no hard release of the accumulated position error.
                    weight = u*u*u*(10.+u*(-15.+6.*u))
                    position += weight*(target-position)
        previous = self.states[mid][0]
        velocity = (position-previous)/real_dt if real_dt > 0 else 0.
        self.last[mid] = dict(phase=phase, blend_weight=weight,
                              startup_elapsed_sec=elapsed)
        return position, velocity

    def apply(self, matrix, feedback, now):
        if not math.isfinite(now) or self.last_time is not None and now < self.last_time:
            raise ValueError('轨迹时间必须有限且不能倒退')
        real_dt = 0. if self.last_time is None else now-self.last_time
        dt = min(real_dt, .05)
        self.last_time = now
        result = [list(row) for row in matrix]
        for mid, (speed, acceleration) in self.limits.items():
            row = result[mid-1]
            if not row[3] and not row[4]:
                self.states.pop(mid, None)
                self.startup_states.pop(mid, None)
                self.last.pop(mid, None)
                continue
            measured = feedback[mid-1]
            if (not all(math.isfinite(v) for v in measured[:4]) or measured[3] != 0
                    or not math.isfinite(row[0])):
                raise ValueError('轨迹约束需要有效的最新电机反馈与目标')
            if mid not in self.states:
                self.states[mid] = (measured[0], 0.)
                if mid in self.startup_profiles:
                    threshold = self.startup_profiles[mid][2]
                    elapsed = 0. if abs(row[0]-measured[0]) >= threshold else None
                    self.startup_states[mid] = dict(initial_position=measured[0],
                        limited=(measured[0], 0.), elapsed=elapsed)
                    self.last[mid] = dict(phase='startup' if elapsed is not None else 'waiting',
                                          blend_weight=0., startup_elapsed_sec=elapsed)
                else:
                    self.last[mid] = dict(phase='continuous', blend_weight=0.,
                                          startup_elapsed_sec=None)
                row[0] = measured[0]
                continue
            if mid in self.startup_profiles:
                position, velocity = self.startup_step(mid, row[0], dt, real_dt, speed, acceleration)
            else:
                position, velocity = self.limited_step(*self.states[mid], row[0], dt, speed, acceleration)
            row[0] = position
            self.states[mid] = (position, velocity)
        return result
