class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, min_out: float = 0.0, max_out: float = 100.0):
        """
        Custom PID Controller implementation.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.min_out = min_out
        self.max_out = max_out
        
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint: float, current_value: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
            
        error = setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with basic anti-windup clamping)
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # Calculate total output
        output = p_term + i_term + d_term
        
        # Clamp output
        clamped_output = max(self.min_out, min(self.max_out, output))
        
        # Simple anti-windup: Revert integral if clamped
        if clamped_output != output:
            self.integral -= error * dt
            
        self.prev_error = error
        
        return clamped_output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
