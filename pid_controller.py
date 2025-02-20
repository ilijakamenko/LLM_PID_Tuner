import control as ctrl

class PIDController:
    def __init__(self, Kp, Ti, Td):
        self.Kp, self.Ti, self.Td = Kp, Ti, Td
    
    def update(self, Kp, Ti, Td):
        """Update PID parameters without creating a new instance."""
        self.Kp, self.Ti, self.Td = Kp, Ti, Td
    
    def tf(self):
        return ctrl.TransferFunction([self.Kp * self.Td, self.Kp, self.Kp / self.Ti], [1, 0])
    
    def copy(self):
        """Returns a new copy of the PIDController."""
        return PIDController(self.Kp, self.Ti, self.Td)

