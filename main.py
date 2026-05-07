import taichi as ti
import numpy as np

@ti.data_oriented
class Screen:
    def __init__(self):
        self.window = ti.ui.Window("GNC Lander - Quaternion Mode", (720, 480), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.gui_ = self.window.get_gui()
        self.camera = ti.ui.Camera()
        self.camera.position(0.0, 10.0, 30.0)
        
        self.lander = Lander()
        
        # Index data for a cube
        self.indices = ti.field(ti.i32, shape=36)
        self.indices.from_numpy(np.array([0,1,2, 0,2,3, 4,5,6, 4,6,7, 0,1,5, 0,5,4, 2,3,7, 2,7,6, 0,3,7, 0,7,4, 1,2,6, 1,6,5], dtype=np.int32))

    def get_delta_q(self):
        """Creates a tiny rotation quaternion based on keys."""
        # Angle of rotation per frame (roughly 1 degree)
        angle = 0.02 
        s = np.sin(angle / 2)
        c = np.cos(angle / 2)
        
        dq = np.array([1.0, 0.0, 0.0, 0.0]) # Identity
        
        if self.window.is_pressed('i'): dq = np.array([c, s, 0, 0])   # Pitch +X
        if self.window.is_pressed('k'): dq = np.array([c, -s, 0, 0])  # Pitch -X
        if self.window.is_pressed('j'): dq = np.array([c, 0, s, 0])   # Yaw +Y
        if self.window.is_pressed('l'): dq = np.array([c, 0, -s, 0])  # Yaw -Y
        if self.window.is_pressed('u'): dq = np.array([c, 0, 0, s])   # Roll +Z
        if self.window.is_pressed('o'): dq = np.array([c, 0, 0, -s])  # Roll -Z
        
        return ti.Vector(dq)

    def gui(self):
        with self.gui_.sub_window("Speed Shower", x=0.05, y=0.05, width=0.3, height=0.2):
            self.gui_.text(f"Speed: {self.lander.vel}")

    def run(self):
        while self.window.running:
            throttle = 1.0 if self.window.is_pressed(ti.ui.SPACE) else 0.0
            dq = self.get_delta_q()
            
            self.lander.update(1/60, throttle, dq)
            
            self.camera.track_user_inputs(self.window, movement_speed=0.5, hold_key=ti.ui.RMB)
            self.scene.set_camera(self.camera)
            self.scene.ambient_light((0.5, 0.5, 0.5))
            self.scene.point_light(pos=(0, 20, 20), color=(1, 1, 1))
            
            self.scene.mesh(self.lander.vertices, indices=self.indices, color=(0.2, 0.6, 1.0))
            self.gui()
            self.canvas.scene(self.scene)
            self.window.show()

@ti.data_oriented
class Lander:
    def __init__(self, pos=None, m=10.0, g=-9.8):
        self.g = g 
        self.mass = m 
        self.max_thrust = 1000.0 

        # State Fields
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=())
        
        # Quaternion [w, x, y, z]. Initialized to Identity (no rotation)
        self.q = ti.Vector.field(4, dtype=ti.f32, shape=())
        
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)

        # Initialization
        if pos is None: self.pos[None] = ti.Vector([0.0, 50.0, 0.0])
        else: self.pos[None] = ti.Vector(pos)
        
        self.q[None] = ti.Vector([1.0, 0.0, 0.0, 0.0]) # W=1 means "Pointing straight"
        self.vel[None] = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def quat_mul(self, q1, q2):
        """Hamilton Product: Smushes two rotations together."""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        return ti.Vector([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @ti.func
    def rotate_vector(self, v, q):
        """The 'Sandwich Product' optimized for GPUs."""
        qw = q[0]
        qv = ti.Vector([q[1], q[2], q[3]])
        return v + 2.0 * qv.cross(qv.cross(v) + qw * v)

    @ti.func
    def apply_phy(self, dt, throttle, q_delta):
        # 1. Update Orientation (Before Data * New Change)
        self.q[None] = self.quat_mul(self.q[None], q_delta).normalized()

        # 2. Derive World Thrust Direction from Local Up [0, 1, 0]
        thrust_dir = self.rotate_vector(ti.Vector([0.0, 1.0, 0.0]), self.q[None])

        # 3. Standard Physics (Euler Method)
        gravity_force = ti.Vector([0.0, self.g * self.mass, 0.0])
        thrust_force = thrust_dir * (throttle * self.max_thrust)
        
        acc = (gravity_force + thrust_force) / self.mass
        self.vel[None] += acc * dt
        self.pos[None] += self.vel[None] * dt

        # Ground Collision
        if self.pos[None].y < 0:
            self.pos[None].y = 0
            self.vel[None] = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def generate_geometry(self, w, h, d):
        hw, hh, hd = w/2.0, h/2.0, d/2.0
        offsets = [
            ti.Vector([-hw,  hh,  hd]), ti.Vector([ hw,  hh,  hd]),
            ti.Vector([ hw,  hh, -hd]), ti.Vector([-hw,  hh, -hd]),
            ti.Vector([-hw, -hh,  hd]), ti.Vector([ hw, -hh,  hd]),
            ti.Vector([ hw, -hh, -hd]), ti.Vector([-hw, -hh, -hd])
        ]
        for i in ti.static(range(8)):
            # Rotate local corners then slide to world position
            self.vertices[i] = self.pos[None] + self.rotate_vector(offsets[i], self.q[None])

    @ti.kernel
    def update(self, dt: ti.f32, throttle: ti.f32, q_delta: ti.types.vector(4, ti.f32)):
        self.apply_phy(dt, throttle, q_delta)
        self.generate_geometry(4, 2, 4)


if __name__ == "__main__":
    ti.init(arch=ti.vulkan)
    s = Screen()
    s.run()