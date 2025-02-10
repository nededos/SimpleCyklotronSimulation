import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===============================
# 1. Wybor czastki
# ===============================
particles = {
    'proton': {'charge': 1.602e-19, 'mass': 1.672e-27},
    'deuteron': {'charge': 1.602e-19, 'mass': 3.344e-27},
    'alpha': {'charge': 2 * 1.602e-19, 'mass': 6.644e-27}
}

particle_choice = 'proton'
q = particles[particle_choice]['charge']
m = particles[particle_choice]['mass']

# ===============================
# 2. Parametry cyklotronu
# ===============================
B = 1.0  # Pole magnetyczne (T)
U = 50e3  # Napiecie
gap_threshold = 0.02  # szerokosc pomiedzy duantanami (m)
E0 = U / gap_threshold # Pole elektryczne w zazorze (V/m)
max_radius = 0.5  # Maksymalny promien (m)
dee_spacing = 0.02  # Szerokosc zazoru (m)

# ===============================
# 3. Parametry symulacji
# ===============================
dt = 1e-9  # krok czasowy (с)
n_steps = 10000  # liczba krokow czasowych

# ===============================
# 4. Warunki poczatkowe
# ===============================
r = np.array([0.0, 0.0])  # startowy promien (m)
v = np.array([0.0, 1e5])  # startowa predkosc (m/s)

trajectory = [] 
polarities = [] 
energy_step = q * U
kinetic_energy = 0.5 * m * np.linalg.norm(v)**2
in_gap_previous = True 
current_gap_sign = 1  

# ===============================
# 5. Symulacja cyklotronu
# ===============================
for step in range(n_steps):
    current_radius = np.linalg.norm(r)
    if current_radius >= max_radius:
        break

    in_gap = -dee_spacing / 2 < r[0] < dee_spacing / 2

    if in_gap:
        if not in_gap_previous:
            current_gap_sign *= -1

        E_field = np.array([current_gap_sign * E0, 0.0])
        a = (q / m) * E_field
    else:
        a = (q / m) * np.array([v[1] * B, -v[0] * B])

    v += a * dt
    r += v * dt

    trajectory.append(r.copy())
    polarities.append(current_gap_sign)
    in_gap_previous = in_gap

trajectory = np.array(trajectory)

# ===============================
# 6. Animacja cyklotronu
# ===============================
fig, ax = plt.subplots(figsize=(10, 10))

theta_left = np.linspace(-np.pi / 2, np.pi / 2, 500)
left_dee_x = -(dee_spacing / 2) + max_radius * np.cos(theta_left)
left_dee_y = max_radius * np.sin(theta_left)

theta_right = np.linspace(np.pi / 2, 3 * np.pi / 2, 500)
right_dee_x = (dee_spacing / 2) + max_radius * np.cos(theta_right)
right_dee_y = max_radius * np.sin(theta_right)

ax.plot(left_dee_x, left_dee_y, color='blue', linestyle='-', label='Lewy duant')
ax.plot(right_dee_x, right_dee_y, color='red', linestyle='-', label='Prawy duant')

ax.axvline(-dee_spacing / 2, color='black', linestyle='--', label='Lewa granica zazoru')
ax.axvline(dee_spacing / 2, color='black', linestyle='--', label='Prawa granica zazoru')

# Parametry wykresu
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title(f'Symulacja cyklotronu: {particle_choice}, U = {U / 1e3:.1f} kV, B = {B:.1f} T')
ax.grid(True)
ax.axis([-max_radius * 1.1, max_radius * 1.1, -max_radius * 1.1, max_radius * 1.1])
legend = ax.legend(loc='upper right')

trajectory_line, = ax.plot([], [], lw=0.5, label='Trajektoria')
particle_dot, = ax.plot([], [], 'o', color='purple')
polarity_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='green')

def update(frame):
    trajectory_line.set_data(trajectory[:frame, 0], trajectory[:frame, 1])
    particle_dot.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
    polarity_text.set_text(f'Polaryzacja: {"+" if polarities[frame] > 0 else "-"}')
    return trajectory_line, particle_dot, polarity_text

def init():
    trajectory_line.set_data([], [])
    particle_dot.set_data([], [])
    polarity_text.set_text('')
    return trajectory_line, particle_dot, polarity_text

ani = FuncAnimation(fig, update, frames=len(trajectory), init_func=init, interval=5, blit=True, repeat=False)
plt.show()
