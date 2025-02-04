import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Определение параметров частицы
# ===============================
particles = {
    'proton': {'charge': 1.602e-19, 'mass': 1.672e-27},
    'deuteron': {'charge': 1.602e-19, 'mass': 3.344e-27},
    'alpha': {'charge': 2 * 1.602e-19, 'mass': 6.644e-27}
}

# Выбор частицы (можете менять)
particle_choice = 'proton'
q = particles[particle_choice]['charge']
m = particles[particle_choice]['mass']

# ===============================
# 2. Параметры циклотрона
# ===============================
B = 1.0  # Магнитное поле (Тесла) внутри дуантов
U = 50e3  # Ускоряющее напряжение (Вольт)
gap_threshold = 0.02  # [м] ширина зазора
E0 = U / gap_threshold  # Напряженность электрического поля в зазоре
max_radius = 0.5  # Максимальный радиус дуанта (м)
dee_spacing = 0.02  # Расстояние между дуантами (м)

# ===============================
# 3. Параметры симуляции
# ===============================
dt = 1e-9  # шаг по времени (сек)
n_steps = 100000  # число шагов симуляции

# ===============================
# 4. Начальные условия
# ===============================
r = np.array([0.0, 0.0])  # стартовая позиция между дуантами (м)
v = np.array([0.0, 1e5])  # стартовая скорость (м/с)

trajectory = []  # для хранения траектории частицы
energy_step = q * U  # увеличение энергии на каждом проходе через зазор
kinetic_energy = 0.5 * m * np.linalg.norm(v)**2  # начальная кинетическая энергия (Дж)
in_gap_previous = True  # Флаг нахождения в зазоре (стартуем в зазоре)
current_gap_sign = 1  # Знак ускоряющего поля в зазоре

# ===============================
# 5. Основной цикл симуляции
# ===============================
for step in range(n_steps):
    # Если радиус превышает максимальный, прекращаем симуляцию
    current_radius = np.linalg.norm(r)
    if current_radius >= max_radius:
        break

    # Проверяем, находится ли частица в зазоре
    in_gap = -dee_spacing / 2 < r[0] < dee_spacing / 2

    if in_gap:
        # Если частица только что вошла в зазор, переключаем знак ускоряющего поля
        if not in_gap_previous:
            current_gap_sign *= -1

        # Применяем ускорение от электрического поля в зазоре
        E_field = np.array([current_gap_sign * E0, 0.0])
        a = (q / m) * E_field
    else:
        # В дуантах применяем ускорение от магнитного поля
        a = (q / m) * np.array([v[1] * B, -v[0] * B])

    # Интеграция уравнений движения (метод Эйлера)
    v += a * dt
    r += v * dt

    # Сохраняем состояние и траекторию
    trajectory.append(r.copy())
    in_gap_previous = in_gap

# ===============================
# 6. Построение графика траектории
# ===============================
trajectory = np.array(trajectory)
plt.figure(figsize=(10, 10))

# Отображение дуантов линиями разных цветов
theta_left = np.linspace(-np.pi / 2, np.pi / 2, 500)
left_dee_x = -(dee_spacing / 2) + max_radius * np.cos(theta_left)
left_dee_y = max_radius * np.sin(theta_left)

theta_right = np.linspace(np.pi / 2, 3 * np.pi / 2, 500)
right_dee_x = (dee_spacing / 2) + max_radius * np.cos(theta_right)
right_dee_y = max_radius * np.sin(theta_right)

plt.plot(left_dee_x, left_dee_y, color='blue', linestyle='-', label='Левый дуант')
plt.plot(right_dee_x, right_dee_y, color='red', linestyle='-', label='Правый дуант')

# Отображение границ дуантов
plt.axvline(-dee_spacing / 2, color='black', linestyle='--', label='Левая граница зазора')
plt.axvline(dee_spacing / 2, color='black', linestyle='--', label='Правая граница зазора')

plt.plot(trajectory[:, 0], trajectory[:, 1], lw=0.5, label='Траектория частицы')
plt.xlabel('x (м)')
plt.ylabel('y (м)')
plt.title(f'Симуляция циклотрона: {particle_choice}, U = {U / 1e3:.1f} kV, B = {B:.1f} T')
plt.grid(True)
plt.axis([-max_radius * 1.1, max_radius * 1.1, -max_radius * 1.1, max_radius * 1.1])
plt.legend()
plt.show()
