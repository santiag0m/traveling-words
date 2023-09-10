import numpy as np
import matplotlib.pyplot as plt

radius = 3
theta = np.linspace(0, 2 * np.pi, 100)
x1 = radius * np.cos(theta)
y1 = radius * np.sin(theta)

scale_x, scale_y = 4, 2
x2 = scale_x * np.cos(theta)
y2 = scale_y * np.sin(theta)


translate_x, translate_y = 3, 2
x3 = x2 + translate_x
y3 = y2 + translate_y


transformation_matrix = np.array([[0.8, -0.5], [0.3, 0.9]])
transformed_coords = np.dot(transformation_matrix, np.array([x3, y3]))
x4, y4 = transformed_coords[0], transformed_coords[1]


# Initialize subplots in a 2x2 grid

fig, ax = plt.subplots(2, 2, figsize=(10, 10))


# Flatten the axis array for easier indexing

ax = ax.flatten()


# Common settings

for a in ax:
    a.set_xlim(-10, 10)

    a.set_ylim(-10, 10)

    a.axis("off")

    a.axhline(0, color="grey", linewidth=0.5)

    a.axvline(0, color="grey", linewidth=0.5)


# Generate random points along the edge of the original circle

theta_rand = np.random.choice(theta, 20, replace=False)
rand_x1 = radius * np.cos(theta_rand)
rand_y1 = radius * np.sin(theta_rand)


# Slide A: Draw a bigger circle with random points

ax[0].plot(x1, y1, color="black")
ax[0].scatter(rand_x1, rand_y1, marker="x", color="blue")


# Slide B: Transform circle to ellipse with scaling (more scaling in x-axis than y-axis)

# Transform the random points from Slide A
rand_x2 = scale_x * np.cos(theta_rand)
rand_y2 = scale_y * np.sin(theta_rand)

ax[1].plot(x2, y2, color="black")
ax[1].scatter(rand_x2, rand_y2, marker="x", color="blue")
ax[1].arrow(0, 0, scale_x, 0, head_width=0.2, head_length=0.2, fc="blue", ec="blue")
ax[1].arrow(0, 0, 0, scale_y, head_width=0.2, head_length=0.2, fc="blue", ec="blue")


# Slide C: Translate ellipse

# Translate the random points from Slide B
rand_x3 = rand_x2 + translate_x
rand_y3 = rand_y2 + translate_y

ax[2].plot(x3, y3, color="black")
ax[2].scatter(rand_x3, rand_y3, marker="x", color="blue")
ax[2].arrow(
    0,
    0,
    translate_x,
    translate_y,
    head_width=0.2,
    head_length=0.2,
    fc="green",
    ec="green",
)


# Slide D: Apply random matrix transformation

# Transform the translated random points from Slide C

rand_transformed_coords = np.dot(transformation_matrix, np.array([rand_x3, rand_y3]))
rand_x4, rand_y4 = rand_transformed_coords[0], rand_transformed_coords[1]

ax[3].plot(x3, y3, color="black")
ax[3].scatter(rand_x3, rand_y3, marker="x", color="blue")
ax[3].plot(x4, y4, color="red")
ax[3].scatter(rand_x4, rand_y4, marker="x", color="red")


ax[0].set_title("1. Layer Normalization", fontsize="large")
ax[1].set_title(r"2. Scaling parameter $\gamma$", fontsize="large")
ax[2].set_title(r"3. Bias parameter $\beta$", fontsize="large")
ax[3].set_title(r"4. Transformation matrix $W_{QK}$", fontsize="large")


# Show the plot

plt.tight_layout()

plt.show()
