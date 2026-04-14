import matplotlib.pyplot as plt
from env.render import render_env

def maybe_render(env, render_flag):
    """Условный рендеринг среды: если render_flag=True, вызывает render_env."""
    if render_flag:
        render_env(env)
        plt.pause(0.01)  # небольшая пауза для обновления