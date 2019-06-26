import numpy as np
from reverse_env import ReverseEnv
import sys
import math

import tkinter as tk
import tkinter.font as font


def play_game(str_len, reverse_len, offset, num_obscured):
    print(f'Game with string length = {str_len}, reverse size = {reverse_len}, offset = {offset}.')
    steps_taken = []
    n_steps = 0

    env = ReverseEnv(str_len, reverse_len, offset, num_obscured)
    ep = env.start_ep()
    n_actions = len(env.actions_list)

    def is_rotatable(i):
        lm = leftmost(i)
        return lm >= 0 and lm % offset == 0 and lm / offset < n_actions

    def press(i):
        nonlocal n_steps
        n_steps += 1
        lm = leftmost(i)
        action = int(lm / offset)
        nonlocal ep
        obs, _, reward, done = ep.make_action(action) # pivot is one left
        if not done:
            for i, b in enumerate(current_buttons):
                b.config(text=str(obs[i]))
        else:
            steps_taken.append(n_steps)
            n_steps = 0
            ep = env.start_ep()
            obs = ep.get_obs()[0]

            current = obs[:str_len]
            goal = obs[str_len:]
            for i, (g, b) in enumerate(zip(goal_buttons, current_buttons)):
                g.config(text=str(goal[i]))
                b.config(text=str(current[i]))

    def leftmost(i):
        return i + 1 - math.ceil(reverse_len / 2)
    
    def rightmost(i):
        return i - 1 + math.ceil(reverse_len / 2) + (reverse_len % 2 == 0)

    def underline(i):
        for b in current_buttons[leftmost(i):rightmost(i)+1]:
            b.config(font=tahoma_underline)

    def ununderline(i):
        for b in current_buttons[leftmost(i):rightmost(i)+1]:
            b.config(font=tahoma_small)



    root = tk.Tk()
    frame = tk.Frame(root)
    frame.winfo_toplevel().title("Reversal Game")
    frame.pack()
    
    goal_buttons = []
    current_buttons = []

    tahoma = font.Font(family='Tahoma', size=26)
    tahoma_small = font.Font(family='Tahoma', size=26)
    tahoma_small2 = font.Font(family='Tahoma', size=16)
    tahoma_bold = font.Font(family='Tahoma bold', size=26)
    tahoma_underline = font.Font(family='Tahoma', size=26, underline=True)

    obs = ep.get_obs()[0]
    current = obs[:str_len]
    goal = obs[str_len:]

    label1 = tk.Label(frame, text='target:', height=1, width=7, fg='light green')
    label1.grid(row=0, column = 0)
    label1['font'] = tahoma_small
    label2 = tk.Label(frame, text='current:', height=1, width=7, fg='teal')
    label2.grid(row=1, column = 0)
    label2['font'] = tahoma_small
    label3 = tk.Label(frame, 
            text=f'Reverse size: {reverse_len}  Offset: {offset}. Click an arrow to reverse around it. Try to obtain target string.', height=1,
            width=80, fg='teal')
    label3['font'] = tahoma_small2
    label3.grid(row=3, column=0, columnspan=1 + 2*str_len)

    for i in range(str_len):
        g = tk.Label(frame, text=goal[i], font = tahoma, height=1, width=1,
                fg='light green')
        g.grid(row=0, column = 2*i+1)
        gap1 = tk.Label(frame, text=' ',height=1, width=1)
        gap1.grid(row=0, column = 2*i + 2)
        goal_buttons.append(g)
                
        b = tk.Label(frame, text=current[i], height=1, width=1, fg='teal', 
                font = tahoma)
        b.grid(row=1, column = 2*i+1)

        current_buttons.append(b)

        gap2 = tk.Label(frame, text=' ',height=1, width=1)
        gap2.grid(row=1, column = 2*i + 2)

        a = tk.Label(frame, text='^', height=1, width=1, font=tahoma_bold,
                fg='coral' if is_rotatable(i) else 'white',
                cursor='exchange' if is_rotatable(i) else None)
        if is_rotatable(i):
            a.bind('<Button-1>', lambda event, i=i: press(i))
            a.bind('<Enter>', lambda event, i=i: underline(i))
            a.bind('<Leave>', lambda event, i=i: ununderline(i))

        a.grid(row=2, column =  2*i + 1 + (reverse_len % 2 == 0))

        gap3 = tk.Label(frame, text=' ', height=1, width=1)
        gap3.grid(row=2, column=2*i + 2 - (reverse_len % 2 == 0))

    root.mainloop()

    print('Steps taken each game: ' + str(steps_taken))


if __name__ == '__main__':
    str_len, reverse_len, offset = map(int, sys.argv[1:])
    num_obscured = 0
    play_game(str_len, reverse_len, offset, num_obscured)
