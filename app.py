
import streamlit as st
import numpy as np
import random

st.set_page_config(
    page_title="RL Smart Path Finder",
    page_icon="🤖",
    layout="wide"
)

st.title("RL Smart Path Finder 🤖")
st.subheader("Robot Navigation Using Q-Learning")

st.write(
    "This Reinforcement Learning project trains "
    "a robot to reach its destination while "
    "avoiding obstacles."
)

st.sidebar.header("Training Settings")

size = st.sidebar.slider("Grid Size", 4, 8, 5)
episodes = st.sidebar.slider(
    "Training Episodes", 100, 5000, 2000, 100
)
alpha = st.sidebar.slider(
    "Learning Rate", 0.01, 1.0, 0.1
)
gamma = st.sidebar.slider(
    "Discount Factor", 0.1, 0.99, 0.9
)
epsilon = st.sidebar.slider(
    "Exploration Rate", 0.01, 1.0, 0.2
)

start = (0, 0)
goal = (size - 1, size - 1)

obstacles = {
    (1, 1), (1, 3), (2, 3), (3, 1)
}
obstacles = {
    p for p in obstacles
    if p[0] < size and p[1] < size
    and p != start and p != goal
}

actions = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}


def move(state, action):
    dr, dc = actions[action]
    new_state = (
        state[0] + dr,
        state[1] + dc
    )

    r, c = new_state

    if (
        r < 0 or r >= size
        or c < 0 or c >= size
        or new_state in obstacles
    ):
        return state

    return new_state


def train_agent():
    q_table = np.zeros((size, size, 4))

    for episode in range(episodes):
        state = start

        for step in range(size * size * 4):
            if state == goal:
                break

            if random.random() < epsilon:
                action = random.choice(list(actions))
            else:
                action = int(
                    np.argmax(
                        q_table[state[0], state[1]]
                    )
                )

            new_state = move(state, action)

            if new_state == goal:
                reward = 10
            elif new_state == state:
                reward = -3
            else:
                reward = -1

            old_value = q_table[
                state[0], state[1], action
            ]

            next_max = np.max(
                q_table[new_state[0], new_state[1]]
            )

            q_table[
                state[0], state[1], action
            ] = old_value + alpha * (
                reward + gamma * next_max - old_value
            )

            state = new_state

    return q_table


def find_path(q_table):
    state = start
    path = [state]
    visited = {state}

    for _ in range(size * size * 2):
        if state == goal:
            break

        choices = []

        for action in actions:
            nxt = move(state, action)

            if nxt != state and nxt not in visited:
                choices.append(action)

        if not choices:
            break

        action = max(
            choices,
            key=lambda a: q_table[
                state[0], state[1], a
            ]
        )

        state = move(state, action)
        path.append(state)
        visited.add(state)

    return path


if st.button("Train Q-Learning Agent", type="primary"):
    with st.spinner("Training the agent..."):
        q_table = train_agent()
        path = find_path(q_table)

        st.session_state["path"] = path

if "path" in st.session_state:
    path = st.session_state["path"]
    reached = path[-1] == goal

    col1, col2, col3 = st.columns(3)
    col1.metric("Episodes", episodes)
    col2.metric("Path Length", len(path) - 1)
    col3.metric(
        "Goal Reached",
        "Yes" if reached else "Not Yet"
    )

    st.subheader("Robot Navigation Grid")

    grid = []

    for r in range(size):
        row = []

        for c in range(size):
            pos = (r, c)

            if pos == start:
                cell = "🟢 START"
            elif pos == goal:
                cell = "🎯 GOAL"
            elif pos in obstacles:
                cell = "⬛ BLOCK"
            elif pos in path:
                cell = "➡️ PATH"
            else:
                cell = "⬜ EMPTY"

            row.append(cell)

        grid.append(row)

    st.table(grid)

    st.write("### Learned Route")
    st.write(" → ".join(str(p) for p in path))

    if reached:
        st.success("The agent reached the goal!")
    else:
        st.warning(
            "Goal not reached. Increase training episodes "
            "and train again."
        )

    with st.expander("How Q-Learning Works"):
        st.markdown("""
        - **State:** Robot's current position.
        - **Actions:** Up, Down, Left, Right.
        - **Goal reward:** +10.
        - **Normal move reward:** -1.
        - **Wall/obstacle penalty:** -3.
        - **Q-table:** Stores action values for each state.
        - **Alpha:** Learning rate.
        - **Gamma:** Discount factor.
        - **Epsilon:** Controls random exploration.

        Q-Learning formula:

        Q(s,a) = Q(s,a) + alpha * [
        reward + gamma * max Q(next_state) - Q(s,a)
        ]
        """)

else:
    st.info(
        "Click 'Train Q-Learning Agent' to start."
    )

st.divider()

st.caption(
    "Reinforcement Learning Practice Project | "
    "Python, NumPy and Streamlit"
)

st.warning(
    "This is an educational simulation, not a "
    "real-world robot controller."
)
