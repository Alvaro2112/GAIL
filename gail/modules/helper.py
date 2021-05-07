def evaluate(n_eval_episodes, env, max_timesteps, policy):
    total_reward = 0
    i = 0

    while i < n_eval_episodes:

        state = env.reset()

        for t in range(max_timesteps):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:  # TODO is this necessary????
                i += 1
                break

    return int(total_reward / n_eval_episodes)
