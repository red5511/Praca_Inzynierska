from air_hockey_env import Air_hockey_env


class Random_game():
	def __init__(self):
		self.env = Air_hockey_env(60, 30)
		self.env.reset()
		done = False

		while not done:
			action = self.env.sample()
			action2 = self.env.sample()
			state1, state2, reward1, reward2, done = self.env.step(action, action2)

	def render(self):
		rend1, rend2 = self.env.render()
		return rend1, rend2
