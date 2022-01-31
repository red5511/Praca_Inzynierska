from ball import Ball
from palette import Palette
import numpy as np

class Air_hockey_env():
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y
		self.palette1 = Palette(28, Y-2, 5) 
		self.palette2 = Palette(28, 2, 5)
		self.ball = Ball(X, Y, self.palette1, self.palette2)
		self.points1 = 0
		self.points2 = 0

		self.actions = ["LEFT", "RIGHT", "SHOOT", "NONE"]
		self.render_history_palette1 = []
		self.render_history_palette2 = []

		self.action_n = len(self.actions)


	def sample(self):
		return np.random.choice(self.actions, 1)[0]

	def reset(self):
		self.palette1 = Palette(28, self.Y-2, 5) 
		self.palette2 = Palette(28, 2, 5)
		self.ball = Ball(self.X, self.Y, self.palette1, self.palette2)
		self.points1 = 0
		self.points2 = 0
		self.render_history_palette1 = []
		self.render_history_palette2 = []
		state1 = np.array([self.palette1.x, self.palette1.y, self.ball.x, self.ball.y, self.ball.dx, self.ball.dy])
		state2 = np.array([self.palette2.x, self.palette2.y, self.ball.x, self.ball.y, self.ball.dx, self.ball.dy])

		return state1, state2

	def step(self, action1, action2):

		if self.palette1.shoot:
				if self.palette1.shoot == 2:
					self.palette1.y += 2
					self.palette1.shoot = 0
				else:
					self.palette1.y -= 1
					self.palette1.shoot += 1
		elif action1 == "LEFT":
			if self.palette1.x > 1:
				self.palette1.x -= 1
		elif action1 == "RIGHT":
			if self.palette1.x < self.X-1 - self.palette1.len:
				self.palette1.x += 1
		elif action1 == "SHOOT":
			self.palette1.shoot = 1
			self.palette1.y -= 1


		if self.palette2.shoot:
			if self.palette2.shoot == 2:
				self.palette2.y -= 2
				self.palette2.shoot = 0
			else:
				self.palette2.y += 1
				self.palette2.shoot += 1
		elif action2 == "LEFT":
			if self.palette2.x > 1:
				self.palette2.x -= 1
		elif action2 == "RIGHT":
			if self.palette2.x < self.X-1 - self.palette2.len:
				self.palette2.x += 1
		elif action2 == "SHOOT":
			self.palette2.shoot = 1
			self.palette2.y += 1

		self.ball.next_move()
		self.ball_clear()

		reward1 = 0
		reward2 = 0

		if self.ball.y == 1:
			reward1 += 1
			reward2 -= 10
			self.points1 += 1

		elif self.ball.y == self.Y-2:
			reward1 -= 10
			reward2 += 1

			self.points2 += 1

		done = False
		if self.points1 == 5 or self.points2 == 5:
			done = True


		self.render_history_palette1.append(action1)
		self.render_history_palette2.append(action2)

		state1 = np.array([self.palette1.x, self.palette1.y, self.ball.x, self.ball.y, self.ball.dx, self.ball.dy])
		state2 = np.array([self.palette2.x, self.palette2.y, self.ball.x, self.ball.y, self.ball.dx, self.ball.dy])


		return state1, state2, reward1, reward2, done

	def render(self):
		return self.render_history_palette1, self.render_history_palette2

	def ball_clear(self):

		if self.ball.reset:
			self.ball.reset = False
			self.ball.y = 10
			self.ball.dy *= -1
			self.ball.dx = int(self.ball.dx / abs(self.ball.dx))

		if self.ball.y == 1 or self.ball.y == self.ball.dimY-1:
			self.ball.reset = True

