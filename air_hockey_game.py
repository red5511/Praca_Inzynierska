import curses
from ball import Ball
from palette import Palette
import time
from tensorflow.keras import models
import numpy as np

class Air_hockey():
	def __init__(self, win, X, Y, AI_PLAY = False, render1 = None, render2 = None):
		self.win = win
		self.X = X
		self.Y = Y
		self.palette1 = Palette(28, Y-2, 5) 
		self.palette2 = Palette(28, 2, 5)
		self.ball = Ball(X, Y, self.palette1, self.palette2)
		self.render1 = render1
		self.render2 = render2
		self.render_dict1 = {"LEFT" : curses.KEY_LEFT, "RIGHT" : curses.KEY_RIGHT, "SHOOT" : ord(' '), "NONE" : None}
		self.render_dict2 = {"LEFT" : ord('a'), "RIGHT" : ord('d'), "SHOOT" : ord('q'), "NONE" : None}
		
		self.model = None
		self.AI_PLAY = AI_PLAY
		self.ACTIONS = ["LEFT", "RIGHT", "SHOOT", "NONE"]

		
		if AI_PLAY:
			self.model = models.load_model('AI/saved_models/model{}.h5'.format("1_7100"))



		win.clear()
		win.border(0)

		for i in range(1, X-1):
			win.addch(Y-1, i, " ")
			win.addch(0, i, " ")

		self.run()


	def ball_clear(self):

		if self.ball.reset:
			self.ball.reset = False
			self.ball.y = 10
			self.ball.dy *= -1
			self.ball.dx = int(self.ball.dx / abs(self.ball.dx))
			for i in range(1, self.X-1):
				self.win.addch(self.Y-1, i, " ")
				self.win.addch(1, i, " ")


		if self.ball.bounced == 1:
			if not self.ball.exception2: #odbicie od lewej badz prawej sciany i dodatkowo paletka wbija
				x_draw = self.ball.x - self.ball.old_dx
				self.win.addch(self.ball.y + self.ball.dy, x_draw, " ")
			else:
				x_draw = self.ball.x + self.ball.old_dx
				self.win.addch(self.ball.y + self.ball.dy, x_draw, " ")


		elif self.ball.x != self.ball.dimX-2 and self.ball.x != 1:
			x_draw =  self.ball.x - self.ball.dx
			self.win.addch(self.ball.y - self.ball.dy, x_draw, " ")
		else:
			x_draw = self.ball.x + self.ball.dx
			self.win.addch(self.ball.y - self.ball.dy, x_draw, " ")

		if self.ball.y == 1 or self.ball.y == self.ball.dimY-1:
			self.ball.reset = True

	def run(self):

		ESC = 27
		key = None

		points1 = 0
		points2 = 0

		curses.use_default_colors()
		curses.init_pair(1, curses.COLOR_RED, -1)

		finished = False
		render_index = 0

		while key != ESC:
			if points1 == 5:
				self.win.clear()
				string = "Player1 wins!!!"
				self.win.addstr(self.Y//2, int(self.X/2) - len(string)//2, string)
				finished = True
			elif points2 == 5:
				self.win.clear()
				string = "Player2 wins!!!"
				self.win.addstr(self.Y//2, int(self.X/2) - len(string)//2, string)
				finished = True

			self.win.timeout(555)
			self.win.addstr(0, 3, " Punkty #1 = " + str(points1) + " ")
			self.win.addstr(0, 40, " Punkty #2 = " + str(points2) + " ")
			if (self.ball.dx != 1 and self.ball.dx != -1) or self.ball.exception:
				self.win.attron(curses.color_pair(1))
				self.win.addch(self.ball.y, self.ball.x, "+")
				self.win.attroff(curses.color_pair(1))
			else:
				self.win.addch(self.ball.y, self.ball.x, "+")

 
			event = self.win.getch()
			key = event
			key2 = key
			if self.render1 != None and not finished and key != 27:				 
				key = self.render_dict1[self.render1[render_index]]
				key2 = self.render_dict2[self.render2[render_index]]
				render_index += 1
			elif self.AI_PLAY and not finished:
				state = np.array([self.palette1.x, self.palette1.y, self.ball.x, self.ball.y, self.ball.dx, self.ball.dy])
				state = state.reshape((1, 6))
				action = self.model.predict(state)[0]
				key = self.render_dict1[self.ACTIONS[np.argmax(action)]]
			if finished:
				if key != ESC:
					key = ord("t")

			if self.palette1.shoot:
				if self.palette1.shoot == 2:
					for i in range(self.palette1.len):
						self.win.addch(self.palette1.y, self.palette1.x + i, " ")
					self.palette1.y += 2
					self.palette1.shoot = 0
				else:
					for i in range(self.palette1.len):
						self.win.addch(self.palette1.y, self.palette1.x + i, " ")

					self.palette1.y -= 1
					self.palette1.shoot += 1
			elif key == curses.KEY_LEFT:
				if self.palette1.x > 1:
					for i in range(self.palette1.len):
						self.win.addch(self.palette1.y, self.palette1.x + i, " ")
					self.palette1.x -= 1

			elif key == curses.KEY_RIGHT:
				if self.palette1.x < self.X-1 - self.palette1.len:
					for i in range(self.palette1.len):
						self.win.addch(self.palette1.y, self.palette1.x + i, " ")
					self.palette1.x += 1
			elif key == ord(" "):
				self.palette1.shoot = 1
				for i in range(self.palette1.len):
					self.win.addch(self.palette1.y, self.palette1.x + i, " ")
				self.palette1.y -= 1

			if self.palette2.shoot:
				if self.palette2.shoot == 2:
					for i in range(self.palette2.len):
						self.win.addch(self.palette2.y, self.palette2.x + i, " ")
					self.palette2.y -= 2
					self.palette2.shoot = 0
				else:
					for i in range(self.palette2.len):
						self.win.addch(self.palette2.y, self.palette2.x + i, " ")
					self.palette2.y += 1
					self.palette2.shoot += 1
			elif key2 == ord('a'):
				if self.palette2.x > 1:
					for i in range(self.palette2.len):
						self.win.addch(self.palette2.y, self.palette2.x + i, " ")
					self.palette2.x -= 1
			elif key2 == ord('d'):
				if self.palette2.x < self.X-1 - self.palette2.len:
					for i in range(self.palette2.len):
						self.win.addch(self.palette2.y, self.palette2.x + i, " ")
					self.palette2.x += 1
			elif key2 == ord("q"):
				self.palette2.shoot = 1
				for i in range(self.palette2.len):
					self.win.addch(self.palette2.y, self.palette2.x + i, " ")
				self.palette2.y += 1

			if not finished:
				self.ball.next_move()

				self.ball_clear()

				for i in range(self.palette1.len):
					self.win.addch(self.palette1.y, self.palette1.x + i, "#")
					self.win.addch(self.palette2.y, self.palette2.x + i, "#")

				if self.ball.y == 1:
					points1 += 1
				elif self.ball.y == self.Y-2:
					points2 += 1


