
class Ball():
	def __init__(self, dimX, dimY, p1, p2):
		self.x = 24
		self.y = 10
		self.dx = 1
		self.dy = 1

		self.dimX = dimX
		self.dimY = dimY

		self.palette1 = p1
		self.palette2 = p2
		self.buf1 = False
		self.buf2 = False
		self.buf_xy1 = None
		self.buf_xy2 = None
		
		self.bounced = 0
		self.reset = False

		self.old_dx = None
		self.exception = False
		self.exception2 = False


	def next_move(self):
		self.exception2 = False
		if self.exception:
			self.exception = False
			self.dx = 2 * int(self.dx / abs(self.dx))

		if (self.x == 2 or self.x == self.dimX - 3) and (self.dx != 1 and self.dx != -1): #if pomagajacy przy dx = 2 (kiedy byl shoot)
			self.dx = int(self.dx / abs(self.dx))
			self.x += self.dx
			self.exception = True
		else:
			self.x += self.dx

		self.y += self.dy

		if self.x == self.dimX - 2 or self.x == 1:
			self.exception2 = True
			self.dx *= -1

		if self.bounced:
			self.bounced = (self.bounced + 1) % 5
		elif (self.y == self.palette1.y - 1 or self.y == self.palette1.y) and self.x >= self.palette1.x - 1 and self.x <= self.palette1.x + self.palette1.len:
			self.old_dx = self.dx

			if self.palette1.y < 28:
				#print('trueeeee', self.palette1.y)
				self.dx = 2 * int(self.dx / abs(self.dx))
			else:
				self.dx = int(self.dx / abs(self.dx))

			self.dy *= -1
			self.buf_xy1 = [self.palette1.x, self.palette1.y]
			self.buf1 = True
			self.bounced += 1

		elif (self.y == self.palette2.y + 1 or self.y == self.palette2.y) and self.x >= self.palette2.x - 1 and self.x <= self.palette2.x + self.palette2.len:
			self.old_dx = self.dx

			if self.palette2.y > 2:
				#print('trueeeee0', self.palette2.y)
				self.dx = 2 * int(self.dx / abs(self.dx))
			else:
				self.dx = int(self.dx / abs(self.dx))

			self.dy *= -1
			self.buf_xy2 = [self.palette2.x, self.palette2.y]
			self.buf2 = True
			self.bounced += 1



