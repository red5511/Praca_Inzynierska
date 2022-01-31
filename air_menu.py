import curses
from air_hockey_game import Air_hockey
from random_game import Random_game
import json


X, Y = 60, 30



curses.initscr()
curses.start_color()
win = curses.newwin(Y, X, 0, 0) #y, x
win.keypad(1)
curses.noecho()
curses.curs_set(0)
win.border(0)
win.nodelay(1)

curses.init_pair(1, curses.COLOR_RED, curses.COLOR_YELLOW)


ESC = 27
key = None

menu = ["1 vs 1", "AI vs AI - render", "Random vs Random", "1 vs AI", "Exit"]

def draw_menu(win, idx):
	curses.init_pair(1, curses.COLOR_RED, curses.COLOR_YELLOW)

	for i, row in enumerate(menu):
		if i == idx:
			win.attron(curses.color_pair(1))
			win.addstr(Y//2 + i - len(menu)//2, int(X/2) - len(row)//2, row)
			win.attroff(curses.color_pair(1))
		else:
			win.addstr(Y//2 + i - len(menu)//2, int(X/2) - len(row)//2, row)


current_row = 0
while key != ESC:
	win.timeout(444)

	draw_menu(win, current_row)

	event = win.getch()
	key = event

	if key == curses.KEY_UP:
		current_row = (current_row - 1) % len(menu)
	elif key == curses.KEY_DOWN:
		current_row = (current_row + 1) % len(menu)
	elif key == ord('\n') or key == ord(' '):
		if current_row == len(menu) - 1:
			break
		elif current_row == 0:
			Air_hockey(win, X, Y, False)
			win.clear()
			win.border(0)
		elif current_row == 1:
			f = open('AI/renders/player_7450.json',)
			buf1 = json.load(f)
			print(type(buf1), buf1)
			f2 = open('AI/renders/player2_7450.json',)
			buf2 = json.load(f2)
			Air_hockey(win, X, Y, False, buf1, buf2)
			win.clear()
			win.border(0)
		elif current_row == 2:
			random_game = Random_game()
			buf1, buf2 = random_game.render()
			print(buf1)
			Air_hockey(win, X, Y, False, buf1, buf2)
			win.clear()
			win.border(0)
		elif current_row == 3:
			Air_hockey(win, X, Y, True)
			win.clear()
			win.border(0)



		


curses.endwin()
