import MySQLdb

def connection():
	conn = MySQLdb.connect(host="localhost", user="root", passwd="er2c", db="mydatabase")
	c = conn.cursor()
	return c, conn
