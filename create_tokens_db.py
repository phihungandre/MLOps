import sqlite3

conn = sqlite3.connect('tokens.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE tokens (token TEXT PRIMARY KEY)''')
cursor.execute('''INSERT INTO tokens (token) VALUES ('test-token')''')
conn.commit()
conn.close()