from flask import Flask, Response, render_template
import cv2
from dbconnect import connection
c, conn = connection()

print(c.execute("SELECT * FROM example"))
c.execute('INSERT INTO example (id,data) VALUES (3,"hi")')
conn.commit()
c.close()
conn.close()
exit()

app = Flask(__name__)

@app.route('/home')
def home():
    return 'halaman home'
app.add_url_rule('/','home',home)

@app.route('/registrasi')
def registrasi():
    return 'halaman registrasi'

@app.route('/log')
def log():
    return 'halaman log'

@app.route('/list')
def list():
    return 'halaman list'

    # return Response('lol swkswk', mimetype="text/event-stream")
    # return render_template("index.html")
if __name__ == '__main__':
    app.run(debug=True)