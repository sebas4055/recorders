from flask import Flask, render_template, jsonify, request
import csv
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/weightroom')
def weightroom():
    return render_template('weightroom.html')

@app.route('/get_total')
def get_total():
    try:
        with open('total.csv', 'r') as file:
            reader = csv.reader(file)
            total = next(reader)[0]
    except Exception:
        total = 0
    return jsonify({'total': total})

@app.route('/get_chart_data')
def get_chart_data():
    period = request.args.get('period', 'week')
    if period == 'week':
        labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    elif period == 'month':
        labels = [f'Week {i}' for i in range(1, 5)]
    elif period == 'year':
        labels = [f'Month {i}' for i in range(1, 13)]
    else:
        labels = []
    values = [random.randint(0, 50) for _ in labels]
    return jsonify({'labels': labels, 'values': values})

if __name__ == '__main__':
    app.run(debug=True)

